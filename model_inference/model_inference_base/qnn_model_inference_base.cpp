#include "qnn_model_inference_base.h"

#include <dlfcn.h>

#include <cstdarg>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>

#include "HTP/QnnHtpDevice.h"

namespace qnn_inference {
namespace {

using QnnInterfaceGetProvidersFn =
    Qnn_ErrorHandle_t (*)(const QnnInterface_t*** provider_list, uint32_t* num_providers);

const char* tensorName(const Qnn_Tensor_t& tensor) {
  if (tensor.version == QNN_TENSOR_VERSION_2) {
    return tensor.v2.name;
  }
  return tensor.v1.name;
}

Qnn_DataType_t tensorDataType(const Qnn_Tensor_t& tensor) {
  if (tensor.version == QNN_TENSOR_VERSION_2) {
    return tensor.v2.dataType;
  }
  return tensor.v1.dataType;
}

uint32_t tensorRank(const Qnn_Tensor_t& tensor) {
  if (tensor.version == QNN_TENSOR_VERSION_2) {
    return tensor.v2.rank;
  }
  return tensor.v1.rank;
}

uint32_t* tensorDimensions(const Qnn_Tensor_t& tensor) {
  if (tensor.version == QNN_TENSOR_VERSION_2) {
    return tensor.v2.dimensions;
  }
  return tensor.v1.dimensions;
}

Qnn_QuantizeParams_t tensorQuantParams(const Qnn_Tensor_t& tensor) {
  if (tensor.version == QNN_TENSOR_VERSION_2) {
    return tensor.v2.quantizeParams;
  }
  return tensor.v1.quantizeParams;
}

void setRawClientBuffer(Qnn_Tensor_t* tensor, void* data, size_t byte_size) {
  Qnn_ClientBuffer_t client_buffer = QNN_CLIENT_BUFFER_INIT;
  client_buffer.data = data;
  client_buffer.dataSize = static_cast<uint32_t>(byte_size);

  // 转换器生成的 graph info 里是包装 tensor；真正执行前需要挂接应用侧
  // 自己持有的 RAW client buffer。
  if (tensor->version == QNN_TENSOR_VERSION_2) {
    tensor->v2.memType = QNN_TENSORMEMTYPE_RAW;
    tensor->v2.clientBuf = client_buffer;
    return;
  }

  tensor->v1.memType = QNN_TENSORMEMTYPE_RAW;
  tensor->v1.clientBuf = client_buffer;
}

// 创建静默 logger，避免 QNN 回退到默认日志并打印大量 HTP prepare warning。
void silentQnnLogCallback(const char*, QnnLog_Level_t, uint64_t, va_list) {}

bool containsHtp(const std::string& path) {
  return path.find("Htp") != std::string::npos || path.find("HTP") != std::string::npos ||
         path.find("htp") != std::string::npos;
}

}  // 匿名命名空间

QnnModelInferenceBase::~QnnModelInferenceBase() {
  release();
}

bool QnnModelInferenceBase::loadModel(const RuntimeConfig& config) {
  release();
  config_ = config;
  init_trace_.clear();

  if (config_.backend_library_path.empty()) {
    setLastError("backend_library_path is empty");
    return false;
  }
  if (config_.model_library_path.empty()) {
    setLastError("model_library_path is empty");
    return false;
  }

  // 初始化顺序对齐官方 SampleApp：
  // backend/model 动态库 -> logger -> backend/device/context -> graph。
  if (!loadBackendLibrary(config_.backend_library_path) ||
      !loadModelLibrary(config_.model_library_path) ||
      !initializeLogger() ||
      !initializeBackend() ||
      !createDevice() ||
      !createContext() ||
      !composeGraphs() ||
      !finalizeGraphs() ||
      !prepareExecutionBuffers()) {
    const std::string error = last_error_;
    release();
    last_error_ = error;
    return false;
  }

  loaded_ = true;
  return true;
}

void QnnModelInferenceBase::release() {
  execution_buffers_.clear();

  // graph metadata 持有 graph handle，需要在销毁 context 前释放。
  if (graphs_info_ != nullptr && symbols_.free_graphs_info != nullptr) {
    symbols_.free_graphs_info(&graphs_info_, graphs_count_);
  }
  graphs_info_ = nullptr;
  graphs_count_ = 0;

  if (context_created_ && context_handle_ != nullptr && symbols_.qnn.contextFree != nullptr) {
    symbols_.qnn.contextFree(context_handle_, nullptr);
  }
  context_handle_ = nullptr;
  context_created_ = false;

  if (device_created_ && device_handle_ != nullptr && symbols_.qnn.deviceFree != nullptr) {
    symbols_.qnn.deviceFree(device_handle_);
  }
  device_handle_ = nullptr;
  device_created_ = false;

  if (backend_initialized_ && backend_handle_ != nullptr && symbols_.qnn.backendFree != nullptr) {
    symbols_.qnn.backendFree(backend_handle_);
  }
  backend_handle_ = nullptr;
  backend_initialized_ = false;

  if (log_handle_ != nullptr && symbols_.qnn.logFree != nullptr) {
    symbols_.qnn.logFree(log_handle_);
  }
  log_handle_ = nullptr;

  if (model_library_handle_ != nullptr) {
    dlclose(model_library_handle_);
  }
  model_library_handle_ = nullptr;

  if (backend_library_handle_ != nullptr) {
    dlclose(backend_library_handle_);
  }
  backend_library_handle_ = nullptr;

  symbols_ = RuntimeSymbols{};
  init_trace_.clear();
  loaded_ = false;
}

bool QnnModelInferenceBase::infer(uint32_t graph_index) {
  if (!checkGraphIndex(graph_index)) {
    return false;
  }

  auto& buffers = execution_buffers_[graph_index];
  if (!preprocess(buffers.inputs)) {
    if (last_error_.empty()) {
      setLastError("preprocess failed");
    }
    return false;
  }

  if (!execute(graph_index)) {
    return false;
  }

  if (!postprocess(buffers.outputs)) {
    if (last_error_.empty()) {
      setLastError("postprocess failed");
    }
    return false;
  }

  return true;
}

bool QnnModelInferenceBase::execute(uint32_t graph_index) {
  if (!checkGraphIndex(graph_index)) {
    return false;
  }

  auto& graph_info = (*graphs_info_)[graph_index];
  auto& buffers = execution_buffers_[graph_index];
  const Qnn_ErrorHandle_t status = symbols_.qnn.graphExecute(
      graph_info.graph,
      buffers.qnn_inputs.data(),
      graph_info.numInputTensors,
      buffers.qnn_outputs.data(),
      graph_info.numOutputTensors,
      nullptr,
      nullptr);
  if (status != QNN_GRAPH_NO_ERROR) {
    std::ostringstream oss;
    oss << "QNN graphExecute failed, graph=" << graph_info.graphName << ", status=" << status;
    setLastError(oss.str());
    return false;
  }

  return true;
}

const char* QnnModelInferenceBase::graphName(uint32_t graph_index) const {
  if (graphs_info_ == nullptr || graph_index >= graphs_count_) {
    return nullptr;
  }
  return (*graphs_info_)[graph_index].graphName;
}

const std::vector<QnnModelInferenceBase::TensorBuffer>& QnnModelInferenceBase::inputTensors(
    uint32_t graph_index) const {
  return execution_buffers_.at(graph_index).inputs;
}

const std::vector<QnnModelInferenceBase::TensorBuffer>& QnnModelInferenceBase::outputTensors(
    uint32_t graph_index) const {
  return execution_buffers_.at(graph_index).outputs;
}

std::vector<QnnModelInferenceBase::TensorBuffer>& QnnModelInferenceBase::mutableInputTensors(
    uint32_t graph_index) {
  return execution_buffers_.at(graph_index).inputs;
}

void QnnModelInferenceBase::setLastError(const std::string& error) {
  last_error_ = error;
}

bool QnnModelInferenceBase::loadBackendLibrary(const std::string& backend_library_path) {
  backend_library_handle_ = dlopen(backend_library_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (backend_library_handle_ == nullptr) {
    setLastError(std::string("failed to load QNN backend library: ") + dlerror());
    return false;
  }

  auto get_providers = reinterpret_cast<QnnInterfaceGetProvidersFn>(
      resolveSymbol(backend_library_handle_, "QnnInterface_getProviders"));
  if (get_providers == nullptr) {
    setLastError("failed to resolve QnnInterface_getProviders");
    return false;
  }

  const QnnInterface_t** providers = nullptr;
  uint32_t provider_count = 0;
  if (get_providers(&providers, &provider_count) != QNN_SUCCESS || providers == nullptr ||
      provider_count == 0) {
    setLastError("failed to get QNN interface providers");
    return false;
  }

  for (uint32_t i = 0; i < provider_count; ++i) {
    const Qnn_ApiVersion_t& version = providers[i]->apiVersion;
    if (version.coreApiVersion.major == QNN_API_VERSION_MAJOR &&
        version.coreApiVersion.minor >= QNN_API_VERSION_MINOR) {
      symbols_.qnn = providers[i]->QNN_INTERFACE_VER_NAME;
      return true;
    }
  }

  setLastError("no compatible QNN interface provider found");
  return false;
}

bool QnnModelInferenceBase::loadModelLibrary(const std::string& model_library_path) {
  model_library_handle_ = dlopen(model_library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (model_library_handle_ == nullptr) {
    setLastError(std::string("failed to load QNN model library: ") + dlerror());
    return false;
  }

  symbols_.compose_graphs = reinterpret_cast<ComposeGraphsFn>(
      resolveSymbol(model_library_handle_, "QnnModel_composeGraphs"));
  symbols_.free_graphs_info = reinterpret_cast<FreeGraphsInfoFn>(
      resolveSymbol(model_library_handle_, "QnnModel_freeGraphsInfo"));

  if (symbols_.compose_graphs == nullptr || symbols_.free_graphs_info == nullptr) {
    setLastError("model library does not export QnnModel_composeGraphs/QnnModel_freeGraphsInfo");
    return false;
  }

  return true;
}

bool QnnModelInferenceBase::initializeLogger() {
  if (symbols_.qnn.logCreate == nullptr) {
    init_trace_ += "logCreate=not_available; ";
    return true;
  }

  const Qnn_ErrorHandle_t status =
      symbols_.qnn.logCreate(silentQnnLogCallback, QNN_LOG_LEVEL_ERROR, &log_handle_);
  init_trace_ += "logCreate=" + std::to_string(status) + "; ";
  if (status != QNN_SUCCESS && status != QNN_COMMON_ERROR_NOT_SUPPORTED) {
    std::ostringstream oss;
    oss << "QNN logCreate failed, status=" << status;
    setLastError(oss.str());
    return false;
  }

  return true;
}

bool QnnModelInferenceBase::initializeBackend() {
  if (symbols_.qnn.backendCreate == nullptr) {
    setLastError("QNN backendCreate is null");
    return false;
  }

  const Qnn_ErrorHandle_t status = symbols_.qnn.backendCreate(log_handle_, nullptr, &backend_handle_);
  init_trace_ += "backendCreate=" + std::to_string(status) + "; ";
  if (status != QNN_BACKEND_NO_ERROR) {
    std::ostringstream oss;
    oss << "QNN backendCreate failed, status=" << status;
    setLastError(oss.str());
    return false;
  }

  backend_initialized_ = true;
  return true;
}

bool QnnModelInferenceBase::createDevice() {
  if (symbols_.qnn.deviceCreate == nullptr) {
    init_trace_ += "deviceCreate=not_available; ";
    return true;
  }

  Qnn_ErrorHandle_t status = symbols_.qnn.deviceCreate(log_handle_, nullptr, &device_handle_);
  init_trace_ += "deviceCreate(default)=" + std::to_string(status) + "; ";

  if (status != QNN_SUCCESS) {
    for (const bool use_signed_pd : {false, true}) {
      QnnHtpDevice_CustomConfig_t htp_config = {};
      htp_config.option = QNN_HTP_DEVICE_CONFIG_OPTION_SIGNEDPD;
      htp_config.useSignedProcessDomain.deviceId = 0;
      htp_config.useSignedProcessDomain.useSignedProcessDomain = use_signed_pd;

      // 部分 HTP 设备需要显式指定 PD 类型。先尝试 unsigned，再尝试 signed，
      // 以兼容普通固件和 signed 固件。
      QnnDevice_Config_t device_config = QNN_DEVICE_CONFIG_INIT;
      device_config.option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
      device_config.customConfig = &htp_config;

      const QnnDevice_Config_t* device_configs[] = {&device_config, nullptr};
      device_handle_ = nullptr;
      status = symbols_.qnn.deviceCreate(log_handle_, device_configs, &device_handle_);
      init_trace_ += std::string("deviceCreate(signedPD=") + (use_signed_pd ? "true" : "false") +
                     ")=" + std::to_string(status) + "; ";
      if (status == QNN_SUCCESS) {
        break;
      }
    }
  }

  if (status == QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE) {
    if (containsHtp(config_.backend_library_path)) {
      std::ostringstream oss;
      oss << "QNN HTP deviceCreate is unsupported; " << init_trace_;
      setLastError(oss.str());
      return false;
    }
    device_handle_ = nullptr;
    device_created_ = false;
    return true;
  }
  if (status != QNN_SUCCESS) {
    std::ostringstream oss;
    oss << "QNN deviceCreate failed, status=" << status << "; " << init_trace_;
    setLastError(oss.str());
    return false;
  }

  device_created_ = device_handle_ != nullptr;
  return true;
}

bool QnnModelInferenceBase::createContext() {
  if (symbols_.qnn.contextCreate == nullptr) {
    setLastError("QNN contextCreate is null");
    return false;
  }

  const Qnn_ErrorHandle_t status =
      symbols_.qnn.contextCreate(backend_handle_, device_handle_, nullptr, &context_handle_);
  init_trace_ += "contextCreate=" + std::to_string(status) + "; ";
  if (status != QNN_CONTEXT_NO_ERROR) {
    std::ostringstream oss;
    oss << "QNN contextCreate failed, status=" << status << "; " << init_trace_;
    setLastError(oss.str());
    return false;
  }

  context_created_ = true;
  return true;
}

bool QnnModelInferenceBase::composeGraphs() {
  if (symbols_.compose_graphs == nullptr) {
    setLastError("QnnModel_composeGraphs is null");
    return false;
  }

  const auto status = symbols_.compose_graphs(backend_handle_,
                                             symbols_.qnn,
                                             context_handle_,
                                             nullptr,
                                             0,
                                             &graphs_info_,
                                             &graphs_count_,
                                             config_.enable_debug_tensors,
                                             nullptr,
                                             QNN_LOG_LEVEL_ERROR);
  if (status != qnn_wrapper_api::MODEL_NO_ERROR || graphs_info_ == nullptr ||
      graphs_count_ == 0) {
    std::ostringstream oss;
    oss << "QnnModel_composeGraphs failed, status=" << status;
    setLastError(oss.str());
    return false;
  }

  return true;
}

bool QnnModelInferenceBase::finalizeGraphs() {
  if (symbols_.qnn.graphFinalize == nullptr) {
    setLastError("QNN graphFinalize is null");
    return false;
  }

  for (uint32_t i = 0; i < graphs_count_; ++i) {
    const Qnn_ErrorHandle_t status =
        symbols_.qnn.graphFinalize((*graphs_info_)[i].graph, nullptr, nullptr);
    if (status != QNN_GRAPH_NO_ERROR) {
      std::ostringstream oss;
      oss << "QNN graphFinalize failed, graph=" << (*graphs_info_)[i].graphName
          << ", status=" << status;
      setLastError(oss.str());
      return false;
    }
  }

  return true;
}

bool QnnModelInferenceBase::prepareExecutionBuffers() {
  execution_buffers_.resize(graphs_count_);
  for (uint32_t i = 0; i < graphs_count_; ++i) {
    if (!prepareGraphTensors((*graphs_info_)[i], &execution_buffers_[i])) {
      return false;
    }
  }
  return true;
}

bool QnnModelInferenceBase::prepareGraphTensors(qnn_wrapper_api::GraphInfo_t& graph_info,
                                                GraphExecutionBuffers* buffers) {
  buffers->qnn_inputs.resize(graph_info.numInputTensors);
  buffers->qnn_outputs.resize(graph_info.numOutputTensors);
  buffers->input_storage.resize(graph_info.numInputTensors);
  buffers->output_storage.resize(graph_info.numOutputTensors);
  buffers->inputs.resize(graph_info.numInputTensors);
  buffers->outputs.resize(graph_info.numOutputTensors);

  for (uint32_t i = 0; i < graph_info.numInputTensors; ++i) {
    if (!copyTensorForExecution(graph_info.inputTensors[i],
                                &buffers->qnn_inputs[i],
                                &buffers->input_storage[i],
                                &buffers->inputs[i])) {
      return false;
    }
  }

  for (uint32_t i = 0; i < graph_info.numOutputTensors; ++i) {
    if (!copyTensorForExecution(graph_info.outputTensors[i],
                                &buffers->qnn_outputs[i],
                                &buffers->output_storage[i],
                                &buffers->outputs[i])) {
      return false;
    }
  }

  return true;
}

bool QnnModelInferenceBase::copyTensorForExecution(const Qnn_Tensor_t& source,
                                                   Qnn_Tensor_t* target,
                                                   std::vector<uint8_t>* storage,
                                                   TensorBuffer* tensor_buffer) {
  const size_t byte_size = tensorByteSize(source);
  if (byte_size == 0) {
    std::ostringstream oss;
    oss << "unsupported or empty tensor: " << (tensorName(source) ? tensorName(source) : "");
    setLastError(oss.str());
    return false;
  }
  if (byte_size > std::numeric_limits<uint32_t>::max()) {
    setLastError("tensor buffer is larger than QNN client buffer supports");
    return false;
  }

  storage->assign(byte_size, 0);
  *target = source;
  setRawClientBuffer(target, storage->data(), byte_size);
  // TensorBuffer 是提供给业务预处理/后处理使用的轻量视图。
  *tensor_buffer = describeTensor(*target, storage->data(), byte_size);
  return true;
}

size_t QnnModelInferenceBase::tensorByteSize(const Qnn_Tensor_t& tensor) {
  const auto dims = tensorDims(tensor);
  const size_t elements =
      std::accumulate(dims.begin(), dims.end(), size_t{1}, std::multiplies<size_t>());
  const size_t data_type_size = dataTypeByteSize(tensorDataType(tensor));
  return elements * data_type_size;
}

size_t QnnModelInferenceBase::dataTypeByteSize(Qnn_DataType_t data_type) {
  switch (data_type) {
    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_SFIXED_POINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8:
    case QNN_DATATYPE_BOOL_8:
      return 1;
    case QNN_DATATYPE_INT_16:
    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_SFIXED_POINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16:
    case QNN_DATATYPE_FLOAT_16:
    case QNN_DATATYPE_BFLOAT_16:
      return 2;
    case QNN_DATATYPE_INT_32:
    case QNN_DATATYPE_UINT_32:
    case QNN_DATATYPE_SFIXED_POINT_32:
    case QNN_DATATYPE_UFIXED_POINT_32:
    case QNN_DATATYPE_FLOAT_32:
      return 4;
    case QNN_DATATYPE_INT_64:
    case QNN_DATATYPE_UINT_64:
    case QNN_DATATYPE_FLOAT_64:
      return 8;
    default:
      return 0;
  }
}

QnnModelInferenceBase::TensorBuffer QnnModelInferenceBase::describeTensor(const Qnn_Tensor_t& tensor,
                                                                          void* data,
                                                                          size_t byte_size) {
  TensorBuffer buffer;
  const char* name = tensorName(tensor);
  buffer.name = name == nullptr ? "" : name;
  buffer.dims = tensorDims(tensor);
  buffer.data_type = tensorDataType(tensor);
  buffer.byte_size = byte_size;
  buffer.data = data;

  const auto quant_params = tensorQuantParams(tensor);
  if (quant_params.encodingDefinition == QNN_DEFINITION_DEFINED &&
      quant_params.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
    buffer.quant_scale = quant_params.scaleOffsetEncoding.scale;
    buffer.quant_offset = quant_params.scaleOffsetEncoding.offset;
  }

  return buffer;
}

std::vector<uint32_t> QnnModelInferenceBase::tensorDims(const Qnn_Tensor_t& tensor) {
  std::vector<uint32_t> dims;
  const uint32_t rank = tensorRank(tensor);
  uint32_t* dimensions = tensorDimensions(tensor);
  dims.reserve(rank);
  for (uint32_t i = 0; i < rank; ++i) {
    dims.push_back(dimensions[i]);
  }
  return dims;
}

void* QnnModelInferenceBase::resolveSymbol(void* library_handle, const char* symbol_name) {
  dlerror();
  return dlsym(library_handle, symbol_name);
}

bool QnnModelInferenceBase::checkGraphIndex(uint32_t graph_index) {
  if (!loaded_ || graphs_info_ == nullptr) {
    setLastError("QNN model is not loaded");
    return false;
  }
  if (graph_index >= graphs_count_ || graph_index >= execution_buffers_.size()) {
    std::ostringstream oss;
    oss << "invalid graph index: " << graph_index << ", graph_count=" << graphs_count_;
    setLastError(oss.str());
    return false;
  }
  return true;
}

}  // 命名空间 qnn_inference
