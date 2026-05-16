#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "QnnInterface.h"
#include "QnnWrapperUtils.hpp"

namespace qnn_inference {

class QnnModelInferenceBase {
 public:
  struct RuntimeConfig {
    // QNN 后端运行库，例如 libQnnHtp.so 或 libQnnCpu.so。
    std::string backend_library_path;
    // qnn-model-lib-generator 生成的模型动态库。
    std::string model_library_path;
    bool enable_debug_tensors = false;
  };

  struct TensorBuffer {
    std::string name;
    std::vector<uint32_t> dims;
    Qnn_DataType_t data_type = QNN_DATATYPE_UNDEFINED;
    size_t byte_size = 0;
    void* data = nullptr;
    // scale-offset 量化张量会填入对应的 scale 和 offset。
    float quant_scale = 0.0f;
    int32_t quant_offset = 0;
  };

  QnnModelInferenceBase() = default;
  virtual ~QnnModelInferenceBase();

  QnnModelInferenceBase(const QnnModelInferenceBase&) = delete;
  QnnModelInferenceBase& operator=(const QnnModelInferenceBase&) = delete;

  bool loadModel(const RuntimeConfig& config);
  void release();

  // 完整推理流程：预处理 -> graphExecute -> 后处理。
  bool infer(uint32_t graph_index = 0);
  // 低层图执行接口，适合调用方自己读写 tensor buffer。
  bool execute(uint32_t graph_index = 0);

  bool isLoaded() const { return loaded_; }
  const std::string& lastError() const { return last_error_; }

  uint32_t graphCount() const { return graphs_count_; }
  const char* graphName(uint32_t graph_index = 0) const;

  const std::vector<TensorBuffer>& inputTensors(uint32_t graph_index = 0) const;
  const std::vector<TensorBuffer>& outputTensors(uint32_t graph_index = 0) const;
  std::vector<TensorBuffer>& mutableInputTensors(uint32_t graph_index = 0);

 protected:
  // 派生类在这里把预处理后的数据写入 inputs[i].data。
  virtual bool preprocess(std::vector<TensorBuffer>& inputs) = 0;
  // 派生类在这里读取 outputs[i].data 并转换为业务结果。
  virtual bool postprocess(const std::vector<TensorBuffer>& outputs) = 0;

  void setLastError(const std::string& error);

 private:
  using ComposeGraphsFn = qnn_wrapper_api::ModelError_t (*)(
      Qnn_BackendHandle_t,
      QNN_INTERFACE_VER_TYPE,
      Qnn_ContextHandle_t,
      const qnn_wrapper_api::GraphConfigInfo_t**,
      const uint32_t,
      qnn_wrapper_api::GraphInfo_t***,
      uint32_t*,
      bool,
      QnnLog_Callback_t,
      QnnLog_Level_t);

  using FreeGraphsInfoFn =
      qnn_wrapper_api::ModelError_t (*)(qnn_wrapper_api::GraphInfo_t***, uint32_t);

  struct RuntimeSymbols {
    ComposeGraphsFn compose_graphs = nullptr;
    FreeGraphsInfoFn free_graphs_info = nullptr;
    QNN_INTERFACE_VER_TYPE qnn = {};
  };

  struct GraphExecutionBuffers {
    std::vector<Qnn_Tensor_t> qnn_inputs;
    std::vector<Qnn_Tensor_t> qnn_outputs;
    std::vector<std::vector<uint8_t>> input_storage;
    std::vector<std::vector<uint8_t>> output_storage;
    std::vector<TensorBuffer> inputs;
    std::vector<TensorBuffer> outputs;
  };

  bool loadBackendLibrary(const std::string& backend_library_path);
  bool loadModelLibrary(const std::string& model_library_path);
  bool initializeLogger();
  bool initializeBackend();
  bool createDevice();
  bool createContext();
  bool composeGraphs();
  bool finalizeGraphs();
  bool prepareExecutionBuffers();

  bool prepareGraphTensors(qnn_wrapper_api::GraphInfo_t& graph_info,
                           GraphExecutionBuffers* buffers);
  bool copyTensorForExecution(const Qnn_Tensor_t& source,
                              Qnn_Tensor_t* target,
                              std::vector<uint8_t>* storage,
                              TensorBuffer* tensor_buffer);

  static size_t tensorByteSize(const Qnn_Tensor_t& tensor);
  static size_t dataTypeByteSize(Qnn_DataType_t data_type);
  static TensorBuffer describeTensor(const Qnn_Tensor_t& tensor, void* data, size_t byte_size);
  static std::vector<uint32_t> tensorDims(const Qnn_Tensor_t& tensor);
  static void* resolveSymbol(void* library_handle, const char* symbol_name);

  bool checkGraphIndex(uint32_t graph_index);

  RuntimeConfig config_;
  RuntimeSymbols symbols_;

  void* backend_library_handle_ = nullptr;
  void* model_library_handle_ = nullptr;

  Qnn_BackendHandle_t backend_handle_ = nullptr;
  Qnn_DeviceHandle_t device_handle_ = nullptr;
  Qnn_ContextHandle_t context_handle_ = nullptr;
  Qnn_LogHandle_t log_handle_ = nullptr;

  qnn_wrapper_api::GraphInfo_t** graphs_info_ = nullptr;
  uint32_t graphs_count_ = 0;
  std::vector<GraphExecutionBuffers> execution_buffers_;

  bool backend_initialized_ = false;
  bool device_created_ = false;
  bool context_created_ = false;
  bool loaded_ = false;
  std::string init_trace_;
  std::string last_error_;
};

}  // 命名空间 qnn_inference
