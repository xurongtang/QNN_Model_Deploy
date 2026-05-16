# QNN C++ 模型推理基类

`QnnModelInferenceBase` 封装了高通 QNN 模型推理的通用流程：

1. 动态加载 QNN backend `.so`
2. 动态加载 `qnn-model-lib-generator` 生成的模型 `.so`
3. 调用模型库导出的 `QnnModel_composeGraphs`
4. 创建 logger/backend/device/context，finalize graph
5. 为输入输出 tensor 分配 buffer
6. 执行 `graphExecute`

后续新增高通模型推理类时，只需要继承该基类，实现 `preprocess()` 和 `postprocess()`。

## 设计边界

基类只处理 QNN Runtime 通用生命周期，不包含业务相关图像处理和结果解析：

- 负责：动态库加载、QNN interface 获取、backend/device/context、graph compose/finalize、tensor buffer 生命周期、错误信息。
- 不负责：图片 resize/normalize、NMS、分类 topk、业务类别名。

派生类通过 `TensorBuffer` 访问输入输出。输入 buffer 在 `preprocess()` 中写入，输出 buffer 在 `postprocess()` 中读取。

## HTP 兼容处理

HTP backend 在不同设备固件上可能要求明确选择 signed/unsigned process domain。基类会在默认 `deviceCreate` 失败后自动尝试：

1. `signedPD=false`
2. `signedPD=true`

同时会创建静默 QNN logger，避免 backend 将大量 HTP prepare warning 直接打印到终端。初始化失败时，`lastError()` 会保留关键 trace，例如 `backendCreate`、`deviceCreate` 和 `contextCreate` 的返回码。

## 使用示例

```cpp
#include "model_inference/model_inference_base/qnn_model_inference_base.h"

#include <cstring>

class ClassifyQnnInference : public qnn_inference::QnnModelInferenceBase {
 protected:
  bool preprocess(std::vector<TensorBuffer>& inputs) override {
    auto& input = inputs[0];

    // 在这里做 resize / normalize / layout 转换 / 量化等预处理。
    // 转换后的数据直接写入 input.data，大小不能超过 input.byte_size。
    std::memset(input.data, 0, input.byte_size);
    return true;
  }

  bool postprocess(const std::vector<TensorBuffer>& outputs) override {
    const auto& output = outputs[0];

    // 在这里读取 output.data，结合 output.dims / output.data_type 做反量化、topk、NMS 等后处理。
    (void)output;
    return true;
  }
};

int main() {
  ClassifyQnnInference inference;

  qnn_inference::QnnModelInferenceBase::RuntimeConfig config;
  config.backend_library_path = "/data/local/tmp/qnn/libQnnHtp.so";
  config.model_library_path = "/data/local/tmp/qnn/libyolo11n_cls_qnn_quantized_model.so";

  if (!inference.loadModel(config)) {
    return -1;
  }

  return inference.infer() ? 0 : -1;
}
```

## 编译要点

需要包含 QNN SDK 头文件、HTP backend 头文件和 converter jni wrapper 头文件，例如：

```bash
-I${QNN_SDK_ROOT}/include/QNN
-I${QNN_SDK_ROOT}/share/QNN/converter/jni
```

链接时需要带上 `dl`：

```bash
-ldl
```

Android 端部署时，`backend_library_path` 通常指向 `libQnnHtp.so`、`libQnnGpu.so` 或 `libQnnCpu.so`，`model_library_path` 指向转换阶段生成的模型库 `.so`。

HTP 运行还需要 `LD_LIBRARY_PATH` 能找到 ARM 侧 `libQnnHtp*.so`，`ADSP_LIBRARY_PATH` 能找到 DSP 侧 `libQnnHtpV*Skel.so`。
