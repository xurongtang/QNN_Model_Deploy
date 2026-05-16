# 模型推理模块

`model_inference` 是 Android C++ 端推理代码目录，当前包含两部分：

- `model_inference_base/`：QNN 推理基类，封装 backend、model、graph 和 tensor buffer 的通用生命周期。
- `classification/`：YOLO 分类模型推理封装，对外暴露 `init()` 和 `infer(cv::Mat)`。

## 调用关系

```text
test_classify
    │
    ▼
YoloClassificationInference
    │  preprocess(cv::Mat -> QNN input)
    ▼
QnnModelInferenceBase
    │  QNN graphExecute
    ▼
YoloClassificationInference
       postprocess(QNN output -> ClassificationResult)
```

## 新增模型类型

新增检测、分割或其他模型时，建议复用 `QnnModelInferenceBase`：

1. 新建业务目录，例如 `object_detection/`。
2. 定义对外接口，例如 `init()`、`infer(cv::Mat)`。
3. 继承 `QnnModelInferenceBase`。
4. 在 `preprocess()` 中把业务输入写入 QNN input tensor。
5. 在 `postprocess()` 中读取 QNN output tensor 并转换为业务结果。

这样可以避免每个模型重复处理 QNN backend/model 动态加载和 graph 执行细节。

## 运行依赖

Android 设备侧至少需要：

- 编译出的业务可执行程序，例如 `test_classify`
- QNN backend：`libQnnHtp.so` 或 `libQnnCpu.so`
- 转换生成的模型库：`lib*_qnn_quantized_model.so`
- OpenCV Android 运行库
- HTP backend 对应的 `Stub` 和 `Skel` 库

SM8650 / Snapdragon 8 Gen 3 推荐使用 `hexagon-v75/unsigned` 下的 HTP skel。
