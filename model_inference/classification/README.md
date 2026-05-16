# YOLO 分类推理

`YoloClassificationInference` 是 10 类 YOLO 分类模型的 Android C++ 推理封装。

## 对外接口

```cpp
qnn_inference::YoloClassificationInference classifier;
classifier.init(backend_so, model_so);

qnn_inference::ClassificationResult result = classifier.infer(image);
```

其中 `image` 是 OpenCV `cv::Mat`，按 BGR 输入即可。

## 预处理

当前实现会自动完成：

- 支持 `BGR`、`BGRA`、灰度图输入
- resize 到模型输入 tensor 尺寸
- BGR 转 RGB
- 归一化到 `[0, 1]`
- 根据 QNN input tensor 的量化参数写入 `uint8` / `ufixed8` buffer
- 同时兼容 NHWC 和 NCHW 输入布局

这个预处理应与量化校准脚本保持一致，否则量化模型精度会明显下降。

## 后处理

当前模型输出为 10 类分数，后处理会：

- 将 QNN native output 反量化为 `float`
- 查找 Top1 类别
- 返回 `class_id`、`label`、`score` 和完整 `scores`

默认类别名为 `class_0` 到 `class_9`。如果有真实标签，直接修改 `class_names_` 即可。

## 测试程序

`test_classify.cpp` 演示了完整调用流程：

```bash
./test_classify <qnn_backend_so> <qnn_model_so> <image_path>
```

输出包含：

- `Init time`：QNN 初始化和 graph finalize 时间
- `Infer time`：预处理、`graphExecute`、后处理总耗时
- `Top1`：最高分类别
- `Scores`：全部 10 类分数
