# 高通 QNN 模型转化及量化加速实践

基于 Qualcomm AI Engine Direct (QAIRT) SDK，将深度学习模型转换为 QNN 量化格式并部署到 Android 设备的完整工程实践。

---

## 目录

- [项目简介](#项目简介)
- [项目结构](#项目结构)
- [环境要求](#环境要求)
- [模型转化及量化](#模型转化及量化)
- [模型推理部署](#模型推理部署)

---

## 项目简介

本项目旨在实现以下工作流：

```
PyTorch 模型 (.pt)
        │
        ▼  步骤一：导出 ONNX
    ONNX 模型 (.onnx)
        │
        ▼  步骤二：构建校准数据集
    校准数据集 (.raw + datasets.txt)
        │
        ▼  步骤三：QNN 量化 + 编译
    QNN 量化模型 (.so)
        │
        ▼  步骤四：Android C++ 推理
    test_classify / 业务推理模块
```

**核心技术栈**：

- **Qualcomm AI Engine Direct SDK** — QNN 模型转换与量化工具链
- **Android NDK r26c** — 交叉编译 Android 动态库
- **ONNX** — 通用模型中间表示格式
- **Ultralytics YOLO** — 示例模型框架（YOLO11n-cls）

---

## 项目结构

```
QNN_proj/
├── readme.md                          # 本文件 — 项目总览
├── CMakeLists.txt                     # Android 端 C++ 推理构建配置
├── build.sh                           # Android arm64-v8a 一键编译脚本
├── test_classify.cpp                  # 10 类分类模型推理测试入口
├── 步骤.txt                           # 环境搭建步骤速查
├── model_convert/                     # 模型转换相关文件
│   ├── readme.md                      # 详细的模型转换教程
│   ├── pt2onnx/                       # 步骤一：PyTorch → ONNX
│   │   ├── pt2onnx.py                 #   ONNX 导出脚本
│   │   ├── yolo11n-cls.pt             #   原始 PyTorch 模型
│   │   └── yolo11n-cls.onnx           #   导出的 ONNX 模型
│   └── onnx2so/                       # 步骤二/三：ONNX → QNN .so
│       ├── convert.sh                 #   QNN 量化转换 + 编译脚本
│       ├── create_dataset.py          #   校准数据集生成脚本
│       ├── datasets.txt               #   校准数据文件列表
│       ├── dataset_img/               #   原始校准图片目录
│       ├── dataset_raw/               #   预处理后的 .raw 文件
│       └── output/                    #   QNN 量化模型输出
├── model_inference/                   # Android C++ 模型推理
│   ├── model_inference_base/          #   QNN backend/model/graph 通用封装
│   └── classification/                #   YOLO 分类模型推理封装
└── model_int8_convert/                # INT8 转换（待完成）
```

---

## 环境要求

| 组件 | 版本要求 | 说明 |
|------|---------|------|
| 操作系统 | Ubuntu 22.04 (Docker) | 推荐在 Docker 容器中操作 |
| Python | 3.10 | QNN SDK 兼容性最佳 |
| Qualcomm AI Engine Direct SDK | 2.44.0+ | 提供 QNN 工具链 |
| Android NDK | r26c | 编译 Android 模型推理程序 |
| OpenCV Android SDK | 4.x | 图像读取、resize、颜色转换 |
| PyTorch | 1.13.1 | 模型导出 |
| ONNX | 1.17.0 | 模型中间格式 |
| NumPy | 1.26.4 | 注意版本兼容性 |

> 📖 **详细的环境搭建教程**请参阅 [`model_convert/readme.md`](model_convert/readme.md)，涵盖 Docker 环境搭建、SDK 依赖安装、Python 虚拟环境配置、NDK 配置及常见问题排查。

---

## 模型转化及量化

完整的模型转化及量化流程分为三个步骤，每个步骤对应项目中的脚本和工具：

### 步骤一：PyTorch → ONNX

**脚本**：[`model_convert/pt2onnx/pt2onnx.py`](model_convert/pt2onnx/pt2onnx.py)

```bash
cd model_convert/pt2onnx
python3 pt2onnx.py
```

将训练好的 YOLO11n-cls `.pt` 模型导出为 ONNX 格式。关键配置：固定输入尺寸（`dynamic=False`）、简化计算图（`simplify=True`）、opset 13。

### 步骤二：构建校准数据集

**脚本**：[`model_convert/onnx2so/create_dataset.py`](model_convert/onnx2so/create_dataset.py)

```bash
cd model_convert/onnx2so
python3 create_dataset.py
```

从 `dataset_img/` 读取校准图片，经预处理后生成 NHWC float32 格式的 `.raw` 文件，并输出 `datasets.txt` 供 QNN 量化使用。

### 步骤三：ONNX → QNN 量化模型 (.so)

**脚本**：[`model_convert/onnx2so/convert.sh`](model_convert/onnx2so/convert.sh)

```bash
cd model_convert/onnx2so
bash convert.sh
```

分两个阶段执行：
1. **量化**：使用 `qnn-onnx-converter` 将 ONNX 模型转换为 INT8 量化模型（8-bit 权重 + 8-bit 激活，逐通道量化 + CLE 算法优化）
2. **编译**：使用 `qnn-model-lib-generator` 将量化模型编译为 `aarch64-android` 架构的 `.so` 动态库

> 📖 **每个步骤的详细参数说明、配置方法和常见问题排查**请参阅 [`model_convert/readme.md`](model_convert/readme.md)。

---

## 模型推理部署

当前已实现 Android C++ 端 QNN 推理链路：

- `QnnModelInferenceBase`：封装 QNN backend/model 动态加载、backend/device/context 创建、graph compose/finalize、tensor buffer 分配和 `graphExecute`。
- `YoloClassificationInference`：继承基类，完成 `cv::Mat` 输入的 YOLO 分类预处理和 10 类输出后处理。
- `test_classify`：命令行测试程序，输出初始化耗时、单次推理耗时、Top1 和 10 类分数。

### 编译

项目内置路径默认指向 `3rdlibrary` 下的 Android NDK、OpenCV Android SDK 和 QNN SDK：

```bash
./build.sh
```

生成文件：

```bash
build/android-arm64-v8a/test_classify
```

### 设备部署

以 HTP 后端为例，需要同时部署 ARM 侧 QNN runtime、模型 `.so`、测试程序，以及 DSP 侧 `hexagon-v*` 的 `Skel` 库。SM8650 / Snapdragon 8 Gen 3 对应 `hexagon-v75/unsigned`。

设备侧运行示例：

```bash
cd /data/local/tmp/test
export LD_LIBRARY_PATH=$PWD:../common_lib:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH="../common_lib;$PWD;/vendor/lib64/rfsa/adsp;/vendor/dsp/cdsp;/system/lib/rfsa/adsp"
./test_classify ../common_lib/libQnnHtp.so ./libyolo26n-cls_qnn_quantized_model.so ./ILSVRC2012_val_00002138.JPEG
```

输出示例：

```text
houji:/data/local/tmp/test $ ./running.sh
Init time: 724.035417 ms
Infer time: 6.519896 ms
Top1: class_id=0, label=class_0, score=0.996094
Scores: [0]=0.996094 [1]=0.000000 [2]=0.003906 [3]=0.000000 [4]=0.000000 [5]=0.000000 [6]=0.000000 [7]=0.000000 [8]=0.000000 [9]=0.000000
```