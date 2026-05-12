# 高通 QNN 模型转化及量化加速实践

基于 Qualcomm AI Engine Direct (QAIRT) SDK，将深度学习模型转换为 QNN 量化格式并部署到 Android 设备的完整工程实践。

---

## 目录

- [项目简介](#项目简介)
- [项目结构](#项目结构)
- [环境要求](#环境要求)
- [模型转化及量化](#模型转化及量化)
- [模型推理部署（待完成）](#模型推理部署待完成)

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
    QNN 量化模型 (.so)  →  部署到 Android 设备
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
├── model_inference/                   # 模型推理（待完成）
└── model_int8_convert/                # INT8 转换（待完成）
```

---

## 环境要求

| 组件 | 版本要求 | 说明 |
|------|---------|------|
| 操作系统 | Ubuntu 22.04 (Docker) | 推荐在 Docker 容器中操作 |
| Python | 3.10 | QNN SDK 兼容性最佳 |
| Qualcomm AI Engine Direct SDK | 2.44.0+ | 提供 QNN 工具链 |
| Android NDK | r26c | 编译 Android `.so` 库 |
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

## 模型推理部署（待完成）

> **TODO**：此部分将包含以下内容：
> - Android 端 QNN Runtime 集成
> - `.so` 模型加载与推理接口调用
> - 前后处理流程（预处理、NMS 后处理等）
> - 性能基准测试与优化
> - 端到端 Demo 应用