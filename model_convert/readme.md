# QNN 模型转换教程

本教程详细说明如何从零开始搭建 Qualcomm QNN (Qualcomm Neural Network) 模型转换环境，并将 PyTorch 模型（如 YOLO11n-cls）转换为可在 Android 设备上运行的 QNN 量化 `.so` 库。

---

## 目录

- [整体流程概览](#整体流程概览)
- [1. Docker 环境搭建](#1-docker-环境搭建)
- [2. QNN SDK 依赖安装](#2-qnn-sdk-依赖安装)
- [3. Python 虚拟环境配置](#3-python-虚拟环境配置)
- [4. Python 依赖安装](#4-python-依赖安装)
- [5. Android NDK 配置](#5-android-ndk-配置)
- [6. 环境验证](#6-环境验证)
- [7. 模型转换流程](#7-模型转换流程)
  - [7.1 PyTorch → ONNX](#71-pytorch--onnx)
  - [7.2 构建校准数据集](#72-构建校准数据集)
  - [7.3 ONNX → QNN 量化模型](#73-onnx--qnn-量化模型)
- [8. 常见问题与排查](#8-常见问题与排查)
- [附录：退出容器后重新进入的操作](#附录退出容器后重新进入的操作)

---

## 整体流程概览

```
PyTorch (.pt)
    │
    ▼  pt2onnx/pt2onnx.py
ONNX (.onnx)
    │
    ▼  onnx2so/create_dataset.py (生成校准数据)
校准数据集 (.raw + datasets.txt)
    │
    ▼  onnx2so/convert.sh (QNN 量化 + 编译)
QNN 量化模型 (.so)
```

---

## 1. Docker 环境搭建

### 1.1 创建并启动 Docker 容器

```bash
docker run -it -v ~/yourproject:/project --name qnn_convert ubuntu:22.04 /bin/bash
```

> **说明**：推荐使用 Ubuntu 22.04 作为基础镜像，与 QNN SDK 的兼容性最佳。如有 GPU 需求，可添加 `--gpus all` 参数。

### 1.2 更新系统包

```bash
apt update
```

### 1.3 安装基础工具

```bash
apt install make -y
```

---

## 2. QNN SDK 依赖安装

> **前提**：已将 Qualcomm AI Engine Direct (QAIRT) SDK 放置在项目目录中（例如 `/project/3rdlibrary/` 下）。

### 2.1 加载 SDK 环境变量

```bash
source ${QAIRT_SDK_ROOT}/bin/envsetup.sh
```

此脚本会设置 `QAIRT_SDK_ROOT` 等关键环境变量，后续工具链依赖这些变量。

### 2.2 安装 Linux 系统依赖

```bash
${QAIRT_SDK_ROOT}/bin/check-linux-dependency.sh
```

该脚本会自动检测并安装 QNN SDK 所需的系统级库（如 `libstdc++`、`libgcc` 等）。

### 2.3 运行环境检查

```bash
# 检查基础编译环境
./envcheck -c

# 检查全部环境（含 Python、NDK 等）
./envcheck -a
```

---

## 3. Python 虚拟环境配置

QNN 工具链依赖特定版本的 Python 及其库，建议在虚拟环境中操作以避免与系统包冲突。

### 3.1 安装 Python 3.10

```bash
apt-get update
apt-get install -y python3.10 python3-distutils libpython3.10
```

### 3.2 安装 venv 并创建虚拟环境

```bash
apt install python3.10-venv

python3 -m venv QNN_venv --without-pip
```

### 3.3 激活虚拟环境并安装 pip

```bash
source QNN_venv/bin/activate

python3 -m ensurepip --upgrade
```

> **注意**：每次重新进入容器后，需要重新激活虚拟环境：
> ```bash
> source QNN_venv/bin/activate
> ```

---

## 4. Python 依赖安装

### 4.1 检查 QNN Python 依赖

```bash
python ${QAIRT_SDK_ROOT}/bin/check-python-dependency
```

该工具会报告 QNN SDK 所需的 Python 包及其版本要求。

### 4.2 安装核心依赖

```bash
pip3 install qairt-visualizer==0.8.0
pip3 install torch==1.13.1
pip3 install torchvision==0.14.1
pip3 install tflite==2.18.0
pip3 install onnx==1.17.0
pip3 install onnxruntime==1.22.0
pip3 install onnxsim==0.4.36
```

### 4.3 可选依赖

```bash
# TensorFlow（容易与其他包冲突，非必要可不装）
pip3 install tensorflow==2.10.1
```

> **⚠️ 注意**：`tensorflow` 依赖较多，可能与 `torch` 或其他包产生版本冲突。如果不需要转换 TensorFlow 格式模型，建议跳过安装。

---

## 5. Android NDK 配置

QNN 模型编译为 Android `.so` 库需要 Android NDK 工具链。

### 5.1 设置 NDK 环境变量

将以下内容添加到 `~/.bashrc` 中（请根据实际 NDK 路径修改）：

```bash
echo 'export ANDROID_NDK_ROOT="/project/3rdlibrary/android-ndk-r26c"' >> ~/.bashrc
echo 'export PATH="${ANDROID_NDK_ROOT}:${PATH}"' >> ~/.bashrc
```

### 5.2 使配置生效

```bash
source ~/.bashrc
```

### 5.3 验证 NDK

```bash
export PATH=${ANDROID_NDK_ROOT}:${PATH}
```

> **推荐**：使用 Android NDK r26c 版本，与 QNN SDK 2.44 兼容性已验证。

---

## 6. 环境验证

完成以上所有配置后，运行环境检查确认一切就绪：

```bash
# 加载 SDK 环境变量
source ${QAIRT_SDK_ROOT}/bin/envsetup.sh

# 确保 NDK 在 PATH 中
export PATH=${ANDROID_NDK_ROOT}:${PATH}

# 检查全部环境
./envcheck -a

# 检查 NDK 环境
./envcheck -n
```

所有检查通过后，即可进行模型转换。

---

## 7. 模型转换流程

### 7.1 PyTorch → ONNX

将训练好的 PyTorch 模型（`.pt`）导出为 ONNX 格式。

#### 脚本位置

`model_convert/pt2onnx/pt2onnx.py`

#### 脚本内容说明

```python
from ultralytics import YOLO

model = YOLO("yolo11n-cls.pt")
success = model.export(
    format="onnx",      # 导出为 ONNX 格式
    imgsz=640,           # 输入图像尺寸
    dynamic=False,       # 固定输入尺寸（QNN 要求）
    simplify=True,       # 简化 ONNX 模型图
    opset=13             # ONNX opset 版本
)
```

#### 运行方式

```bash
cd /project/model_convert/pt2onnx
python3 pt2onnx.py
```

#### 输出

- 导出的 ONNX 模型文件（如 `yolo11n-cls.onnx`）

> **关键参数说明**：
> - `dynamic=False`：QNN 要求固定输入维度，不可使用动态轴。
> - `simplify=True`：简化计算图以减少不必要的算子，提高 QNN 转换成功率。
> - `opset=13`：使用 ONNX opset 13，兼容性良好。

---

### 7.2 构建校准数据集

QNN 量化需要校准数据集（calibration dataset）来确定量化参数。

#### 脚本位置

`model_convert/onnx2so/create_dataset.py`

#### 工作流程

1. 从 `dataset_img/` 目录读取校准图片
2. 按照与推理时一致的预处理流程处理图片（resize → 归一化）
3. 将预处理后的张量保存为 NHWC float32 格式的 `.raw` 文件到 `dataset_raw/`
4. 生成 `datasets.txt`，包含所有 `.raw` 文件的绝对路径

#### 运行方式

```bash
cd /project/model_convert/onnx2so
python3 create_dataset.py
```

#### 自定义参数

```bash
python3 create_dataset.py \
    --dataset-img-dir ./dataset_img \
    --dataset-raw-dir ./dataset_raw \
    --dataset-list ./datasets.txt
```

#### 输出

- `dataset_raw/` 目录下的 `.raw` 文件（每个文件为 `640×640×3` NHWC float32 格式）
- `datasets.txt`：包含所有 `.raw` 文件的绝对路径列表

> **重要**：校准图片应具有代表性，覆盖实际推理场景中的数据分布，通常选择 50~200 张图片即可。预处理方式必须与模型训练/推理时一致。

---

### 7.3 ONNX → QNN 量化模型

将 ONNX 模型通过 QNN 工具链转换为量化的 `.so` 动态库，可直接部署到 Android 设备。

#### 脚本位置

`model_convert/onnx2so/convert.sh`

#### 两阶段转换过程

**阶段一：ONNX → QNN 量化模型（.cpp + .bin）**

使用 `qnn-onnx-converter` 将 ONNX 模型转换为 QNN 量化格式：

```bash
${SCRIPT_PATH} \
    --input_network "${MODEL_PATH}" \         # 输入 ONNX 模型路径
    --input_list "${QNN_INPUT_LIST}" \         # 校准数据集列表文件
    -d "${ONNX_INPUT_NAME}" "${ONNX_INPUT_DIMENSIONS}" \  # 输入名称和维度
    --weights_bitwidth 8 \                     # 权重量化位宽：8-bit
    --act_bitwidth 8 \                         # 激活量化位宽：8-bit
    --use_per_channel_quantization \           # 启用逐通道量化（精度更高）
    --algorithms cle \                         # 使用 Cross-Layer Equalization 算法
    --output_path "${OUTPUT_PATH}/${OUTPUT_NAME}.cpp"  # 输出路径
```

**关键参数说明**：

| 参数 | 说明 |
|------|------|
| `--input_network` | 输入的 ONNX 模型文件路径 |
| `--input_list` | 校准数据集列表（`datasets.txt`） |
| `-d` | 指定输入张量的名称和维度，格式为 `名称 N,C,H,W` |
| `--weights_bitwidth 8` | 权重使用 8-bit 量化 |
| `--act_bitwidth 8` | 激活值使用 8-bit 量化 |
| `--use_per_channel_quantization` | 启用逐通道量化，相比逐张量量化精度更高 |
| `--algorithms cle` | 使用 Cross-Layer Equalization 算法优化量化精度 |
| `--output_path` | 量化后模型的输出路径（.cpp 文件） |

**阶段二：QNN 模型 → Android .so 库**

使用 `qnn-model-lib-generator` 将量化模型编译为 Android 动态库：

```bash
python3 "${ROOT_PATH}/bin/x86_64-linux-clang/qnn-model-lib-generator" \
    -c "${OUTPUT_PATH}/${OUTPUT_NAME}.cpp" \   # 量化模型 C++ 源文件
    -b "${OUTPUT_PATH}/${OUTPUT_NAME}.bin" \   # 量化模型二进制权重
    -o ${OUTPUT_PATH} \                         # 输出目录
    -t ${QNN_TARGET_ARCH}                       # 目标架构：aarch64-android
```

#### 运行方式

```bash
cd /project/model_convert/onnx2so
bash convert.sh
```

#### 输出

在 `model_convert/onnx2so/output/` 目录下生成：

| 文件 | 说明 |
|------|------|
| `*_qnn_quantized_model.cpp` | QNN 量化模型 C++ 源文件 |
| `*_qnn_quantized_model.bin` | QNN 量化模型二进制权重 |
| `*_qnn_quantized_model_net.json` | 模型网络结构描述 |
| `lib*.so` | 编译后的 Android 动态库（可直接部署到设备） |

---

## 8. 常见问题与排查

### 问题 1：Broadcast Shape 错误

**错误信息**：

```
ValueError: getBroadcastedTensorShape: Unable to broadcast shapes:
IrTensorShape(dims = [1,1098563595,2,336], dy_axes=[]) and
IrTensorShape(dims = [1,16,1,178486], dy_axes=[])
```

**原因**：`numpy` 版本过高导致与 QNN 工具链不兼容，产生异常的 Tensor Shape。

**解决方案**：

```bash
pip3 install numpy==1.26.4
```

> **说明**：`numpy 1.26.4` 是经过验证的稳定版本，与 QNN SDK 2.44 兼容性良好。

---

### 问题 2：Python 依赖冲突

如果安装 `tensorflow` 后出现其他包异常，建议在新的虚拟环境中单独安装所需包，或直接卸载 `tensorflow`：

```bash
pip3 uninstall tensorflow
```

---

### 问题 3：NDK 找不到

确保 `ANDROID_NDK_ROOT` 环境变量正确指向 NDK 目录，并且该目录下存在 `ndk-build` 等工具：

```bash
echo $ANDROID_NDK_ROOT
ls ${ANDROID_NDK_ROOT}/ndk-build
```

---

### 问题 4：虚拟环境未激活

如果运行 Python 脚本时报包找不到的错误，检查是否在虚拟环境中：

```bash
which python3
# 应输出类似: /project/QNN_venv/bin/python3
```

如不在虚拟环境中，重新激活：

```bash
source QNN_venv/bin/activate
```

---

## 附录：退出容器后重新进入的操作

每次重新进入 Docker 容器后，需要重新执行以下操作：

```bash
# 1. 重启容器（如已停止）
docker start qnn_convert
docker exec -it qnn_convert /bin/bash

# 2. 激活 Python 虚拟环境
source QNN_venv/bin/activate

# 3. 加载 QNN SDK 环境变量
source ./envsetup.sh

# 4. 确保 NDK 在 PATH 中
export PATH=${ANDROID_NDK_ROOT}:${PATH}

# 5. 验证环境
./envcheck -a
```

> **提示**：可以将上述命令写入一个启动脚本（如 `setup_env.sh`），每次进入容器后只需 `source setup_env.sh` 即可一键配置所有环境。