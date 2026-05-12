ONNX_INPUT_NAME="images"
ONNX_INPUT_DIMENSIONS="1,3,512,512"
MODEL_PATH="/project/model_convert/pt2onnx/best.onnx"
QNN_INPUT_LIST="/project/model_convert/onnx2so/datasets.txt"
SCRIPT_PATH=/project/3rdlibrary/v2.44.0.260225/qairt/2.44.0.260225/bin/x86_64-linux-clang/qnn-onnx-converter

OUTPUT_PATH="/project/model_convert/onnx2so/output"
OUTPUT_NAME="yolo11n-cls_qnn_quantized_model"

${SCRIPT_PATH} \
    --input_network "${MODEL_PATH}" \
    --input_list "${QNN_INPUT_LIST}" \
    -d "${ONNX_INPUT_NAME}" "${ONNX_INPUT_DIMENSIONS}" \
    --weights_bitwidth 8 \
    --act_bitwidth 8 \
    --use_per_channel_quantization \
    --algorithms cle \
    --output_path "${OUTPUT_PATH}/${OUTPUT_NAME}.cpp"

# 将模型转化为动态so库
ROOT_PATH=/project/3rdlibrary/v2.44.0.260225/qairt/2.44.0.260225
QNN_TARGET_ARCH="aarch64-android"

python3 "${ROOT_PATH}/bin/x86_64-linux-clang/qnn-model-lib-generator" \
-c "${OUTPUT_PATH}/${OUTPUT_NAME}.cpp" \
-b "${OUTPUT_PATH}/${OUTPUT_NAME}.bin" \
-o ${OUTPUT_PATH} \
-t ${QNN_TARGET_ARCH}