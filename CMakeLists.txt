cmake_minimum_required(VERSION 3.16)

project(QNNProj LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(QNN_SDK_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/3rdlibrary/v2.44.0.260225/qairt/2.44.0.260225"
    CACHE PATH "Qualcomm QAIRT/QNN SDK root")

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# 静态库会链接进 test_classify，运行时不需要单独部署。
add_library(qnn_model_inference STATIC
    model_inference/model_inference_base/qnn_model_inference_base.cpp
    model_inference/classification/yolo_classification_inference.cpp
)

target_include_directories(qnn_model_inference
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${QNN_SDK_ROOT}/include/QNN
        ${QNN_SDK_ROOT}/share/QNN/converter/jni
)

target_link_libraries(qnn_model_inference
    PUBLIC
        ${OpenCV_LIBS}
        # QNN backend 和模型库运行时通过 dlopen 加载。
        dl
)

add_executable(test_classify
    test_classify.cpp
)

target_link_libraries(test_classify
    PRIVATE
        qnn_model_inference
)
