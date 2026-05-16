#include <iomanip>
#include <iostream>
#include <chrono>
#include <string>

#include <opencv2/imgcodecs.hpp>

#include "model_inference/classification/yolo_classification_inference.h"

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0]
              << " <qnn_backend_so> <qnn_model_so> <image_path>\n"
              << "Example: " << argv[0]
              << " /data/local/tmp/qnn/libQnnHtp.so"
              << " /data/local/tmp/qnn/libyolo11n_cls_qnn_quantized_model.so"
              << " test.jpg\n";
    return 1;
  }

  const std::string backend_path = argv[1];
  const std::string model_path = argv[2];
  const std::string image_path = argv[3];

  cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  if (image.empty()) {
    std::cerr << "Failed to read image: " << image_path << '\n';
    return 1;
  }

  qnn_inference::YoloClassificationInference classifier;
  // 初始化耗时包含 dlopen、QNN backend/device/context 创建和 graph finalize。
  const auto init_start = std::chrono::steady_clock::now();
  if (!classifier.init(backend_path, model_path)) {
    std::cerr << "Failed to init classifier: " << classifier.lastError() << '\n';
    return 1;
  }
  const auto init_end = std::chrono::steady_clock::now();

  // 推理耗时包含预处理、graphExecute 和后处理。
  const auto infer_start = std::chrono::steady_clock::now();
  const qnn_inference::ClassificationResult result = classifier.infer(image);
  const auto infer_end = std::chrono::steady_clock::now();
  if (!result.success) {
    std::cerr << "Failed to run inference: " << result.error << '\n';
    return 1;
  }

  const double init_ms =
      std::chrono::duration<double, std::milli>(init_end - init_start).count();
  const double infer_ms =
      std::chrono::duration<double, std::milli>(infer_end - infer_start).count();

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Init time: " << init_ms << " ms\n";
  std::cout << "Infer time: " << infer_ms << " ms\n";
  std::cout << "Top1: class_id=" << result.class_id << ", label=" << result.label
            << ", score=" << result.score << '\n';

  std::cout << "Scores:";
  for (size_t i = 0; i < result.scores.size(); ++i) {
    std::cout << " [" << i << "]=" << result.scores[i];
  }
  std::cout << '\n';

  return 0;
}
