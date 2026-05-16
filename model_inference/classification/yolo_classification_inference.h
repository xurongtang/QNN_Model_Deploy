#pragma once

#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "model_inference/model_inference_base/qnn_model_inference_base.h"

namespace qnn_inference {

struct ClassificationResult {
  bool success = false;
  int class_id = -1;
  float score = 0.0f;
  std::string label;
  std::vector<float> scores;
  std::string error;
};

class YoloClassificationInference final : private QnnModelInferenceBase {
 public:
  // 对外只保留初始化和 cv::Mat 推理两个核心接口。
  bool init(const std::string& backend_library_path, const std::string& model_library_path);
  ClassificationResult infer(const cv::Mat& image);
  const std::string& lastError() const { return QnnModelInferenceBase::lastError(); }

 private:
  bool preprocess(std::vector<TensorBuffer>& inputs) override;
  bool postprocess(const std::vector<TensorBuffer>& outputs) override;

  bool fillQuantizedInput(const cv::Mat& rgb, TensorBuffer* input, bool nhwc_layout);
  bool fillFloatInput(const cv::Mat& rgb, TensorBuffer* input, bool nhwc_layout);
  bool decodeOutputScores(const TensorBuffer& output, std::vector<float>* scores);

  cv::Mat current_image_;
  ClassificationResult last_result_;
  std::vector<std::string> class_names_ = {
      "class_0",
      "class_1",
      "class_2",
      "class_3",
      "class_4",
      "class_5",
      "class_6",
      "class_7",
      "class_8",
      "class_9",
  };
};

}  // 命名空间 qnn_inference
