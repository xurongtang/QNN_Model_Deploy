#include "model_inference/classification/yolo_classification_inference.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>

#include <opencv2/imgproc.hpp>

namespace qnn_inference {
namespace {

constexpr size_t kClassCount = 10;

uint8_t clampToU8(int value) {
  return static_cast<uint8_t>(std::max(0, std::min(255, value)));
}

size_t elementCount(const std::vector<uint32_t>& dims) {
  size_t count = 1;
  for (const uint32_t dim : dims) {
    count *= dim;
  }
  return count;
}

}  // 匿名命名空间

bool YoloClassificationInference::init(const std::string& backend_library_path,
                                       const std::string& model_library_path) {
  RuntimeConfig config;
  config.backend_library_path = backend_library_path;
  config.model_library_path = model_library_path;
  return loadModel(config);
}

ClassificationResult YoloClassificationInference::infer(const cv::Mat& image) {
  last_result_ = ClassificationResult{};

  if (image.empty()) {
    last_result_.error = "input image is empty";
    return last_result_;
  }

  current_image_ = image;
  if (!QnnModelInferenceBase::infer()) {
    last_result_.error = lastError();
    return last_result_;
  }

  last_result_.success = true;
  return last_result_;
}

bool YoloClassificationInference::preprocess(std::vector<TensorBuffer>& inputs) {
  if (inputs.size() != 1) {
    std::ostringstream oss;
    oss << "classification model expects 1 input tensor, got " << inputs.size();
    setLastError(oss.str());
    return false;
  }

  TensorBuffer& input = inputs[0];
  if (input.dims.size() != 4) {
    setLastError("classification input tensor must be rank-4");
    return false;
  }

  const bool nhwc_layout = input.dims[3] == 3;
  const bool nchw_layout = input.dims[1] == 3;
  if (!nhwc_layout && !nchw_layout) {
    setLastError("classification input tensor must be NHWC or NCHW with 3 channels");
    return false;
  }

  const int height = static_cast<int>(nhwc_layout ? input.dims[1] : input.dims[2]);
  const int width = static_cast<int>(nhwc_layout ? input.dims[2] : input.dims[3]);

  // 保持与校准/导出时一致：BGR/灰度/BGRA -> RGB，resize，
  // 归一化到 0..1，再按模型 tensor 布局写入。
  cv::Mat bgr;
  if (current_image_.channels() == 1) {
    cv::cvtColor(current_image_, bgr, cv::COLOR_GRAY2BGR);
  } else if (current_image_.channels() == 4) {
    cv::cvtColor(current_image_, bgr, cv::COLOR_BGRA2BGR);
  } else if (current_image_.channels() == 3) {
    bgr = current_image_;
  } else {
    setLastError("unsupported input image channel count");
    return false;
  }

  cv::Mat resized;
  cv::resize(bgr, resized, cv::Size(width, height), 0.0, 0.0, cv::INTER_LINEAR);

  cv::Mat rgb;
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

  switch (input.data_type) {
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8:
      return fillQuantizedInput(rgb, &input, nhwc_layout);
    case QNN_DATATYPE_FLOAT_32:
      return fillFloatInput(rgb, &input, nhwc_layout);
    default: {
      std::ostringstream oss;
      oss << "unsupported input tensor data type: " << input.data_type;
      setLastError(oss.str());
      return false;
    }
  }
}

bool YoloClassificationInference::postprocess(const std::vector<TensorBuffer>& outputs) {
  if (outputs.size() != 1) {
    std::ostringstream oss;
    oss << "classification model expects 1 output tensor, got " << outputs.size();
    setLastError(oss.str());
    return false;
  }

  std::vector<float> scores;
  if (!decodeOutputScores(outputs[0], &scores)) {
    return false;
  }
  if (scores.size() != kClassCount) {
    std::ostringstream oss;
    oss << "classification output expects " << kClassCount << " scores, got " << scores.size();
    setLastError(oss.str());
    return false;
  }

  const auto best_it = std::max_element(scores.begin(), scores.end());
  const int class_id = static_cast<int>(std::distance(scores.begin(), best_it));

  last_result_.class_id = class_id;
  last_result_.score = *best_it;
  last_result_.scores = std::move(scores);
  last_result_.label = class_id >= 0 && static_cast<size_t>(class_id) < class_names_.size()
                           ? class_names_[class_id]
                           : "unknown";
  return true;
}

bool YoloClassificationInference::fillQuantizedInput(const cv::Mat& rgb,
                                                     TensorBuffer* input,
                                                     bool nhwc_layout) {
  const size_t expected_elements = elementCount(input->dims);
  if (input->byte_size < expected_elements) {
    setLastError("input tensor buffer is smaller than expected");
    return false;
  }

  auto* dst = static_cast<uint8_t*>(input->data);
  const float scale = input->quant_scale > 0.0f ? input->quant_scale : (1.0f / 255.0f);
  const int32_t offset = input->quant_offset;

  // QNN scale-offset 的反量化公式是 (q - offset) * scale，
  // 因此量化时使用 round(real / scale) + offset。
  if (nhwc_layout) {
    for (int y = 0; y < rgb.rows; ++y) {
      const cv::Vec3b* row = rgb.ptr<cv::Vec3b>(y);
      for (int x = 0; x < rgb.cols; ++x) {
        for (int c = 0; c < 3; ++c) {
          const float normalized = static_cast<float>(row[x][c]) / 255.0f;
          const int quantized = static_cast<int>(std::lround(normalized / scale)) + offset;
          *dst++ = clampToU8(quantized);
        }
      }
    }
    return true;
  }

  const int plane_size = rgb.rows * rgb.cols;
  for (int c = 0; c < 3; ++c) {
    uint8_t* channel_dst = dst + c * plane_size;
    for (int y = 0; y < rgb.rows; ++y) {
      const cv::Vec3b* row = rgb.ptr<cv::Vec3b>(y);
      for (int x = 0; x < rgb.cols; ++x) {
        const float normalized = static_cast<float>(row[x][c]) / 255.0f;
        const int quantized = static_cast<int>(std::lround(normalized / scale)) + offset;
        channel_dst[y * rgb.cols + x] = clampToU8(quantized);
      }
    }
  }

  return true;
}

bool YoloClassificationInference::fillFloatInput(const cv::Mat& rgb,
                                                 TensorBuffer* input,
                                                 bool nhwc_layout) {
  const size_t expected_elements = elementCount(input->dims);
  if (input->byte_size < expected_elements * sizeof(float)) {
    setLastError("input tensor buffer is smaller than expected");
    return false;
  }

  auto* dst = static_cast<float*>(input->data);
  if (nhwc_layout) {
    cv::Mat float_rgb;
    rgb.convertTo(float_rgb, CV_32FC3, 1.0 / 255.0);
    std::memcpy(dst, float_rgb.data, expected_elements * sizeof(float));
    return true;
  }

  const int plane_size = rgb.rows * rgb.cols;
  for (int c = 0; c < 3; ++c) {
    float* channel_dst = dst + c * plane_size;
    for (int y = 0; y < rgb.rows; ++y) {
      const cv::Vec3b* row = rgb.ptr<cv::Vec3b>(y);
      for (int x = 0; x < rgb.cols; ++x) {
        channel_dst[y * rgb.cols + x] = static_cast<float>(row[x][c]) / 255.0f;
      }
    }
  }

  return true;
}

bool YoloClassificationInference::decodeOutputScores(const TensorBuffer& output,
                                                     std::vector<float>* scores) {
  const size_t count = elementCount(output.dims);
  scores->assign(count, 0.0f);

  switch (output.data_type) {
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8: {
      const auto* data = static_cast<const uint8_t*>(output.data);
      const float scale = output.quant_scale > 0.0f ? output.quant_scale : (1.0f / 255.0f);
      // 将 QNN native 输出反量化为 float 分数。
      for (size_t i = 0; i < count; ++i) {
        (*scores)[i] = static_cast<float>(static_cast<int32_t>(data[i]) - output.quant_offset) *
                       scale;
      }
      return true;
    }
    case QNN_DATATYPE_FLOAT_32: {
      const auto* data = static_cast<const float*>(output.data);
      std::copy(data, data + count, scores->begin());
      return true;
    }
    default: {
      std::ostringstream oss;
      oss << "unsupported output tensor data type: " << output.data_type;
      setLastError(oss.str());
      return false;
    }
  }
}

}  // 命名空间 qnn_inference
