#define _CRT_SECURE_NO_WARNINGS
#include "mobilenet.h"

#include <algorithm>
#include <string>

#include "classification.h"

#if MIRROR_VULKAN
#include "gpu.h"
#endif  // MIRROR_VULKAN

namespace mirror {
Mobilenet::Mobilenet() {
  mobilenet_ = new ncnn::Net();
  initialized_ = false;
#if MIRROR_VULKAN
  ncnn::create_gpu_instance();
  mobilenet_->opt.use_vulkan_compute = true;
#endif  // MIRROR_VULKAN
}

Mobilenet::~Mobilenet() {
  if (mobilenet_) {
    delete mobilenet_;
    mobilenet_ = nullptr;
  }
#if MIRROR_VULKAN
  ncnn::destroy_gpu_instance();
#endif  // MIRROR_VULKAN
}

int Mobilenet::LoadModel(const char* root_path) {
  initialized_ = false;
  if (!root_path || !*root_path)
    return 10000;
  mobilenet_->clear();
  std::cout << "start load model." << std::endl;
  std::string param_file = std::string(root_path) + "/mobilenet.param";
  std::string model_file = std::string(root_path) + "/mobilenet.bin";
  if (mobilenet_->load_param(param_file.c_str()) != 0 ||
      mobilenet_->load_model(model_file.c_str()) != 0 || LoadLabels(root_path) != 0) {
    std::cout << "load model or label file failed." << std::endl;
    return 10000;
  }
  initialized_ = true;
  std::cout << "end load model." << std::endl;

  return 0;
}
int Mobilenet::Classify(const cv::Mat& img_src, std::vector<ImageInfo>* images) {
  std::cout << "start classify." << std::endl;
  if (!images)
    return 10001;
  images->clear();
  if (!initialized_) {
    std::cout << "model uninitialized." << std::endl;
    return 10000;
  }
  if (img_src.empty() || img_src.type() != CV_8UC3) {
    std::cout << "input empty." << std::endl;
    return 10001;
  }
  ncnn::Mat in = ncnn::Mat::from_pixels_resize(img_src.data, ncnn::Mat::PIXEL_BGR2RGB, img_src.cols,
                                               img_src.rows, inputSize.width, inputSize.height);
  in.substract_mean_normalize(meanVals, normVals);

  ncnn::Extractor ex = mobilenet_->create_extractor();
  if (ex.input("data", in) != 0)
    return 10000;
  ncnn::Mat out;
  if (ex.extract("prob", out) != 0 || out.empty() || out.dims != 1 || out.elemsize != sizeof(float))
    return 10000;

  try {
    for (const auto& score : demo::topScores(out, out.w, labels_.size())) {
      ImageInfo info;
      info.label_ = labels_[score.second];
      info.score_ = score.first;
      images->push_back(info);
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 10000;
  }
  std::cout << "end classify." << std::endl;
  return 0;
}

int Mobilenet::LoadLabels(const char* root_path) {
  labels_.clear();
  try {
    labels_ = demo::loadLabels(std::string(root_path) + "/label.txt");
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 10000;
  }
  return 0;
}
}  // namespace mirror
