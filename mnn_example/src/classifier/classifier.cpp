#include "classifier.h"

#include <algorithm>
#include <iostream>

#include "classification.h"
#include "opencv2/imgproc.hpp"

namespace mirror {

Classifier::Classifier() {
  labels_.clear();
  initialized_ = false;
  topk_ = 5;
}

Classifier::~Classifier() {
  if (classifier_interpreter_ && classifier_sess_) {
    classifier_interpreter_->releaseSession(classifier_sess_);
    classifier_sess_ = nullptr;
  }
}

int Classifier::Init(const char* root_path) {
  if (!root_path || !*root_path)
    return 10000;
  std::cout << "start Init." << std::endl;
  std::string model_file = std::string(root_path) + "/mobilenet.mnn";
  initialized_ = false;
  classifier_sess_ = nullptr;
  classifier_interpreter_ =
      std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_file.c_str()));

  if (!classifier_interpreter_ || LoadLabels(root_path) != 0) {
    std::cout << "load model failed." << std::endl;
    return 10000;
  }

  MNN::ScheduleConfig schedule_config;
  schedule_config.type = MNN_FORWARD_CPU;
  schedule_config.numThread = 1;
  MNN::BackendConfig backend_config;
  backend_config.precision = MNN::BackendConfig::Precision_Normal;
  schedule_config.backendConfig = &backend_config;

  classifier_sess_ = classifier_interpreter_->createSession(schedule_config);
  if (!classifier_sess_)
    return 10000;
  input_tensor_ = classifier_interpreter_->getSessionInput(classifier_sess_, nullptr);
  if (!input_tensor_)
    return 10000;

  classifier_interpreter_->resizeTensor(input_tensor_, {1, 3, inputSize_.height, inputSize_.width});
  classifier_interpreter_->resizeSession(classifier_sess_);

  std::cout << "End Init." << std::endl;

  initialized_ = true;

  return 0;
}

int Classifier::Classify(const cv::Mat& img_src, std::vector<ImageInfo>* images) {
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

  cv::Mat img_resized;
  cv::resize(img_src.clone(), img_resized, inputSize_);
  std::shared_ptr<MNN::CV::ImageProcess> pretreat(
      MNN::CV::ImageProcess::create(MNN::CV::BGR, MNN::CV::RGB, meanVals, 3, normVals, 3));
  pretreat->convert((uint8_t*)img_resized.data, inputSize_.width, inputSize_.height,
                    img_resized.step[0], input_tensor_);

  // forward
  if (classifier_interpreter_->runSession(classifier_sess_) != MNN::NO_ERROR)
    return 10000;

  // get output
  // mobilenet: "classifierV1/Predictions/Reshape_1"
  MNN::Tensor* output_score = classifier_interpreter_->getSessionOutput(classifier_sess_, nullptr);

  if (!output_score)
    return 10000;
  // copy to host
  MNN::Tensor score_host(output_score, output_score->getDimensionType());
  output_score->copyToHostTensor(&score_host);

  auto score_ptr = score_host.host<float>();
  try {
    const auto scores = demo::topScores(score_ptr, score_host.elementSize(), labels_.size(), topk_);
    for (const auto& score : scores) {
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

int Classifier::LoadLabels(const char* root_path) {
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
