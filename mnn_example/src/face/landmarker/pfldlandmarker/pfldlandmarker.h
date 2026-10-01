#ifndef _FACE_PFLDLANDMARKER_H_
#define _FACE_PFLDLANDMARKER_H_

#include <memory>
#include <vector>

#include "../landmarker.h"
#include "MNN/ImageProcess.hpp"
#include "MNN/Interpreter.hpp"
#include "MNN/MNNDefine.h"
#include "MNN/Tensor.hpp"
#include "opencv2/core.hpp"

namespace mirror {
class PFLDLandmarker : public Landmarker {
 public:
  PFLDLandmarker();
  ~PFLDLandmarker();
  int Init(const char* model_path);
  int ExtractKeypoints(const cv::Mat& img_src, const cv::Rect& face,
                       std::vector<cv::Point2f>* keypoints);

 private:
  bool initialized_;
  std::shared_ptr<MNN::CV::ImageProcess> pretreat_ = nullptr;
  std::shared_ptr<MNN::Interpreter> pfld_interpreter_ = nullptr;
  MNN::Session* pfld_sess_ = nullptr;
  MNN::Tensor* input_tensor_ = nullptr;

  const cv::Size inputSize_ = cv::Size(96, 96);
  const float meanVals_[3] = {123.0f, 123.0f, 123.0f};
  const float normVals_[3] = {0.01724f, 0.01724f, 0.01724f};
};

}  // namespace mirror

#endif  // !_FACE_PFLDLANDMARKER_H_
