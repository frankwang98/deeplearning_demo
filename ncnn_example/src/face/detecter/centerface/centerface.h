#ifndef _FACE_CENTERFACE_H_
#define _FACE_CENTERFACE_H_

#include <vector>

#include "../detecter.h"
#include "net.h"
#include "opencv2/core.hpp"

namespace mirror {
class CenterFace : public Detecter {
 public:
  CenterFace();
  ~CenterFace();
  int LoadModel(const char* root_path);
  int DetectFace(const cv::Mat& img_src, std::vector<FaceInfo>* faces);

 private:
  ncnn::Net* centernet_ = nullptr;
  const float scoreThreshold_ = 0.5f;
  const float nmsThreshold_ = 0.5f;
  bool initialized_;
};

}  // namespace mirror

#endif  // !_FACE_CENTERFACE_H_
