#ifndef _FACE_RECOGNIZER_H_
#define _FACE_RECOGNIZER_H_

#include <vector>

#include "../../common/common.h"
#include "opencv2/core.hpp"

namespace mirror {
class Recognizer {
 public:
  virtual int Init(const char* model_path) = 0;
  virtual int ExtractFeature(const cv::Mat& img_face, std::vector<float>* feat) = 0;
  virtual ~Recognizer() {}
};

class RecognizerFactory {
 public:
  virtual Recognizer* CreateRecognizer() = 0;
  virtual ~RecognizerFactory() {}
};

class MobilefacenetFactory : public RecognizerFactory {
 public:
  MobilefacenetFactory() {}
  Recognizer* CreateRecognizer();
  ~MobilefacenetFactory() {}
};

}  // namespace mirror

#endif  // !_FACE_RECOGNIZER_H_