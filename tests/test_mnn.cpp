#include "classifier.h"
#include "vision_engine.h"
int main() {
  { mirror::VisionEngine engine; }
  mirror::Classifier classifier;
  if (classifier.Init("/missing-deeplearning-models") == 0)
    return 1;
  if (classifier.Init("/missing-deeplearning-models") == 0)
    return 1;
  std::vector<mirror::ImageInfo> output;
  if (classifier.Classify(cv::Mat(), &output) == 0)
    return 1;
  if (classifier.Classify(cv::Mat(), nullptr) == 0)
    return 1;
  return 0;
}
