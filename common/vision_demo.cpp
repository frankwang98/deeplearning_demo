#include <chrono>
#include <filesystem>
#include <iostream>

#include "demo_options.h"
#include "opencv2/opencv.hpp"
#if defined(DEMO_CLASSIFIER)
#if defined(DEMO_MNN)
#include "classifier.h"
#else
#include "classifier_engine.h"
#endif
#elif defined(DEMO_OBJECT)
#include "object_engine.h"
#else
#include "face_engine.h"
#endif
int main(int argc, char** argv) {
  try {
    const auto options = demo::parseOptions(argc, argv);
    if (options.help) {
      std::cout
          << "Usage: --image photo.jpg --models model_directory [--output result.jpg] [--show]\n";
      return 0;
    }
    if (!std::filesystem::is_directory(options.models))
      throw std::runtime_error("Model directory does not exist: " + options.models);
    cv::Mat image = cv::imread(options.image);
    if (image.empty())
      throw std::runtime_error("Cannot read image: " + options.image);
#if defined(DEMO_CLASSIFIER)
#if defined(DEMO_MNN)
    mirror::Classifier engine;
#else
    mirror::ClassifierEngine engine;
#endif
#elif defined(DEMO_OBJECT)
    mirror::ObjectEngine engine;
#else
    mirror::FaceEngine engine;
#endif
#if defined(DEMO_MNN)
    const int loaded = engine.Init(options.models.c_str());
#else
    const int loaded = engine.LoadModel(options.models.c_str());
#endif
    if (loaded != 0)
      throw std::runtime_error("Model initialization failed (code " + std::to_string(loaded) + ")");
    const auto start = std::chrono::steady_clock::now();
#if defined(DEMO_CLASSIFIER)
    std::vector<mirror::ImageInfo> results;
    const int status = engine.Classify(image, &results);
#elif defined(DEMO_OBJECT)
    std::vector<mirror::ObjectInfo> results;
    const int status = engine.DetectObject(image, &results);
#else
    std::vector<mirror::FaceInfo> results;
    const int status = engine.DetectFace(image, &results);
#endif
    const auto end = std::chrono::steady_clock::now();
    if (status != 0)
      throw std::runtime_error("Inference failed (code " + std::to_string(status) + ")");
    for (size_t index = 0; index < results.size(); ++index) {
#if defined(DEMO_CLASSIFIER)
      const std::string text = results[index].label_ + " " + std::to_string(results[index].score_);
      std::cout << text << '\n';
      cv::putText(image, text, cv::Point(10, 25 + 25 * static_cast<int>(index)),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 180, 0), 1);
#else
      cv::rectangle(image, results[index].location_, cv::Scalar(0, 180, 0), 2);
#endif
    }
    if (!cv::imwrite(options.output, image))
      throw std::runtime_error("Cannot write output: " + options.output);
    std::cout << "Pipeline call (preprocess + inference + postprocess): "
              << std::chrono::duration<double, std::milli>(end - start).count()
              << " ms\nSaved: " << options.output << '\n';
    if (options.show) {
      cv::imshow("result", image);
      cv::waitKey(0);
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
