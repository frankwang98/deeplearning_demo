#pragma once
#include <stdexcept>
#include <string>
namespace demo {
struct Options {
  std::string image;
  std::string models;
  std::string output = "result.jpg";
  bool show = false;
  bool help = false;
};
inline Options parseOptions(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string key = argv[index];
    if (key == "--help") {
      options.help = true;
      continue;
    }
    if (key == "--show") {
      options.show = true;
      continue;
    }
    if (key != "--image" && key != "--models" && key != "--output")
      throw std::invalid_argument("Unknown option: " + key);
    if (++index >= argc || std::string(argv[index]).find("--") == 0)
      throw std::invalid_argument("Missing value for " + key);
    const std::string value = argv[index];
    if (key == "--image")
      options.image = value;
    else if (key == "--models")
      options.models = value;
    else
      options.output = value;
  }
  if (!options.help && (options.image.empty() || options.models.empty() || options.output.empty()))
    throw std::invalid_argument("--image and --models are required; --output must not be empty");
  return options;
}
}  // namespace demo
