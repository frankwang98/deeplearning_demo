#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>

#include "classification.h"
#include "demo_options.h"
#include "file_stream.h"
void check(bool value) {
  if (!value)
    throw std::runtime_error("regression failed");
}
int main() {
  try {
    const auto directory =
        std::filesystem::temp_directory_path() / "deeplearning-demo-common-tests";
    std::filesystem::create_directories(directory);
    const auto path = directory / "labels.txt";
    {
      std::ofstream file(path);
      file << "0 cat\r\n1 dog\nred fox\nn01234567 sea lion";
    }
    const auto labels = demo::loadLabels(path.string());
    check(labels.size() == 4 && labels[2] == "red fox" && labels[3] == "sea lion");
    const float values[] = {0.1f, 0.9f};
    const auto scores = demo::topScores(values, 2, 2);
    check(scores.size() == 2 && scores[0].second == 1);
    bool failed = false;
    try {
      demo::topScores(values, 2, 1);
    } catch (...) {
      failed = true;
    }
    check(failed);
    failed = false;
    try {
      demo::loadLabels((directory / "missing").string());
    } catch (...) {
      failed = true;
    }
    check(failed);
    mirror::FileStream stream(path.string(), mirror::FileStream::Input);
    stream.close();
    stream.close();
    check(!stream.is_opened());
    check(stream.open(path.string(), mirror::FileStream::Input));
    check(!stream.open((directory / "missing").string(), mirror::FileStream::Input));
    char* help[] = {const_cast<char*>("demo"), const_cast<char*>("--help")};
    check(demo::parseOptions(2, help).help);
    char* invalid[] = {const_cast<char*>("demo"), const_cast<char*>("--models")};
    failed = false;
    try {
      demo::parseOptions(2, invalid);
    } catch (...) {
      failed = true;
    }
    check(failed);
    std::filesystem::remove_all(directory);
    std::cout << "Label, score, file lifetime and CLI regressions passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
