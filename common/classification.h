#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace demo {
// Accept plain names and ImageNet-style indexed/synset-prefixed names.
inline std::vector<std::string> loadLabels(const std::string& path) {
  std::ifstream input(path);
  if (!input)
    throw std::runtime_error("Cannot open label file: " + path);
  std::vector<std::string> labels;
  std::string line;
  while (std::getline(input, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    const auto first = line.find_first_not_of(" \t");
    if (first == std::string::npos)
      continue;
    line.erase(0, first);
    const auto split = line.find_first_of(" \t");
    const auto prefix = line.substr(0, split);
    const bool index =
        !prefix.empty() && std::all_of(prefix.begin(), prefix.end(),
                                       [](unsigned char c) { return c >= '0' && c <= '9'; });
    const bool synset = prefix.size() == 9 && prefix[0] == 'n' &&
                        std::all_of(prefix.begin() + 1, prefix.end(),
                                    [](unsigned char c) { return c >= '0' && c <= '9'; });
    if (split != std::string::npos && (index || synset)) {
      const auto label_start = line.find_first_not_of(" \t", split);
      if (label_start == std::string::npos)
        throw std::runtime_error("Empty indexed label");
      line.erase(0, label_start);
    }
    line.erase(line.find_last_not_of(" \t") + 1);
    labels.push_back(line);
  }
  if (input.bad() || labels.empty())
    throw std::runtime_error("Empty or unreadable label file: " + path);
  return labels;
}
inline std::vector<std::pair<float, size_t>> topScores(const float* scores, size_t count,
                                                       size_t label_count, size_t requested = 5) {
  if (!scores || !count || count != label_count || !requested)
    throw std::runtime_error("Output classes and labels must match and be non-empty");
  std::vector<std::pair<float, size_t>> ranked;
  ranked.reserve(count);
  for (size_t index = 0; index < count; ++index) {
    if (!std::isfinite(scores[index]))
      throw std::runtime_error("Non-finite classification score");
    ranked.emplace_back(scores[index], index);
  }
  const auto keep = std::min(requested, count);
  std::partial_sort(ranked.begin(), ranked.begin() + keep, ranked.end(),
                    std::greater<std::pair<float, size_t>>());
  ranked.resize(keep);
  return ranked;
}
}  // namespace demo
