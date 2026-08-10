#pragma once

#include "localai/types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <sstream>

namespace localai::image_tensor {

inline bool validRgb(const InferenceRequest& request) {
    return request.imageWidth > 0 && request.imageHeight > 0 && request.imageChannels == 3 &&
           request.imagePixels.size() == static_cast<std::size_t>(request.imageWidth * request.imageHeight * 3);
}

inline std::array<float, 3> sampleBilinear(const InferenceRequest& request, float x, float y) {
    x = std::clamp(x, 0.0F, static_cast<float>(request.imageWidth - 1));
    y = std::clamp(y, 0.0F, static_cast<float>(request.imageHeight - 1));
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, request.imageWidth - 1);
    const int y1 = std::min(y0 + 1, request.imageHeight - 1);
    const float fx = x - std::floor(x);
    const float fy = y - std::floor(y);
    std::array<float, 3> result{};
    for (int channel = 0; channel < 3; ++channel) {
        const auto at = [&](int px, int py) {
            return request.imagePixels[static_cast<std::size_t>((py * request.imageWidth + px) * 3 + channel)] / 255.0F;
        };
        const float upper = at(x0, y0) * (1.0F - fx) + at(x1, y0) * fx;
        const float lower = at(x0, y1) * (1.0F - fx) + at(x1, y1) * fx;
        result[static_cast<std::size_t>(channel)] = upper * (1.0F - fy) + lower * fy;
    }
    return result;
}

inline std::vector<float> imagenetNchw(const InferenceRequest& request, int width, int height) {
    constexpr std::array<float, 3> mean{0.485F, 0.456F, 0.406F};
    constexpr std::array<float, 3> deviation{0.229F, 0.224F, 0.225F};
    std::vector<float> output(static_cast<std::size_t>(3 * width * height));
    for (int y = 0; y < height; ++y) {
        const float sourceY = (y + 0.5F) * request.imageHeight / height - 0.5F;
        for (int x = 0; x < width; ++x) {
            const float sourceX = (x + 0.5F) * request.imageWidth / width - 0.5F;
            const auto pixel = sampleBilinear(request, sourceX, sourceY);
            for (int channel = 0; channel < 3; ++channel) {
                output[static_cast<std::size_t>(channel * width * height + y * width + x)] =
                    (pixel[static_cast<std::size_t>(channel)] - mean[static_cast<std::size_t>(channel)]) /
                    deviation[static_cast<std::size_t>(channel)];
            }
        }
    }
    return output;
}

inline std::vector<float> mobilenetNhwc(const InferenceRequest& request, int width, int height) {
    std::vector<float> output(static_cast<std::size_t>(width * height * 3));
    for (int y = 0; y < height; ++y) {
        const float sourceY = (y + 0.5F) * request.imageHeight / height - 0.5F;
        for (int x = 0; x < width; ++x) {
            const float sourceX = (x + 0.5F) * request.imageWidth / width - 0.5F;
            const auto pixel = sampleBilinear(request, sourceX, sourceY);
            for (int channel = 0; channel < 3; ++channel) {
                output[static_cast<std::size_t>((y * width + x) * 3 + channel)] =
                    pixel[static_cast<std::size_t>(channel)] * 2.0F - 1.0F;
            }
        }
    }
    return output;
}

inline std::string topClasses(const std::vector<float>& values, std::size_t offset = 0, std::size_t count = 5) {
    if (values.size() <= offset) return "The classifier returned no class scores.";
    std::vector<std::size_t> indices(values.size() - offset);
    std::iota(indices.begin(), indices.end(), offset);
    const auto limit = std::min(count, indices.size());
    std::partial_sort(indices.begin(), indices.begin() + static_cast<std::ptrdiff_t>(limit), indices.end(),
                      [&](std::size_t left, std::size_t right) { return values[left] > values[right]; });
    const float maximum = *std::max_element(values.begin() + static_cast<std::ptrdiff_t>(offset), values.end());
    double denominator{};
    for (auto iterator = values.begin() + static_cast<std::ptrdiff_t>(offset); iterator != values.end(); ++iterator) {
        denominator += std::exp(static_cast<double>(*iterator - maximum));
    }
    std::ostringstream result;
    result.setf(std::ios::fixed);
    result.precision(2);
    result << "Top ImageNet class indices:\n";
    for (std::size_t rank = 0; rank < limit; ++rank) {
        const auto index = indices[rank];
        const double probability = std::exp(static_cast<double>(values[index] - maximum)) / denominator * 100.0;
        result << rank + 1 << ". class " << index - offset << " — " << probability << "%\n";
    }
    result << "Class names are intentionally not guessed without a verified label map.";
    return result.str();
}

} // namespace localai::image_tensor
