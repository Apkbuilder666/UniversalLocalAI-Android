#pragma once

#include "localai/types.hpp"

#include <filesystem>
#include <vector>

namespace localai {

struct PcmAudio {
    std::vector<float> samples;
    int sampleRate{};
    int channels{};
};

Result<PcmAudio> readWaveFile(const std::filesystem::path& path, int targetSampleRate = 16000);
Result<bool> writeWaveFile(const std::filesystem::path& path, const std::vector<float>& samples,
                           int sampleRate, int channels = 1);

} // namespace localai
