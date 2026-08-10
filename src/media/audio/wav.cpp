#include "localai/audio.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>

namespace localai {
namespace {

template <typename T>
bool read(std::istream& input, T& value) {
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    return static_cast<bool>(input);
}

std::vector<float> resampleMono(const std::vector<float>& source, int sourceRate, int targetRate) {
    if (sourceRate == targetRate || source.empty()) return source;
    const auto outputCount = static_cast<std::size_t>(
        std::ceil(static_cast<double>(source.size()) * targetRate / sourceRate));
    std::vector<float> output(outputCount);
    const double scale = static_cast<double>(sourceRate) / targetRate;
    for (std::size_t index = 0; index < output.size(); ++index) {
        const double position = index * scale;
        const auto left = std::min(static_cast<std::size_t>(position), source.size() - 1);
        const auto right = std::min(left + 1, source.size() - 1);
        const float fraction = static_cast<float>(position - left);
        output[index] = source[left] * (1.0F - fraction) + source[right] * fraction;
    }
    return output;
}

void writeU16(std::ostream& output, std::uint16_t value) {
    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void writeU32(std::ostream& output, std::uint32_t value) {
    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

}

Result<PcmAudio> readWaveFile(const std::filesystem::path& path, int targetSampleRate) {
    std::ifstream input(path, std::ios::binary);
    std::array<char, 4> riff{};
    std::uint32_t fileSize{};
    std::array<char, 4> wave{};
    if (!input.read(riff.data(), 4) || !read(input, fileSize) || !input.read(wave.data(), 4) ||
        std::memcmp(riff.data(), "RIFF", 4) != 0 || std::memcmp(wave.data(), "WAVE", 4) != 0) {
        return Result<PcmAudio>::failure("audio.wave", "Input is not a RIFF/WAVE file");
    }
    static_cast<void>(fileSize);
    std::uint16_t format{};
    std::uint16_t channels{};
    std::uint32_t sampleRate{};
    std::uint16_t bits{};
    std::vector<std::byte> data;
    while (input && (!format || data.empty())) {
        std::array<char, 4> id{};
        std::uint32_t size{};
        if (!input.read(id.data(), 4) || !read(input, size) || size > 1024U * 1024U * 1024U) break;
        if (std::memcmp(id.data(), "fmt ", 4) == 0) {
            std::uint32_t bytesPerSecond{};
            std::uint16_t blockAlign{};
            if (size < 16 || !read(input, format) || !read(input, channels) || !read(input, sampleRate) ||
                !read(input, bytesPerSecond) || !read(input, blockAlign) || !read(input, bits)) {
                return Result<PcmAudio>::failure("audio.wave_format", "WAVE format chunk is malformed");
            }
            static_cast<void>(bytesPerSecond);
            static_cast<void>(blockAlign);
            if (size > 16) input.seekg(size - 16, std::ios::cur);
        } else if (std::memcmp(id.data(), "data", 4) == 0) {
            data.resize(size);
            input.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size));
        } else {
            input.seekg(size, std::ios::cur);
        }
        if (size & 1U) input.seekg(1, std::ios::cur);
    }
    if (data.empty() || channels == 0 || channels > 8 || sampleRate < 1000 || sampleRate > 768000) {
        return Result<PcmAudio>::failure("audio.wave_data", "WAVE audio format or data is invalid");
    }
    const std::size_t bytesPerSample = bits / 8;
    if ((format != 1 && format != 3) || bytesPerSample == 0 || data.size() % (bytesPerSample * channels) != 0) {
        return Result<PcmAudio>::failure("audio.wave_encoding", "Supported WAVE encodings are PCM integer and IEEE float");
    }
    const std::size_t frames = data.size() / (bytesPerSample * channels);
    std::vector<float> mono(frames);
    const auto* bytes = reinterpret_cast<const unsigned char*>(data.data());
    for (std::size_t frame = 0; frame < frames; ++frame) {
        double sum{};
        for (std::size_t channel = 0; channel < channels; ++channel) {
            const auto* sample = bytes + (frame * channels + channel) * bytesPerSample;
            float value{};
            if (format == 3 && bits == 32) {
                std::memcpy(&value, sample, 4);
            } else if (format == 1 && bits == 16) {
                std::int16_t integer{};
                std::memcpy(&integer, sample, 2);
                value = integer / 32768.0F;
            } else if (format == 1 && bits == 24) {
                std::int32_t integer = sample[0] | (sample[1] << 8) | (sample[2] << 16);
                if (integer & 0x800000) integer |= ~0xffffff;
                value = integer / 8388608.0F;
            } else if (format == 1 && bits == 32) {
                std::int32_t integer{};
                std::memcpy(&integer, sample, 4);
                value = integer / 2147483648.0F;
            } else {
                return Result<PcmAudio>::failure("audio.wave_depth", "WAVE bit depth is not supported");
            }
            sum += std::clamp(value, -1.0F, 1.0F);
        }
        mono[frame] = static_cast<float>(sum / channels);
    }
    PcmAudio output;
    output.samples = resampleMono(mono, static_cast<int>(sampleRate), targetSampleRate);
    output.sampleRate = targetSampleRate;
    output.channels = 1;
    return Result<PcmAudio>::success(std::move(output));
}

Result<bool> writeWaveFile(const std::filesystem::path& path, const std::vector<float>& samples,
                           int sampleRate, int channels) {
    if (sampleRate <= 0 || channels <= 0 || samples.size() > std::numeric_limits<std::uint32_t>::max() / 2) {
        return Result<bool>::failure("audio.output", "Invalid audio output parameters");
    }
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) return Result<bool>::failure("audio.output", "Unable to create output WAVE file");
    const auto dataBytes = static_cast<std::uint32_t>(samples.size() * sizeof(std::int16_t));
    output.write("RIFF", 4);
    writeU32(output, 36 + dataBytes);
    output.write("WAVEfmt ", 8);
    writeU32(output, 16);
    writeU16(output, 1);
    writeU16(output, static_cast<std::uint16_t>(channels));
    writeU32(output, static_cast<std::uint32_t>(sampleRate));
    writeU32(output, static_cast<std::uint32_t>(sampleRate * channels * sizeof(std::int16_t)));
    writeU16(output, static_cast<std::uint16_t>(channels * sizeof(std::int16_t)));
    writeU16(output, 16);
    output.write("data", 4);
    writeU32(output, dataBytes);
    for (const float sample : samples) {
        const auto integer = static_cast<std::int16_t>(std::lrint(std::clamp(sample, -1.0F, 1.0F) * 32767.0F));
        output.write(reinterpret_cast<const char*>(&integer), sizeof(integer));
    }
    return output ? Result<bool>::success(true)
                  : Result<bool>::failure("audio.output", "Failed while writing output WAVE data");
}

} // namespace localai
