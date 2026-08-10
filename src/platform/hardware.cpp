#include "localai/hardware.hpp"

#include "localai/backend.hpp"
#ifdef __ANDROID__
#include "localai/android_hardware.hpp"
#endif

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <thread>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <intrin.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#else
#include <sys/sysinfo.h>
#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif
#endif

namespace localai {
namespace {

std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n\0", 0);
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r\n\0");
    return value.substr(first, last - first + 1);
}

std::string cpuBrand() {
#if defined(__ANDROID__)
    return androidProcessorName();
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    std::array<int, 4> registers{};
    std::array<char, 49> brand{};
    for (int leaf = 0x80000002, offset = 0; leaf <= 0x80000004; ++leaf, offset += 16) {
        __cpuid(registers.data(), leaf);
        std::memcpy(brand.data() + offset, registers.data(), 16);
    }
    return trim(brand.data());
#elif (defined(__x86_64__) || defined(__i386__)) && !defined(__APPLE__)
    std::array<unsigned, 4> registers{};
    std::array<char, 49> brand{};
    for (unsigned leaf = 0x80000002, offset = 0; leaf <= 0x80000004; ++leaf, offset += 16) {
        __get_cpuid(leaf, &registers[0], &registers[1], &registers[2], &registers[3]);
        std::memcpy(brand.data() + offset, registers.data(), 16);
    }
    return trim(brand.data());
#elif defined(__APPLE__)
    std::size_t size{};
    sysctlbyname("machdep.cpu.brand_string", nullptr, &size, nullptr, 0);
    if (size == 0) sysctlbyname("hw.model", nullptr, &size, nullptr, 0);
    std::string value(size, '\0');
    if (size && sysctlbyname("machdep.cpu.brand_string", value.data(), &size, nullptr, 0) == 0) return trim(value);
    return "Apple processor";
#else
    std::ifstream input("/proc/cpuinfo");
    std::string line;
    while (std::getline(input, line)) {
        const auto separator = line.find(':');
        if (separator != std::string::npos &&
            (line.starts_with("model name") || line.starts_with("Hardware"))) {
            return trim(line.substr(separator + 1));
        }
    }
    return "Unknown processor";
#endif
}

MemoryInfo memoryInfo() {
    MemoryInfo result;
#ifdef _WIN32
    MEMORYSTATUSEX state{};
    state.dwLength = sizeof(state);
    if (GlobalMemoryStatusEx(&state)) {
        result.totalBytes = state.ullTotalPhys;
        result.availableBytes = state.ullAvailPhys;
    }
#elif defined(__APPLE__)
    std::size_t size = sizeof(result.totalBytes);
    sysctlbyname("hw.memsize", &result.totalBytes, &size, nullptr, 0);
    std::uint64_t pageSize{};
    size = sizeof(pageSize);
    sysctlbyname("hw.pagesize", &pageSize, &size, nullptr, 0);
    result.availableBytes = result.totalBytes;
#else
    struct sysinfo state {};
    if (sysinfo(&state) == 0) {
        result.totalBytes = static_cast<std::uint64_t>(state.totalram) * state.mem_unit;
        result.availableBytes = static_cast<std::uint64_t>(state.freeram + state.bufferram) * state.mem_unit;
    }
#endif
    return result;
}

unsigned physicalCoreCount(unsigned fallback) {
#ifdef __APPLE__
    unsigned value{};
    std::size_t size = sizeof(value);
    return sysctlbyname("hw.physicalcpu", &value, &size, nullptr, 0) == 0 && value ? value : fallback;
#elif defined(__linux__) && !defined(__ANDROID__)
    std::ifstream input("/proc/cpuinfo");
    std::set<std::pair<int, int>> cores;
    std::string line;
    int package = 0;
    int core = -1;
    auto commit = [&] { if (core >= 0) cores.emplace(package, core); core = -1; };
    while (std::getline(input, line)) {
        if (line.empty()) { commit(); continue; }
        const auto separator = line.find(':');
        if (separator == std::string::npos) continue;
        const auto key = trim(line.substr(0, separator));
        try {
            if (key == "physical id") package = std::stoi(trim(line.substr(separator + 1)));
            else if (key == "core id") core = std::stoi(trim(line.substr(separator + 1)));
        } catch (...) {}
    }
    commit();
    return cores.empty() ? fallback : static_cast<unsigned>(cores.size());
#else
    return fallback;
#endif
}

std::vector<std::string> instructionSets() {
    std::vector<std::string> values;
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f")) values.emplace_back("AVX-512");
    if (__builtin_cpu_supports("avx2")) values.emplace_back("AVX2");
    if (__builtin_cpu_supports("avx")) values.emplace_back("AVX");
    if (__builtin_cpu_supports("sse4.2")) values.emplace_back("SSE4.2");
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    std::array<int, 4> registers{};
    __cpuidex(registers.data(), 7, 0);
    if (registers[1] & (1 << 16)) values.emplace_back("AVX-512");
    if (registers[1] & (1 << 5)) values.emplace_back("AVX2");
    __cpuid(registers.data(), 1);
    if (registers[2] & (1 << 28)) values.emplace_back("AVX");
    if (registers[2] & (1 << 20)) values.emplace_back("SSE4.2");
#endif
#if defined(__ARM_FEATURE_DOTPROD)
    values.emplace_back("ARM dot product");
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    values.emplace_back("NEON");
#endif
    return values;
}

AcceleratorType inferType(const BackendInfo& info) {
    std::string value = info.provider + ' ' + info.device;
    std::ranges::transform(value, value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (value.find("npu") != std::string::npos || value.find("neural") != std::string::npos) return AcceleratorType::Npu;
    if (value.find("cuda") != std::string::npos || value.find("vulkan") != std::string::npos ||
        value.find("metal") != std::string::npos || value.find("gpu") != std::string::npos ||
        value.find("directml") != std::string::npos || value.find("rocm") != std::string::npos) {
        return AcceleratorType::Gpu;
    }
    return AcceleratorType::Cpu;
}

void appendLinuxGraphicsDevices(std::vector<ExecutionDevice>& devices) {
#if defined(__linux__) && !defined(__ANDROID__)
    const std::filesystem::path drm("/sys/class/drm");
    std::error_code error;
    if (!std::filesystem::is_directory(drm, error)) return;
    std::set<std::string> seen;
    for (const auto& entry : std::filesystem::directory_iterator(drm, error)) {
        const auto name = entry.path().filename().string();
        if (!name.starts_with("card") || name.find('-') != std::string::npos) continue;
        auto readValue = [](const std::filesystem::path& path) {
            std::ifstream input(path);
            std::string value;
            std::getline(input, value);
            return trim(value);
        };
        const auto vendorId = readValue(entry.path() / "device" / "vendor");
        const auto deviceId = readValue(entry.path() / "device" / "device");
        if (vendorId.empty() || !seen.insert(vendorId + deviceId).second) continue;
        std::string vendor = "PCI vendor " + vendorId;
        if (vendorId == "0x10de") vendor = "NVIDIA";
        else if (vendorId == "0x1002") vendor = "AMD";
        else if (vendorId == "0x8086") vendor = "Intel";
        ExecutionDevice device;
        device.id = "drm:" + name;
        device.name = vendor + " graphics " + deviceId;
        device.type = AcceleratorType::Gpu;
        device.vendor = vendor;
        device.runtime = "Linux DRM hardware discovery";
        device.available = true;
        device.initialized = false;
        device.status = "Hardware detected; no inference provider verified for this entry";
        devices.push_back(std::move(device));
    }
#else
    static_cast<void>(devices);
#endif
}

}

HardwareSnapshot HardwareDiscovery::discover() {
    HardwareSnapshot snapshot;
#if defined(_M_X64) || defined(__x86_64__)
    snapshot.cpu.architecture = "x86-64";
#elif defined(_M_ARM64) || defined(__aarch64__)
    snapshot.cpu.architecture = "ARM64";
#else
    snapshot.cpu.architecture = "Unsupported architecture";
#endif
    snapshot.cpu.model = cpuBrand();
#ifdef __ANDROID__
    snapshot.cpu.vendor = androidProcessorVendor();
#else
    if (snapshot.cpu.model.find("AMD") != std::string::npos) snapshot.cpu.vendor = "AMD";
    else if (snapshot.cpu.model.find("Intel") != std::string::npos) snapshot.cpu.vendor = "Intel";
    else if (snapshot.cpu.model.find("Apple") != std::string::npos) snapshot.cpu.vendor = "Apple";
    else snapshot.cpu.vendor = "Unknown";
#endif
    snapshot.cpu.logicalCores = std::max(1U, std::thread::hardware_concurrency());
    snapshot.cpu.physicalCores = physicalCoreCount(snapshot.cpu.logicalCores);
    snapshot.cpu.instructionSets = instructionSets();
    snapshot.memory = memoryInfo();

    ExecutionDevice cpu;
    cpu.id = "cpu";
    cpu.name = snapshot.cpu.model;
    cpu.type = AcceleratorType::Cpu;
    cpu.vendor = snapshot.cpu.vendor;
    cpu.runtime = "Native CPU";
    cpu.availableMemoryBytes = snapshot.memory.availableBytes;
    cpu.available = true;
    cpu.initialized = true;
    cpu.status = "Ready";
    snapshot.devices.push_back(std::move(cpu));
#ifdef __ANDROID__
    appendAndroidAccelerators(snapshot.devices);
#else
    appendLinuxGraphicsDevices(snapshot.devices);
#endif

    registerCompiledBackends();
    for (const auto& backend : BackendRegistry::instance().probe()) {
        ExecutionDevice device;
        device.id = backend.id + ':' + backend.provider;
        device.name = backend.device.empty() ? backend.name : backend.device;
        device.type = inferType(backend);
        device.runtime = backend.runtime;
        device.formats = backend.formats;
        device.available = backend.available;
        device.initialized = backend.initialized;
        device.status = backend.verification;
        snapshot.devices.push_back(std::move(device));
    }
    return snapshot;
}

} // namespace localai
