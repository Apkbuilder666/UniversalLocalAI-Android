#pragma once

#include "localai/types.hpp"

#include <string>
#include <vector>

namespace localai {

struct CpuInfo {
    std::string vendor;
    std::string model;
    std::string architecture;
    unsigned logicalCores{};
    unsigned physicalCores{};
    std::vector<std::string> instructionSets;
};

struct MemoryInfo {
    std::uint64_t totalBytes{};
    std::uint64_t availableBytes{};
};

struct HardwareSnapshot {
    CpuInfo cpu;
    MemoryInfo memory;
    std::vector<ExecutionDevice> devices;
};

class HardwareDiscovery {
public:
    static HardwareSnapshot discover();
};

} // namespace localai
