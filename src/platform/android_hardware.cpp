#include "localai/android_hardware.hpp"

#include <android/dlext.h>
#include <dlfcn.h>
#include <sys/system_properties.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace localai {
namespace {

std::string property(const char* name) {
    std::array<char, PROP_VALUE_MAX> value{};
    const int size = __system_property_get(name, value.data());
    return size > 0 ? std::string(value.data(), static_cast<std::size_t>(size)) : std::string{};
}

std::string firstProperty(std::initializer_list<const char*> names) {
    for (const auto* name : names) {
        auto value = property(name);
        if (!value.empty() && value != "unknown") return value;
    }
    return {};
}

std::string vulkanVendor(std::uint32_t id) {
    switch (id) {
    case 0x1002: return "AMD";
    case 0x1010: return "Imagination Technologies";
    case 0x10DE: return "NVIDIA";
    case 0x13B5: return "Arm";
    case 0x5143: return "Qualcomm";
    case 0x8086: return "Intel";
    case 0x144D: return "Samsung";
    default: return "Vulkan vendor 0x" + [&] {
        std::ostringstream stream;
        stream << std::hex << id;
        return stream.str();
    }();
    }
}

void appendVulkanDevices(std::vector<ExecutionDevice>& devices) {
    VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application.pApplicationName = "Universal Local AI";
    application.applicationVersion = VK_MAKE_VERSION(0, 3, 0);
    application.pEngineName = "GGML";
    application.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo createInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    createInfo.pApplicationInfo = &application;
    VkInstance instance{};
    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) return;

    std::uint32_t count{};
    if (vkEnumeratePhysicalDevices(instance, &count, nullptr) != VK_SUCCESS || count == 0) {
        vkDestroyInstance(instance, nullptr);
        return;
    }
    std::vector<VkPhysicalDevice> physicalDevices(count);
    if (vkEnumeratePhysicalDevices(instance, &count, physicalDevices.data()) != VK_SUCCESS) {
        vkDestroyInstance(instance, nullptr);
        return;
    }
    for (std::uint32_t index = 0; index < count; ++index) {
        VkPhysicalDeviceProperties properties{};
        VkPhysicalDeviceMemoryProperties memory{};
        vkGetPhysicalDeviceProperties(physicalDevices[index], &properties);
        vkGetPhysicalDeviceMemoryProperties(physicalDevices[index], &memory);
        std::uint64_t localBytes{};
        for (std::uint32_t heap = 0; heap < memory.memoryHeapCount; ++heap) {
            if ((memory.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0)
                localBytes += memory.memoryHeaps[heap].size;
        }
        ExecutionDevice device;
        device.id = "android-vulkan:" + std::to_string(properties.vendorID) + ':' +
                    std::to_string(properties.deviceID);
        device.name = properties.deviceName;
        device.type = AcceleratorType::Gpu;
        device.vendor = vulkanVendor(properties.vendorID);
        device.runtime = "Android Vulkan " + std::to_string(VK_VERSION_MAJOR(properties.apiVersion)) + '.' +
                         std::to_string(VK_VERSION_MINOR(properties.apiVersion));
        if (localBytes > 0) device.availableMemoryBytes = localBytes;
        device.formats = {ModelFormat::Gguf, ModelFormat::SafeTensors};
        device.precisions = {"FP32", "FP16", "quantized integer"};
        device.available = true;
        device.initialized = false;
        device.status = "Vulkan device detected; model execution is verified only after a GGML backend runs";
        devices.push_back(std::move(device));
    }
    vkDestroyInstance(instance, nullptr);
}

void appendNnapiDevices(std::vector<ExecutionDevice>& devices) {
    void* library = dlopen("libneuralnetworks.so", RTLD_NOW | RTLD_LOCAL);
    if (!library) return;
    using GetCount = int (*)(std::uint32_t*);
    using GetDevice = int (*)(std::uint32_t, const void**);
    using GetName = int (*)(const void*, const char**);
    using GetType = int (*)(const void*, std::int32_t*);
    using GetFeatureLevel = int (*)(const void*, std::int64_t*);
    const auto getCount = reinterpret_cast<GetCount>(dlsym(library, "ANeuralNetworks_getDeviceCount"));
    const auto getDevice = reinterpret_cast<GetDevice>(dlsym(library, "ANeuralNetworks_getDevice"));
    const auto getName = reinterpret_cast<GetName>(dlsym(library, "ANeuralNetworksDevice_getName"));
    const auto getType = reinterpret_cast<GetType>(dlsym(library, "ANeuralNetworksDevice_getType"));
    const auto getFeatureLevel = reinterpret_cast<GetFeatureLevel>(
        dlsym(library, "ANeuralNetworksDevice_getFeatureLevel"));
    if (!getCount || !getDevice || !getName || !getType || !getFeatureLevel) {
        dlclose(library);
        return;
    }
    std::uint32_t count{};
    if (getCount(&count) != 0) {
        dlclose(library);
        return;
    }
    for (std::uint32_t index = 0; index < count; ++index) {
        const void* handle{};
        const char* name{};
        std::int32_t type{};
        std::int64_t featureLevel{};
        if (getDevice(index, &handle) != 0 || !handle || getName(handle, &name) != 0 || !name ||
            getType(handle, &type) != 0 || getFeatureLevel(handle, &featureLevel) != 0) continue;
        if (type != 2 && type != 3) continue;
        ExecutionDevice device;
        device.id = "android-nnapi:" + std::to_string(index);
        device.name = name;
        device.type = type == 2 ? AcceleratorType::Gpu : AcceleratorType::Npu;
        device.vendor = androidProcessorVendor();
        device.runtime = "Android NNAPI feature level " + std::to_string(featureLevel);
        device.formats = {ModelFormat::Onnx, ModelFormat::LiteRt};
        device.precisions = {"FP32", "FP16", "INT8"};
        device.available = true;
        device.initialized = false;
        device.status = "NNAPI driver detected; graph placement is verified only after delegate/provider initialization";
        devices.push_back(std::move(device));
    }
    dlclose(library);
}

} // namespace

std::string androidProcessorName() {
    const auto soc = firstProperty({"ro.soc.model", "ro.board.platform", "ro.hardware"});
    const auto product = firstProperty({"ro.product.marketname", "ro.product.model", "ro.product.device"});
    if (!soc.empty() && !product.empty()) return product + " / " + soc;
    if (!soc.empty()) return soc;
    if (!product.empty()) return product;
    return "Android ARM64 processor";
}

std::string androidProcessorVendor() {
    auto vendor = firstProperty({"ro.soc.manufacturer", "ro.product.manufacturer", "ro.product.brand"});
    if (vendor.empty()) vendor = "Android device vendor";
    return vendor;
}

void appendAndroidAccelerators(std::vector<ExecutionDevice>& devices) {
    appendVulkanDevices(devices);
    appendNnapiDevices(devices);
}

} // namespace localai
