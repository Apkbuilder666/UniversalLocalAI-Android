#pragma once

#include "localai/types.hpp"

#include <filesystem>
#include <vector>

namespace localai {

class ModelInspector {
public:
    static Result<ModelDescriptor> inspect(const std::filesystem::path& path);
};

class ModelCatalog {
public:
    explicit ModelCatalog(std::filesystem::path databasePath);
    Result<ModelDescriptor> importModel(const std::filesystem::path& path);
    bool remove(std::string_view id);
    const std::vector<ModelDescriptor>& models() const { return models_; }
    Result<bool> load();
    Result<bool> save() const;

private:
    std::filesystem::path databasePath_;
    std::vector<ModelDescriptor> models_;
};

} // namespace localai
