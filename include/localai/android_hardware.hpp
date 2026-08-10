#pragma once

#include "localai/types.hpp"

#include <string>
#include <vector>

namespace localai {

std::string androidProcessorName();
std::string androidProcessorVendor();
void appendAndroidAccelerators(std::vector<ExecutionDevice>& devices);

} // namespace localai
