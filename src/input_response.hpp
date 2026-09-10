#pragma once

#include "types.hpp"

InputResponse inspectInputResponse(const std::string& path);
InputResponse inputResponseForImage(const Options& opt, size_t index);
void resolveInputResponses(Options& opt);
std::string inputResponseSummary(const Options& opt);
const char* inputResponseModeName(InputResponseMode mode);
