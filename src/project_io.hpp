#pragma once

#include "types.hpp"
#include <filesystem>
#include <string>
#include <vector>

struct LoadedProject {
    Options options;
    std::string manifestPath;
    std::string completedOutputDir;
    std::vector<std::string> warnings;
};

LoadedProject loadCompletedProject(const std::filesystem::path& manifestOrDirectory);
std::string freshProjectOutputDirectory(const std::filesystem::path& requested);
