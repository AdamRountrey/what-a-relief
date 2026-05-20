#pragma once

#include "types.hpp"

#include <opencv2/core.hpp>

#include <functional>
#include <string>
#include <vector>

void exportRtiPackage(
    const Options& opt,
    const std::vector<cv::Vec3f>& lights,
    const cv::Size& expectedSize,
    const std::function<void(const std::string&)>& progress = {});
