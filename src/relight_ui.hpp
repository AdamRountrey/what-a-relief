#pragma once

#include <opencv2/core.hpp>

#include <string>

void launchRelightViewer(const cv::Mat& normalMap, const cv::Mat& validMask, const std::string& outputDir);
