#pragma once

#include <opencv2/core.hpp>

#include <vector>

cv::Mat convertToLinearLuminance(const cv::Mat& input, bool srgb);
cv::Mat convertToLinearColor(const cv::Mat& input, bool srgb);
double normalizeRelativeIntensityStack(
    std::vector<cv::Mat>& images,
    bool scalePeakToOne = true);
