#pragma once

#include "types.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

HighlightEstimate estimateHighlight(const cv::Mat& image, const Sphere& sphere, const Options& opt);
std::vector<cv::Vec3f> loadLightsFile(const std::string& path, size_t expectedCount);
void solvePhotometricStereo(
    const std::vector<cv::Mat>& images,
    const std::vector<cv::Vec3f>& lights,
    const cv::Mat& inputMask,
    float shadowThreshold,
    cv::Mat& normalMap,
    cv::Mat& albedo,
    cv::Mat& residual,
    cv::Mat& validMask);
cv::Mat integrateHeight(const cv::Mat& normalMap, const cv::Mat& validMask, int iterations);
