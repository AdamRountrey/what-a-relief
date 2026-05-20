#pragma once

#include "types.hpp"

#include <opencv2/core.hpp>

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

std::vector<cv::Mat> loadLuminanceImages(const std::vector<std::string>& paths, bool srgb);
double readPixelScaleMmFromImage(const std::string& path);
cv::Mat loadDisplayImage(const std::string& path);
cv::Mat loadMask(const std::string& path, const cv::Size& size);
void applyCropToMask(cv::Mat& mask, const cv::Rect& crop);
void removeSphereFromMask(cv::Mat& mask, const Sphere& sphere);
void saveOutputs(
    const Options& opt,
    const std::vector<cv::Vec3f>& lights,
    const std::vector<HighlightEstimate>& estimates,
    const cv::Mat& normalMap,
    const cv::Mat& albedo,
    const cv::Mat& residual,
    const cv::Mat& validMask,
    const PhotometricDiagnostics& diagnostics,
    const cv::Mat& height,
    const cv::Mat& heightMask,
    const std::function<void(const std::string&)>& progress = {});
