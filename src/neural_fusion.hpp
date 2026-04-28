#pragma once

#include "types.hpp"

#include <opencv2/core.hpp>

#include <vector>

void applyNeuralFusion(
    const Options& opt,
    const std::vector<cv::Mat>& images,
    const std::vector<cv::Vec3f>& lights,
    const cv::Mat& solveMask,
    cv::Point2d lightingCenter,
    cv::Mat& normalMap,
    cv::Mat& albedo,
    cv::Mat& residual,
    cv::Mat& validMask,
    PhotometricDiagnostics& diagnostics);
