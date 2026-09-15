#pragma once

#include "types.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

struct SphereSelection {
    Sphere sphere;
    size_t imageIndex = 0;
    std::vector<cv::Point2d> edgePoints;
    double fitRmsPixels = -1.0;
    double fitMaxResidualPixels = -1.0;
    double fitCoverageDegrees = -1.0;
};

using SphereImageLoader = std::function<cv::Mat(size_t)>;

SphereSelection chooseSphereInteractive(
    size_t imageCount,
    const SphereImageLoader& loader,
    const std::vector<std::string>& imageLabels = {},
    size_t initialImageIndex = 0);

Sphere chooseSphereInteractive(const cv::Mat& displayImage);
