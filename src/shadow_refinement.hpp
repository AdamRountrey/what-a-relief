#pragma once

#include "types.hpp"

#include <opencv2/core.hpp>

#include <functional>
#include <string>
#include <vector>

struct ShadowRefinementSettings {
    LightingModel lightingModel = LightingModel::Directional;
    double ringLightRadiusMm = 10.0;
    double ringLightHeightMm = 10.0;
    double pixelScaleMm = 0.0;
    double referenceSurfaceZMm = 0.0;
    double ledDiameterMm = 0.0;
    cv::Point2d lightingCenter;
    int maximumCoarseSide = 160;
    int iterations = 5;
};

void refineHeightFromCastShadows(
    cv::Mat& height,
    const cv::Mat& receiverMask,
    const cv::Mat& occluderMask,
    const cv::Mat& normalMap,
    const std::vector<cv::Mat>& images,
    const std::vector<cv::Vec3f>& lights,
    const ShadowRefinementSettings& settings,
    PhotometricDiagnostics& diagnostics,
    const std::function<void(const std::string&)>& progress = {});
