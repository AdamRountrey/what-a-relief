#pragma once

#include "types.hpp"

#include <opencv2/core.hpp>

#include <functional>
#include <string>
#include <vector>

HighlightEstimate estimateHighlight(const cv::Mat& image, const Sphere& sphere, const Options& opt);
bool loadLightsFileMetadata(const std::string& path, Options& opt);
std::vector<cv::Vec3f> loadLightsFile(const std::string& path, size_t expectedCount, Options* opt = nullptr);
void solvePhotometricStereo(
    const std::vector<cv::Mat>& images,
    const std::vector<cv::Vec3f>& lights,
    const cv::Mat& inputMask,
    float shadowThreshold,
    NormalSolverMode solverMode,
    float highOutlierThreshold,
    LightingModel lightingModel,
    double ringLightRadiusMm,
    double ringLightHeightMm,
    double pixelScaleMm,
    cv::Point2d lightingCenter,
    cv::Mat& normalMap,
    cv::Mat& albedo,
    cv::Mat& residual,
    cv::Mat& validMask,
    PhotometricDiagnostics& diagnostics);
void solveUncalibratedPhotometricStereo(
    const std::vector<cv::Mat>& images,
    const cv::Mat& inputMask,
    float shadowThreshold,
    cv::Mat& normalMap,
    cv::Mat& albedo,
    cv::Mat& residual,
    cv::Mat& validMask,
    const std::function<void(const std::string&)>& progress = {});
cv::Mat integrateHeight(
    const cv::Mat& normalMap,
    const cv::Mat& validMask,
    HeightSolverMode solverMode,
    double slopeCap,
    int iterations,
    const std::function<void(int, int)>& progress = {});
void flattenNormalField(cv::Mat& normalMap, const cv::Mat& validMask, FlattenMode mode);
void removeBestFitPlane(cv::Mat& height, const cv::Mat& validMask);
void removeHeightCurl(cv::Mat& height, const cv::Mat& validMask, HeightFlattenMode mode);
