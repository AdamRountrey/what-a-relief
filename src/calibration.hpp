#pragma once

#include "types.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <string>
#include <vector>

struct MicroscopeCalibration {
    int imageWidth = 0;
    int imageHeight = 0;
    int gridColumns = 0;
    int gridRows = 0;
    double squareSizeMm = 0.0;
    double pixelScaleMm = 0.0;
    double reprojectionRmsPixels = 0.0;
    double reprojectionMaxPixels = 0.0;
    double boardCoverageFraction = 0.0;
    std::array<double, 10> sourceX{};
    std::array<double, 10> sourceY{};
    std::vector<double> lightGains;
    std::vector<std::string> calibrationImageNames;
};

struct LightGainEstimate {
    bool attempted = false;
    bool accepted = false;
    std::string decision = "not_requested";
    std::vector<double> gains;
    double validationErrorBefore = -1.0;
    double validationErrorAfter = -1.0;
    double stability = -1.0;
    double normalDiversity = -1.0;
    int sampledPixels = 0;
};

struct MicroscopeRectificationMaps {
    cv::Mat sourceCoordinates;
};

struct LightGainGeometry {
    LightingModel model = LightingModel::Directional;
    double ringLightRadiusMm = 0.0;
    double ringLightHeightMm = 0.0;
    double pixelScaleMm = 0.0;
    cv::Point2d lightingCenter;
};

void writePrintableCheckerboardSvg(
    const std::string& path,
    int gridColumns,
    int gridRows,
    double squareSizeMm,
    double pageWidthMm,
    double pageHeightMm);

MicroscopeCalibration createMicroscopeCalibration(
    const Options& calibrationImages,
    int gridColumns,
    int gridRows,
    double squareSizeMm);
void saveMicroscopeCalibration(const std::string& path, const MicroscopeCalibration& calibration);
MicroscopeCalibration loadMicroscopeCalibration(const std::string& path);
void validateMicroscopeCalibration(
    const MicroscopeCalibration& calibration,
    const cv::Size& imageSize,
    size_t lightCount);
MicroscopeRectificationMaps buildMicroscopeRectificationMaps(
    const MicroscopeCalibration& calibration);
cv::Mat rectifyMicroscopeImage(
    const cv::Mat& image,
    const MicroscopeRectificationMaps& maps,
    int interpolation = 1);
cv::Mat rectifyMicroscopeImage(
    const cv::Mat& image,
    const MicroscopeCalibration& calibration,
    int interpolation = 1);

void applyLightGainCorrection(std::vector<cv::Mat>& images, const std::vector<double>& gains);
LightGainEstimate estimateRelativeLightGains(
    const std::vector<cv::Mat>& images,
    const std::vector<cv::Vec3f>& lights,
    const cv::Mat& mask,
    float shadowThreshold,
    float highOutlierThreshold,
    const std::vector<double>& priorGains = {},
    LightGainGeometry geometry = {},
    const std::vector<cv::Mat>& saturationMasks = {});
