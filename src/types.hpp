#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

struct Sphere {
    double cx = 0.0;
    double cy = 0.0;
    double radius = 0.0;
};

enum class NormalSolverMode {
    Standard,
    Robust
};

enum class FlattenMode {
    None,
    Gentle,
    Strong
};

enum class LightingModel {
    Directional,
    NearFieldRing
};

struct Options {
    std::vector<std::string> imagePaths;
    std::string outputDir = "out";
    std::string maskPath;
    std::string lightsFile;
    std::string meshPath;
    Sphere sphere;
    cv::Rect crop;
    bool hasSphere = false;
    bool hasCrop = false;
    bool guiMode = false;
    bool noGui = false;
    bool srgb = false;
    bool keepSphere = false;
    bool uncalibratedLighting = false;
    bool calculateHeight = true;
    bool openRelightViewer = false;
    bool specularDiagnostics = false;
    NormalSolverMode solverMode = NormalSolverMode::Robust;
    FlattenMode flattenMode = FlattenMode::None;
    LightingModel lightingModel = LightingModel::Directional;
    double highlightPercentile = 99.8;
    double minHighlight = 0.05;
    double shadowThreshold = 0.02;
    double highOutlierThreshold = 0.98;
    double ringLightRadiusMm = 10.0;
    double ringLightHeightMm = 10.0;
    double pixelScaleMm = 0.0;
    int integrationIterations = 800;
    int meshStep = 1;
    double heightScale = 1.0;
    cv::Vec3f viewDir = cv::Vec3f(0.0f, 0.0f, 1.0f);
};

struct PhotometricDiagnostics {
    cv::Mat robustWeight;
    cv::Mat shadowCount;
    cv::Mat highlightOutlierCount;
    cv::Mat specularCueMask;
};

struct HighlightEstimate {
    cv::Point2f point = cv::Point2f(0.0f, 0.0f);
    cv::Vec3f light = cv::Vec3f(0.0f, 0.0f, 1.0f);
    float threshold = 0.0f;
    float peak = 0.0f;
    int selectedPixels = 0;
};
