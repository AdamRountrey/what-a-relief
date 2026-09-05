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

enum class HeightSolverMode {
    RobustMasked,
    FastDct
};

enum class HeightFlattenMode {
    None,
    Plane,
    Radial,
    Quadratic
};

enum class RtiLayoutMode {
    Image,
    DeepZoom,
    WebRtiViewer
};

enum class RtiColorMode {
    Rgb,
    Lrgb
};

enum class MitsubaBackendMode {
    Auto,
    Cuda,
    Cpu
};

enum class MitsubaQualityMode {
    Preview,
    Standard,
    Research
};

struct MitsubaRefinementDiagnostics {
    bool attempted = false;
    bool succeeded = false;
    bool accepted = false;
    std::string status = "not_requested";
    std::string decision = "not_requested";
    std::string requestedBackend = "auto";
    std::string selectedBackend;
    std::string variant;
    std::string optimizer;
    std::string mitsubaVersion;
    std::string drjitVersion;
    std::string numpyVersion;
    std::string pythonVersion;
    std::string resultPath;
    int iterationsCompleted = 0;
    int renderWidth = 0;
    int renderHeight = 0;
    double trainLossBefore = -1.0;
    double trainLossAfter = -1.0;
    double holdoutLossBefore = -1.0;
    double holdoutLossAfter = -1.0;
    double correctionRmsPixels = 0.0;
    double correctionMaximumPixels = 0.0;
};

struct Options {
    std::vector<std::string> imagePaths;
    std::string outputDir = "out";
    std::string maskPath;
    std::string heightMaskPath;
    std::string lightsFile;
    std::string meshPath;
    std::string printableMeshPath;
    std::string rtiPath;
    Sphere sphere;
    cv::Rect crop;
    bool hasSphere = false;
    bool hasCrop = false;
    bool hasHeightMask = false;
    bool guiMode = false;
    bool noGui = false;
    bool srgb = false;
    bool keepSphere = false;
    bool uncalibratedLighting = false;
    bool lightsFileByOrder = false;
    bool calculateHeight = true;
    bool openRelightViewer = false;
    bool specularDiagnostics = false;
    bool neuralFusion = false;
    bool exportRti = false;
    bool printableFillHoles = false;
    bool shadowHeightRefinement = false;
    bool mitsubaInverseRefinement = false;
    NormalSolverMode solverMode = NormalSolverMode::Robust;
    FlattenMode flattenMode = FlattenMode::None;
    LightingModel lightingModel = LightingModel::Directional;
    HeightSolverMode heightSolverMode = HeightSolverMode::RobustMasked;
    HeightFlattenMode heightFlattenMode = HeightFlattenMode::None;
    RtiLayoutMode rtiLayoutMode = RtiLayoutMode::Image;
    RtiColorMode rtiColorMode = RtiColorMode::Rgb;
    MitsubaBackendMode mitsubaBackendMode = MitsubaBackendMode::Auto;
    MitsubaQualityMode mitsubaQualityMode = MitsubaQualityMode::Standard;
    std::string neuralModelPath;
    std::string mitsubaPythonPath;
    std::string mitsubaWorkerPath;
    double highlightPercentile = 99.8;
    double minHighlight = 0.05;
    double shadowThreshold = 0.02;
    double highOutlierThreshold = 0.98;
    double ringLightRadiusMm = 10.0;
    double ringLightHeightMm = 10.0;
    double pixelScaleMm = 0.0;
    double shadowReferenceZMm = 0.0;
    double shadowLedDiameterMm = 0.0;
    int integrationIterations = 800;
    int meshStep = 1;
    int neuralMaxSide = 2048;
    double heightSlopeCap = 3.0;
    double heightScale = 1.0;
    double printableThicknessMm = 2.0;
    cv::Vec3f viewDir = cv::Vec3f(0.0f, 0.0f, 1.0f);
    cv::Mat heightMask;
};

struct PhotometricDiagnostics {
    bool collectObservationMasks = false;
    double lightingConditionNumber = 0.0;
    double solvedFraction = 0.0;
    cv::Mat robustWeight;
    cv::Mat robustFallbackMask;
    cv::Mat unsupportedMask;
    cv::Mat effectiveInlierCount;
    cv::Mat localConditionNumber;
    cv::Mat shadowCount;
    cv::Mat highlightOutlierCount;
    cv::Mat saturationCount;
    cv::Mat modelMismatchCount;
    cv::Mat specularCueMask;
    std::vector<cv::Mat> shadowObservationMasks;
    std::vector<cv::Mat> shadowObservationConfidence;
    std::vector<cv::Mat> highlightObservationMasks;
    std::vector<cv::Mat> saturationObservationMasks;
    cv::Mat classicalConfidence;
    cv::Mat classicalValidMask;
    cv::Mat classicalNormal;
    cv::Mat neuralValidMask;
    cv::Mat neuralNormal;
    bool shadowHeightRefinementApplied = false;
    std::string shadowHeightRefinementDecision = "not_requested";
    double shadowMismatchRateBefore = -1.0;
    double shadowMismatchRateAfter = -1.0;
    double shadowHoldoutMismatchRateBefore = -1.0;
    double shadowHoldoutMismatchRateAfter = -1.0;
    double shadowNormalSlopeRmsBefore = -1.0;
    double shadowNormalSlopeRmsAfter = -1.0;
    double shadowCorrectionRms = 0.0;
    int shadowRefinementConstraintCount = 0;
    int shadowRefinementShadowSamples = 0;
    int shadowRefinementLitSamples = 0;
    cv::Mat shadowHeightCorrection;
    cv::Mat shadowConstraintCount;
    cv::Mat shadowMismatchBefore;
    cv::Mat shadowMismatchAfter;
    cv::Mat shadowObservability;
    cv::Mat shadowEdgeSupport;
    cv::Mat shadowOccluderSupport;
    double shadowSelectedStepFraction = 0.0;
    double shadowWorstHoldoutDelta = 0.0;
    std::vector<int> shadowRefinementLightIndices;
    std::vector<cv::Mat> shadowObservedCastMasks;
    std::vector<cv::Mat> shadowPredictedBeforeMasks;
    std::vector<cv::Mat> shadowPredictedAfterMasks;
    std::vector<cv::Mat> shadowPredictedBeforeProbability;
    std::vector<cv::Mat> shadowPredictedAfterProbability;
    MitsubaRefinementDiagnostics mitsuba;
};

struct HighlightEstimate {
    cv::Point2f point = cv::Point2f(0.0f, 0.0f);
    cv::Vec3f light = cv::Vec3f(0.0f, 0.0f, 1.0f);
    float threshold = 0.0f;
    float peak = 0.0f;
    int selectedPixels = 0;
};
