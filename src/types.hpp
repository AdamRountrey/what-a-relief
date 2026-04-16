#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

struct Sphere {
    double cx = 0.0;
    double cy = 0.0;
    double radius = 0.0;
};

struct Options {
    std::vector<std::string> imagePaths;
    std::string outputDir = "out";
    std::string maskPath;
    std::string lightsFile;
    Sphere sphere;
    bool hasSphere = false;
    bool noGui = false;
    bool srgb = false;
    bool keepSphere = false;
    double highlightPercentile = 99.8;
    double minHighlight = 0.05;
    double shadowThreshold = 0.02;
    int integrationIterations = 800;
    cv::Vec3f viewDir = cv::Vec3f(0.0f, 0.0f, 1.0f);
};

struct HighlightEstimate {
    cv::Point2f point = cv::Point2f(0.0f, 0.0f);
    cv::Vec3f light = cv::Vec3f(0.0f, 0.0f, 1.0f);
    float threshold = 0.0f;
    float peak = 0.0f;
    int selectedPixels = 0;
};
