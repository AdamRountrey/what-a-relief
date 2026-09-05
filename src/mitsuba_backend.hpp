#pragma once

#include "types.hpp"

#include <opencv2/core.hpp>

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

struct MitsubaBackendPaths {
    std::filesystem::path python;
    std::filesystem::path worker;
    std::string problem;

    bool available() const {
        return problem.empty() && !python.empty() && !worker.empty();
    }
};

MitsubaBackendPaths resolveMitsubaBackend(const Options& opt);
std::string describeMitsubaBackend(const Options& opt);

void runMitsubaInverseRefinement(
    const Options& opt,
    const std::vector<cv::Vec3f>& lights,
    const std::vector<cv::Mat>& images,
    const cv::Mat& albedo,
    const cv::Mat& height,
    const cv::Mat& heightMask,
    PhotometricDiagnostics& diagnostics,
    const std::function<void(const std::string&, int)>& progress = {},
    const std::function<bool()>& cancellationRequested = {});
