#pragma once

#include "types.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

struct RunManifestContext {
    std::string startedUtc;
};

RunManifestContext beginRunManifest(const Options& opt);

void completeRunManifest(
    const Options& opt,
    const RunManifestContext& context,
    const std::vector<cv::Vec3f>& lights,
    const PhotometricDiagnostics& diagnostics);
