#include "args.hpp"
#include "gui_workflow.hpp"
#include "image_io.hpp"
#include "photometric.hpp"
#include "relight_ui.hpp"
#include "sphere_ui.hpp"

#include <opencv2/core.hpp>

#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

void logStage(const std::string& message) {
    std::cout << message << std::endl;
}

void runPhotometricStereo(Options& opt) {
    if (!opt.uncalibratedLighting && opt.lightsFile.empty() && !opt.hasSphere) {
        std::cout << "Select the highlight sphere on the first image.\n";
        opt.sphere = chooseSphereInteractive(loadDisplayImage(opt.imagePaths.front()));
        opt.hasSphere = true;
        std::cout << "Sphere: cx=" << opt.sphere.cx
                  << " cy=" << opt.sphere.cy
                  << " radius=" << opt.sphere.radius << '\n';
    }

    logStage("[1/6] Loading images...");
    const std::vector<cv::Mat> images = loadLuminanceImages(opt.imagePaths, opt.srgb);
    logStage("[2/6] Loading mask...");
    cv::Mat mask = loadMask(opt.maskPath, images[0].size());
    if (opt.hasCrop) {
        std::cout << "      applying crop x=" << opt.crop.x
                  << " y=" << opt.crop.y
                  << " width=" << opt.crop.width
                  << " height=" << opt.crop.height << std::endl;
        applyCropToMask(mask, opt.crop);
    }

    logStage(opt.uncalibratedLighting ? "[3/6] Preparing unknown-lighting solve..." : "[3/6] Estimating light directions...");
    std::vector<cv::Vec3f> lights;
    std::vector<HighlightEstimate> estimates;
    if (opt.uncalibratedLighting) {
        // No calibrated light vectors are available in this mode.
    } else if (!opt.lightsFile.empty()) {
        lights = loadLightsFile(opt.lightsFile, opt.imagePaths.size());
    } else {
        estimates.reserve(images.size());
        lights.reserve(images.size());
        for (const cv::Mat& image : images) {
            HighlightEstimate estimate = estimateHighlight(image, opt.sphere, opt);
            lights.push_back(estimate.light);
            estimates.push_back(estimate);
        }
    }

    if (!opt.uncalibratedLighting && opt.hasSphere && !opt.keepSphere) {
        logStage("[4/6] Removing selected sphere from solve mask...");
        removeSphereFromMask(mask, opt.sphere);
    } else {
        logStage("[4/6] Preparing solve mask...");
    }

    logStage(opt.uncalibratedLighting ? "[5/6] Solving relative normals from unknown lighting..." : "[5/6] Solving normal map and albedo...");
    cv::Mat normalMap;
    cv::Mat albedo;
    cv::Mat residual;
    cv::Mat validMask;
    if (opt.uncalibratedLighting) {
        solveUncalibratedPhotometricStereo(
            images,
            mask,
            static_cast<float>(opt.shadowThreshold),
            normalMap,
            albedo,
            residual,
            validMask,
            [](const std::string& message) {
                std::cout << "      " << message << std::endl;
            });
    } else {
        solvePhotometricStereo(
            images,
            lights,
            mask,
            static_cast<float>(opt.shadowThreshold),
            normalMap,
            albedo,
            residual,
            validMask);
    }

    cv::Mat height;
    if (opt.calculateHeight) {
        logStage("[6/6] Solving DCT/Poisson height preview...");
        height = integrateHeight(
            normalMap,
            validMask,
            opt.integrationIterations,
            [](int done, int total) {
                std::cout << "      height solve " << done << "/" << total << std::endl;
            });
        if (opt.uncalibratedLighting) {
            std::cout << "      removing uncalibrated height tilt" << std::endl;
            removeBestFitPlane(height, validMask);
        }
        if (!opt.meshPath.empty()) {
            std::cout << "      PLY mesh will be written to: " << opt.meshPath << std::endl;
        }
    } else {
        logStage("[6/6] Skipping height preview.");
    }
    logStage("Writing outputs...");
    saveOutputs(
        opt,
        lights,
        estimates,
        normalMap,
        albedo,
        residual,
        validMask,
        height,
        [](const std::string& message) {
            std::cout << "      " << message << std::endl;
        });

    std::cout << "Wrote photometric stereo outputs to: " << opt.outputDir << '\n';
    if (opt.guiMode && askGuiYesNo(
                           "Open Interactive Relight?",
                           "Open the specular relight viewer?\n\n"
                           "Drag in the viewer to move the virtual light. Press S to save the current view.",
                           true)) {
        launchRelightViewer(normalMap, validMask, opt.outputDir);
    }
}

} // namespace

int main(int argc, char** argv) {
    bool guiMode = false;
    try {
        Options opt = parseArgs(argc, argv);
        guiMode = opt.guiMode;
        if (opt.guiMode) {
            launchGuiWorkflow(opt);
        }
        runPhotometricStereo(opt);
        if (opt.guiMode) {
            showGuiInfo("What A Relief Complete", "Outputs were written to:\n\n" + opt.outputDir);
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        if (guiMode) {
            showGuiInfo("What A Relief Error", e.what());
        }
        return 1;
    }
}
