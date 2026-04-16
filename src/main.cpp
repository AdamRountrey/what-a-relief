#include "args.hpp"
#include "gui_workflow.hpp"
#include "image_io.hpp"
#include "photometric.hpp"
#include "sphere_ui.hpp"

#include <opencv2/core.hpp>

#include <exception>
#include <iostream>
#include <vector>

namespace {

void runPhotometricStereo(Options& opt) {
    if (opt.lightsFile.empty() && !opt.hasSphere) {
        std::cout << "Select the highlight sphere on the first image.\n";
        opt.sphere = chooseSphereInteractive(loadDisplayImage(opt.imagePaths.front()));
        opt.hasSphere = true;
        std::cout << "Sphere: cx=" << opt.sphere.cx
                  << " cy=" << opt.sphere.cy
                  << " radius=" << opt.sphere.radius << '\n';
    }

    const std::vector<cv::Mat> images = loadLuminanceImages(opt.imagePaths, opt.srgb);
    cv::Mat mask = loadMask(opt.maskPath, images[0].size());

    std::vector<cv::Vec3f> lights;
    std::vector<HighlightEstimate> estimates;
    if (!opt.lightsFile.empty()) {
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

    if (opt.hasSphere && !opt.keepSphere) {
        removeSphereFromMask(mask, opt.sphere);
    }

    cv::Mat normalMap;
    cv::Mat albedo;
    cv::Mat residual;
    cv::Mat validMask;
    solvePhotometricStereo(
        images,
        lights,
        mask,
        static_cast<float>(opt.shadowThreshold),
        normalMap,
        albedo,
        residual,
        validMask);

    cv::Mat height = integrateHeight(normalMap, validMask, opt.integrationIterations);
    saveOutputs(opt, lights, estimates, normalMap, albedo, residual, validMask, height);

    std::cout << "Wrote photometric stereo outputs to: " << opt.outputDir << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        Options opt = parseArgs(argc, argv);
        if (opt.guiMode) {
            launchGuiWorkflow(opt);
        }
        runPhotometricStereo(opt);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
