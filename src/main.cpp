#include "args.hpp"
#include "gui_workflow.hpp"
#include "image_io.hpp"
#include "neural_fusion.hpp"
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
        lights = loadLightsFile(opt.lightsFile, opt.imagePaths.size(), &opt);
        if (opt.lightingModel == LightingModel::NearFieldRing) {
            std::cout << "      loaded near-field ring calibration metadata from: " << opt.lightsFile << std::endl;
        }
    } else {
        estimates.reserve(images.size());
        lights.reserve(images.size());
        for (const cv::Mat& image : images) {
            HighlightEstimate estimate = estimateHighlight(image, opt.sphere, opt);
            lights.push_back(estimate.light);
            estimates.push_back(estimate);
        }
    }

    if (!opt.uncalibratedLighting && opt.lightingModel == LightingModel::NearFieldRing) {
        if (opt.pixelScaleMm <= 0.0) {
            opt.pixelScaleMm = readPixelScaleMmFromImage(opt.imagePaths.front());
            if (opt.pixelScaleMm > 0.0) {
                std::cout << "      pixel scale from TIFF metadata: " << opt.pixelScaleMm << " mm/pixel" << std::endl;
            }
        }
        if (opt.pixelScaleMm <= 0.0) {
            throw std::runtime_error(
                "Near-field ring lighting needs a pixel scale. Enter --pixel-scale-mm in mm/pixel, "
                "or use a TIFF with readable physical scale tags.");
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
    cv::Mat geometryNormalMap;
    cv::Mat geometryValidMask;
    PhotometricDiagnostics diagnostics;
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
        cv::Point2d lightingCenter(
            static_cast<double>(images[0].cols - 1) * 0.5,
            static_cast<double>(images[0].rows - 1) * 0.5);
        if (opt.hasCrop) {
            lightingCenter = cv::Point2d(
                static_cast<double>(opt.crop.x) + static_cast<double>(opt.crop.width - 1) * 0.5,
                static_cast<double>(opt.crop.y) + static_cast<double>(opt.crop.height - 1) * 0.5);
        }
        if (opt.lightingModel == LightingModel::NearFieldRing) {
            std::cout << "      using near-field ring light model radius=" << opt.ringLightRadiusMm
                      << " mm height=" << opt.ringLightHeightMm
                      << " mm pixel_scale=" << opt.pixelScaleMm << " mm/pixel" << std::endl;
        }
        solvePhotometricStereo(
            images,
            lights,
            mask,
            static_cast<float>(opt.shadowThreshold),
            opt.solverMode,
            static_cast<float>(opt.highOutlierThreshold),
            opt.lightingModel,
            opt.ringLightRadiusMm,
            opt.ringLightHeightMm,
            opt.pixelScaleMm,
            lightingCenter,
            normalMap,
            albedo,
            residual,
            validMask,
            diagnostics);
        geometryNormalMap = normalMap.clone();
        geometryValidMask = validMask.clone();

        if (opt.neuralFusion) {
            logStage("[5.5/6] Running experimental PS-FCN neural fusion...");
            std::cout << "      note: neural fusion changes the normal-map outputs only; height preview and PLY stay classical to avoid exaggerated geometry." << std::endl;
            applyNeuralFusion(
                opt,
                images,
                lights,
                mask,
                lightingCenter,
                normalMap,
                albedo,
                residual,
                validMask,
                diagnostics);
        }
    }

    if (opt.flattenMode != FlattenMode::None) {
        logStage("      applying low-frequency relief flattening");
        flattenNormalField(normalMap, validMask, opt.flattenMode);
        if (!geometryNormalMap.empty()) {
            flattenNormalField(geometryNormalMap, geometryValidMask, opt.flattenMode);
        }
    }

    cv::Mat height;
    if (opt.calculateHeight) {
        logStage("[6/6] Solving DCT/Poisson height preview...");
        height = integrateHeight(
            geometryNormalMap.empty() ? normalMap : geometryNormalMap,
            geometryValidMask.empty() ? validMask : geometryValidMask,
            opt.integrationIterations,
            [](int done, int total) {
                std::cout << "      height solve " << done << "/" << total << std::endl;
            });
        std::cout << "      removing best-fit plane from height preview" << std::endl;
        removeBestFitPlane(height, geometryValidMask.empty() ? validMask : geometryValidMask);
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
        diagnostics,
        height,
        [](const std::string& message) {
            std::cout << "      " << message << std::endl;
        });

    std::cout << "Wrote photometric stereo outputs to: " << opt.outputDir << '\n';
    if (opt.guiMode && opt.openRelightViewer) {
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
