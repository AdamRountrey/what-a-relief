#include "args.hpp"
#include "gui_workflow.hpp"
#include "image_io.hpp"
#include "neural_fusion.hpp"
#include "photometric.hpp"
#include "relight_ui.hpp"
#include "run_manifest.hpp"
#include "rti_export.hpp"
#include "sphere_ui.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace {

void logStage(const std::string& message) {
    std::cout << message << std::endl;
}

using ProgressCallback = std::function<void(const std::string&, int)>;

void reportStage(const ProgressCallback& progress, const std::string& message, int percent) {
    logStage(message);
    if (progress) {
        progress(message, percent);
    }
}

void reportDetail(const ProgressCallback& progress, const std::string& message, int percent) {
    std::cout << "      " << message << std::endl;
    if (progress) {
        progress(message, percent);
    }
}

cv::Mat buildHeightMask(const Options& opt, const cv::Mat& baseGeometryMask, const cv::Size& imageSize) {
    cv::Mat heightMask = baseGeometryMask.clone();
    bool hasCustomMask = false;

    if (!opt.heightMaskPath.empty()) {
        cv::Mat fileMask = loadMask(opt.heightMaskPath, imageSize);
        cv::bitwise_and(heightMask, fileMask, heightMask);
        hasCustomMask = true;
    }
    if (opt.hasHeightMask) {
        if (opt.heightMask.size() != imageSize) {
            throw std::runtime_error("Interactive specimen mask dimensions do not match input images.");
        }
        cv::Mat drawnMask;
        cv::threshold(opt.heightMask, drawnMask, 0, 255, cv::THRESH_BINARY);
        cv::bitwise_and(heightMask, drawnMask, heightMask);
        hasCustomMask = true;
    }

    if (hasCustomMask) {
        const int rawCount = cv::countNonZero(heightMask);
        if (rawCount < 100) {
            throw std::runtime_error("Specimen height mask is empty after intersecting with the solved-pixel mask.");
        }
        cv::Mat erodedMask;
        cv::erode(heightMask, erodedMask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
        if (cv::countNonZero(erodedMask) >= 100) {
            heightMask = erodedMask;
        }
            std::cout << "      using specimen mask for height preview and PLY only" << std::endl;
    }
    return heightMask;
}

const char* heightSolverName(HeightSolverMode mode) {
    switch (mode) {
    case HeightSolverMode::FastDct:
        return "fast DCT/Poisson";
    case HeightSolverMode::RobustMasked:
    default:
        return "robust masked weighted Poisson";
    }
}

const char* heightFlattenName(HeightFlattenMode mode) {
    switch (mode) {
    case HeightFlattenMode::Plane:
        return "least-squares plane height leveling";
    case HeightFlattenMode::Radial:
        return "radial/dome height curl correction";
    case HeightFlattenMode::Quadratic:
        return "quadratic height curl correction";
    case HeightFlattenMode::None:
    default:
        return "none";
    }
}

void runPhotometricStereo(Options& opt, const ProgressCallback& progress = {}) {
    if (!opt.uncalibratedLighting && opt.lightsFile.empty() && !opt.hasSphere) {
        std::cout << "Select the highlight sphere on the first image.\n";
        opt.sphere = chooseSphereInteractive(loadDisplayImage(opt.imagePaths.front()));
        opt.hasSphere = true;
        std::cout << "Sphere: cx=" << opt.sphere.cx
                  << " cy=" << opt.sphere.cy
                  << " radius=" << opt.sphere.radius << '\n';
    }

    const RunManifestContext runManifest = beginRunManifest(opt);

    reportStage(progress, "[1/6] Loading images...", 5);
    const std::vector<cv::Mat> images = loadLuminanceImages(opt.imagePaths, opt.srgb);
    reportStage(progress, "[2/6] Loading mask...", 15);
    cv::Mat mask = loadMask(opt.maskPath, images[0].size());
    if (opt.hasCrop) {
        std::cout << "      applying crop x=" << opt.crop.x
                  << " y=" << opt.crop.y
                  << " width=" << opt.crop.width
                  << " height=" << opt.crop.height << std::endl;
        applyCropToMask(mask, opt.crop);
    }

    reportStage(progress, opt.uncalibratedLighting ? "[3/6] Preparing unknown-lighting solve..." : "[3/6] Estimating light directions...", 25);
    std::vector<cv::Vec3f> lights;
    std::vector<HighlightEstimate> estimates;
    if (opt.uncalibratedLighting) {
        // No calibrated light vectors are available in this mode.
    } else if (!opt.lightsFile.empty()) {
        lights = loadLightsFile(opt.lightsFile, opt.imagePaths, &opt, opt.lightsFileByOrder);
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

    if (((!opt.uncalibratedLighting && opt.lightingModel == LightingModel::NearFieldRing) || !opt.printableMeshPath.empty()) &&
        opt.pixelScaleMm <= 0.0) {
        opt.pixelScaleMm = readPixelScaleMmFromImage(opt.imagePaths.front());
        if (opt.pixelScaleMm > 0.0) {
            std::cout << "      pixel scale from TIFF metadata: " << opt.pixelScaleMm << " mm/pixel" << std::endl;
        }
    }
    if (!opt.uncalibratedLighting && opt.lightingModel == LightingModel::NearFieldRing) {
        if (opt.pixelScaleMm <= 0.0) {
            throw std::runtime_error(
                "Near-field ring lighting needs a pixel scale. Enter --pixel-scale-mm in mm/pixel, "
                "or use a TIFF with readable physical scale tags.");
        }
    }
    if (!opt.printableMeshPath.empty() && opt.pixelScaleMm <= 0.0) {
        throw std::runtime_error(
            "Printable mesh export needs a pixel scale. Enter --pixel-scale-mm in mm/pixel, "
            "draw a GUI scale line, or use a TIFF with readable physical scale tags.");
    }

    if (!opt.uncalibratedLighting && opt.hasSphere && !opt.keepSphere) {
        reportStage(progress, "[4/6] Removing selected sphere from solve mask...", 35);
        removeSphereFromMask(mask, opt.sphere);
    } else {
        reportStage(progress, "[4/6] Preparing solve mask...", 35);
    }

    reportStage(progress, opt.uncalibratedLighting ? "[5/6] Solving relative normals from unknown lighting..." : "[5/6] Solving normal map and albedo...", 45);
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
            [&](const std::string& message) {
                reportDetail(progress, message, 52);
            });
        const int maskPixels = cv::countNonZero(mask);
        diagnostics.solvedFraction = maskPixels > 0
            ? static_cast<double>(cv::countNonZero(validMask)) / static_cast<double>(maskPixels)
            : 0.0;
        std::cout << "      solved mask coverage: "
                  << (100.0 * diagnostics.solvedFraction) << "%" << std::endl;
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
            opt.viewDir,
            normalMap,
            albedo,
            residual,
            validMask,
            diagnostics);
        std::cout << "      light geometry condition number: "
                  << diagnostics.lightingConditionNumber << std::endl;
        std::cout << "      solved mask coverage: "
                  << (100.0 * diagnostics.solvedFraction) << "%" << std::endl;
        if (diagnostics.solvedFraction < 0.10) {
            std::cout << "      warning: fewer than 10% of masked pixels produced valid normals; "
                         "inspect the input mask, shadows, exposure, and light calibration"
                      << std::endl;
        }
        if (opt.solverMode == NormalSolverMode::Robust && !diagnostics.robustFallbackMask.empty()) {
            const int fallbackPixels = cv::countNonZero(diagnostics.robustFallbackMask);
            if (fallbackPixels > 0) {
                std::cout << "      robust fallback: " << fallbackPixels
                          << " pixels had only three usable observations and used least squares" << std::endl;
            }
        }
        geometryNormalMap = normalMap.clone();
        geometryValidMask = validMask.clone();

        if (opt.neuralFusion) {
            reportStage(progress, "[5.5/6] Running experimental PS-FCN neural fusion...", 62);
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
        reportDetail(progress, "applying low-frequency relief flattening", 68);
        flattenNormalField(normalMap, validMask, opt.flattenMode);
        if (!geometryNormalMap.empty()) {
            flattenNormalField(geometryNormalMap, geometryValidMask, opt.flattenMode);
        }
    }

    cv::Mat height;
    cv::Mat heightMask;
    if (opt.calculateHeight) {
        try {
            const cv::Mat& baseGeometryMask = geometryValidMask.empty() ? validMask : geometryValidMask;
            heightMask = buildHeightMask(opt, baseGeometryMask, images[0].size());
            reportStage(progress, std::string("[6/6] Solving ") + heightSolverName(opt.heightSolverMode) + " height preview...", 72);
            height = integrateHeight(
                geometryNormalMap.empty() ? normalMap : geometryNormalMap,
                heightMask,
                opt.heightSolverMode,
                opt.heightSlopeCap,
                opt.integrationIterations,
                [&](int done, int total) {
                    std::cout << "      height solve " << done << "/" << total << std::endl;
                    if (progress && total > 0) {
                        progress("height solve " + std::to_string(done) + "/" + std::to_string(total), 72 + (12 * done) / total);
                    }
                });
            if (opt.heightFlattenMode != HeightFlattenMode::None) {
                std::cout << "      applying " << heightFlattenName(opt.heightFlattenMode) << std::endl;
                applyHeightFlattening(height, heightMask, opt.heightFlattenMode);
            }
            if (!opt.meshPath.empty()) {
                std::cout << "      PLY mesh will be written to: " << opt.meshPath << std::endl;
            }
            if (!opt.printableMeshPath.empty()) {
                std::cout << "      printable PLY solid will be written to: " << opt.printableMeshPath << std::endl;
            }
        } catch (const std::exception& e) {
            height.release();
            heightMask.release();
            throw std::runtime_error(std::string("Height/mesh export failed: ") + e.what());
        }
    } else {
        reportStage(progress, "[6/6] Skipping height preview.", 78);
    }
    reportStage(progress, "Writing outputs...", 84);
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
        heightMask,
        [&](const std::string& message) {
            reportDetail(progress, message, 88);
        });

    if (opt.exportRti) {
        reportStage(progress, "Writing RTI package...", 92);
        exportRtiPackage(
            opt,
            lights,
            images[0].size(),
            [&](const std::string& message) {
                reportDetail(progress, message, 95);
            });
    }

    completeRunManifest(opt, runManifest, lights, diagnostics);

    if (progress) {
        progress("Complete.", 100);
    }
    std::cout << "Wrote photometric stereo outputs to: " << opt.outputDir << '\n';
    if (opt.guiMode && opt.openRelightViewer) {
        launchRelightViewer(normalMap, validMask, opt.outputDir);
    }
}

} // namespace

int main(int argc, char** argv) {
    bool guiMode = false;
    bool progressShown = false;
    try {
        Options opt = parseArgs(argc, argv);
        guiMode = opt.guiMode;
        if (opt.guiMode) {
            if (!launchGuiWorkflow(opt)) {
                return 0;
            }
            showGuiProgress("what-a-relief Processing", "Starting photometric stereo...");
            progressShown = true;
        }
        runPhotometricStereo(
            opt,
            opt.guiMode
                ? ProgressCallback([](const std::string& message, int percent) {
                      updateGuiProgress(message, percent);
                  })
                : ProgressCallback());
        if (progressShown) {
            closeGuiProgress();
            progressShown = false;
        }
        if (opt.guiMode) {
            showGuiInfo("what-a-relief Complete", "Outputs were written to:\n\n" + opt.outputDir);
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        if (progressShown) {
            closeGuiProgress();
        }
        if (guiMode) {
            showGuiInfo("what-a-relief Error", e.what());
        }
        return 1;
    }
}
