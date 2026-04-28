#include "args.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

constexpr size_t kMinImages = 3;
constexpr size_t kMaxImages = 25;

[[noreturn]] void die(const std::string& message) {
    throw std::runtime_error(message);
}

bool validImageCount(size_t count) {
    return count >= kMinImages && count <= kMaxImages;
}

double parseDouble(const std::string& s, const std::string& label) {
    try {
        size_t end = 0;
        const double value = std::stod(s, &end);
        if (end != s.size()) {
            die("Invalid " + label + ": " + s);
        }
        return value;
    } catch (const std::exception&) {
        die("Invalid " + label + ": " + s);
    }
}

int parseInt(const std::string& s, const std::string& label) {
    try {
        size_t end = 0;
        const int value = std::stoi(s, &end);
        if (end != s.size()) {
            die("Invalid " + label + ": " + s);
        }
        return value;
    } catch (const std::exception&) {
        die("Invalid " + label + ": " + s);
    }
}

NormalSolverMode parseSolverMode(const std::string& s) {
    if (s == "standard" || s == "least-squares" || s == "ls") {
        return NormalSolverMode::Standard;
    }
    if (s == "robust" || s == "irls") {
        return NormalSolverMode::Robust;
    }
    die("Invalid solver mode: " + s + " (use standard or robust)");
}

FlattenMode parseFlattenMode(const std::string& s) {
    if (s == "none" || s == "off") {
        return FlattenMode::None;
    }
    if (s == "gentle" || s == "light") {
        return FlattenMode::Gentle;
    }
    if (s == "strong") {
        return FlattenMode::Strong;
    }
    die("Invalid flatten mode: " + s + " (use none, gentle, or strong)");
}

} // namespace

void printUsage() {
    std::cout
        << "What A Relief: relief visualization and photometric stereo from 3 to 25 images.\n\n"
        << "GUI:\n"
        << "  what-a-relief.exe            Open file picker, output folder picker, and lighting choice.\n"
        << "  --gui                        Same as running with no arguments.\n\n"
        << "Required:\n"
        << "  --image path                 Add one image. Use 3 to 25 times.\n\n"
        << "Interactive sphere selection:\n"
        << "  If --sphere, --lights-file, and --uncalibrated are omitted, the first image opens in a window.\n"
        << "  Click three points on the sphere edge, then press Enter or Space.\n"
        << "  Mouse wheel/+/- zoom; right-drag, WASD, or arrow keys pan.\n\n"
        << "Options:\n"
        << "  --out dir                    Output directory. Default: out\n"
        << "  --mask path                  Optional object mask; white pixels are solved.\n"
        << "  --uncalibrated               Skip sphere calibration and estimate a relative normal field.\n"
        << "  --crop x y width height      Restrict solve to a rectangular image region.\n"
        << "  --sphere cx cy radius        Reuse a known sphere circle without the GUI.\n"
        << "  --lights-file path           CSV/text file with one x,y,z light vector per image.\n"
        << "  --no-gui                     Disable interactive selection. Requires --sphere, --lights-file, or --uncalibrated.\n"
        << "  --srgb                       Linearize sRGB image intensities.\n"
        << "  --solver standard|robust     Calibrated normal solve. Default: robust\n"
        << "  --high-outlier-threshold v   Robust solve highlight/saturation cutoff. Default: 0.98\n"
        << "  --near-field-ring r h        Use point lights on a ring with radius r and height h, in mm.\n"
        << "  --pixel-scale-mm s           Pixel size in mm/pixel for near-field solving; 0 auto-reads TIFF tags.\n"
        << "  --specular-diagnostics       Write experimental shiny-cue and robust outlier diagnostic maps.\n"
        << "  --neural-fusion             Run bundled PS-FCN neural normal prior and fuse it with the classical solve.\n"
        << "                               Experimental; calibrated mode only, supports 3 to 25 images.\n"
        << "                               Height/PLY remain classical-only to avoid exaggerated geometry.\n"
        << "  --neural-model path         Override bundled PS-FCN ONNX model path, or point at a model directory.\n"
        << "  --flatten none|gentle|strong Remove low-frequency slope trend before output. Default: none\n"
        << "  --open-relight               Open the interactive relight viewer after GUI processing.\n"
        << "  --highlight-percentile p     Bright percentile inside sphere. Default: 99.8\n"
        << "  --min-highlight v            Minimum highlight intensity. Default: 0.05\n"
        << "  --shadow-threshold v         Per-observation shadow cutoff. Default: 0.02\n"
        << "  --integration-iterations n   Legacy option accepted; DCT/Poisson height ignores it.\n"
        << "  --no-height                  Skip height.png and height.pfm.\n"
        << "  --mesh path.ply              Export a PLY mesh from the height preview.\n"
        << "  --mesh-step n                Use every nth pixel for PLY export. Default: 1\n"
        << "  --height-scale s             Scale mesh z coordinates. Default: 1.0\n"
        << "  --keep-sphere                Do not remove the calibration sphere from the solve mask.\n"
        << "  --view-dir x y z             Camera view vector. Default: 0 0 1\n"
        << "  --help                       Show this help.\n";
}

Options parseArgs(int argc, char** argv) {
    Options opt;
    if (argc == 1) {
        opt.guiMode = true;
        return opt;
    }

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need = [&](int n) {
            if (i + n >= argc) {
                die("Missing value after " + arg);
            }
        };

        if (arg == "--help" || arg == "-h") {
            printUsage();
            std::exit(0);
        } else if (arg == "--gui") {
            opt.guiMode = true;
        } else if (arg == "--image") {
            need(1);
            opt.imagePaths.emplace_back(argv[++i]);
        } else if (arg == "--out") {
            need(1);
            opt.outputDir = argv[++i];
        } else if (arg == "--mask") {
            need(1);
            opt.maskPath = argv[++i];
        } else if (arg == "--uncalibrated" || arg == "--no-sphere") {
            opt.uncalibratedLighting = true;
        } else if (arg == "--crop") {
            need(4);
            opt.crop.x = parseInt(argv[++i], "crop x");
            opt.crop.y = parseInt(argv[++i], "crop y");
            opt.crop.width = parseInt(argv[++i], "crop width");
            opt.crop.height = parseInt(argv[++i], "crop height");
            opt.hasCrop = true;
        } else if (arg == "--lights-file") {
            need(1);
            opt.lightsFile = argv[++i];
        } else if (arg == "--sphere") {
            need(3);
            opt.sphere.cx = parseDouble(argv[++i], "sphere cx");
            opt.sphere.cy = parseDouble(argv[++i], "sphere cy");
            opt.sphere.radius = parseDouble(argv[++i], "sphere radius");
            opt.hasSphere = true;
        } else if (arg == "--no-gui") {
            opt.noGui = true;
        } else if (arg == "--srgb") {
            opt.srgb = true;
        } else if (arg == "--solver") {
            need(1);
            opt.solverMode = parseSolverMode(argv[++i]);
        } else if (arg == "--high-outlier-threshold") {
            need(1);
            opt.highOutlierThreshold = parseDouble(argv[++i], "high outlier threshold");
        } else if (arg == "--near-field-ring") {
            need(2);
            opt.lightingModel = LightingModel::NearFieldRing;
            opt.ringLightRadiusMm = parseDouble(argv[++i], "ring light radius");
            opt.ringLightHeightMm = parseDouble(argv[++i], "ring light height");
        } else if (arg == "--pixel-scale-mm") {
            need(1);
            opt.pixelScaleMm = parseDouble(argv[++i], "pixel scale");
        } else if (arg == "--specular-diagnostics") {
            opt.specularDiagnostics = true;
        } else if (arg == "--neural-fusion") {
            opt.neuralFusion = true;
        } else if (arg == "--neural-model") {
            need(1);
            opt.neuralModelPath = argv[++i];
        } else if (arg == "--flatten") {
            need(1);
            opt.flattenMode = parseFlattenMode(argv[++i]);
        } else if (arg == "--open-relight") {
            opt.openRelightViewer = true;
        } else if (arg == "--keep-sphere") {
            opt.keepSphere = true;
        } else if (arg == "--highlight-percentile") {
            need(1);
            opt.highlightPercentile = parseDouble(argv[++i], "highlight percentile");
        } else if (arg == "--min-highlight") {
            need(1);
            opt.minHighlight = parseDouble(argv[++i], "minimum highlight");
        } else if (arg == "--shadow-threshold") {
            need(1);
            opt.shadowThreshold = parseDouble(argv[++i], "shadow threshold");
        } else if (arg == "--integration-iterations") {
            need(1);
            opt.integrationIterations = parseInt(argv[++i], "integration iterations");
        } else if (arg == "--no-height") {
            opt.calculateHeight = false;
        } else if (arg == "--mesh") {
            need(1);
            opt.meshPath = argv[++i];
            opt.calculateHeight = true;
        } else if (arg == "--mesh-step") {
            need(1);
            opt.meshStep = parseInt(argv[++i], "mesh step");
        } else if (arg == "--height-scale") {
            need(1);
            opt.heightScale = parseDouble(argv[++i], "height scale");
        } else if (arg == "--view-dir") {
            need(3);
            opt.viewDir = cv::Vec3f(
                static_cast<float>(parseDouble(argv[++i], "view x")),
                static_cast<float>(parseDouble(argv[++i], "view y")),
                static_cast<float>(parseDouble(argv[++i], "view z")));
        } else {
            die("Unknown argument: " + arg);
        }
    }

    if (!opt.guiMode && !validImageCount(opt.imagePaths.size())) {
        die("Use 3 to 25 --image arguments.");
    }
    if (opt.guiMode && !opt.imagePaths.empty() && !validImageCount(opt.imagePaths.size())) {
        die("Use 3 to 25 --image arguments.");
    }
    if (opt.noGui && opt.lightsFile.empty() && !opt.hasSphere && !opt.uncalibratedLighting) {
        die("--no-gui requires --sphere, --lights-file, or --uncalibrated.");
    }
    if (opt.uncalibratedLighting && (!opt.lightsFile.empty() || opt.hasSphere)) {
        die("--uncalibrated cannot be combined with --sphere or --lights-file.");
    }
    if (opt.uncalibratedLighting && (!opt.guiMode || !opt.imagePaths.empty()) && opt.imagePaths.size() < 4) {
        die("--uncalibrated requires at least 4 images.");
    }
    if (opt.neuralFusion) {
        if (opt.uncalibratedLighting) {
            die("--neural-fusion cannot be combined with --uncalibrated.");
        }
        if ((!opt.guiMode || !opt.imagePaths.empty()) &&
            (opt.imagePaths.size() < kMinImages || opt.imagePaths.size() > kMaxImages)) {
            die("--neural-fusion supports 3 to 25 images.");
        }
    }
    if (opt.hasCrop && (opt.crop.width <= 0 || opt.crop.height <= 0)) {
        die("--crop width and height must be positive.");
    }
    if (opt.hasSphere && opt.sphere.radius <= 0.0) {
        die("Sphere radius must be positive.");
    }
    if (opt.highlightPercentile <= 0.0 || opt.highlightPercentile >= 100.0) {
        die("--highlight-percentile must be between 0 and 100.");
    }
    if (!std::isfinite(opt.highOutlierThreshold) ||
        opt.highOutlierThreshold <= opt.shadowThreshold ||
        opt.highOutlierThreshold > 1.0) {
        die("--high-outlier-threshold must be above --shadow-threshold and no more than 1.0.");
    }
    if (opt.lightingModel == LightingModel::NearFieldRing) {
        if (opt.uncalibratedLighting) {
            die("--near-field-ring cannot be combined with --uncalibrated.");
        }
        if (!std::isfinite(opt.ringLightRadiusMm) || opt.ringLightRadiusMm <= 0.0 ||
            !std::isfinite(opt.ringLightHeightMm) || opt.ringLightHeightMm <= 0.0) {
            die("--near-field-ring radius and height must be positive finite values.");
        }
        if (!std::isfinite(opt.pixelScaleMm) || opt.pixelScaleMm < 0.0) {
            die("--pixel-scale-mm must be positive, or 0 to auto-read TIFF tags.");
        }
    }
    if (opt.integrationIterations < 0) {
        die("--integration-iterations must be non-negative.");
    }
    if (opt.meshStep < 1) {
        die("--mesh-step must be at least 1.");
    }
    if (!std::isfinite(opt.heightScale) || opt.heightScale == 0.0) {
        die("--height-scale must be finite and non-zero.");
    }
    if (!opt.meshPath.empty()) {
        opt.calculateHeight = true;
    }

    const float viewNorm = std::sqrt(opt.viewDir.dot(opt.viewDir));
    if (viewNorm <= 0.0f) {
        die("--view-dir must be non-zero.");
    }
    opt.viewDir /= viewNorm;

    return opt;
}
