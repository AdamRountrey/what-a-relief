#include "args.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

[[noreturn]] void die(const std::string& message) {
    throw std::runtime_error(message);
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

} // namespace

void printUsage() {
    std::cout
        << "Photometric stereo from 4 or 8 images with highlight sphere light calibration.\n\n"
        << "GUI:\n"
        << "  ps_spheres.exe               Open file picker, output folder picker, and sphere marker.\n"
        << "  --gui                        Same as running with no arguments.\n\n"
        << "Required:\n"
        << "  --image path                 Add one image. Use exactly 4 or 8 times.\n\n"
        << "Interactive sphere selection:\n"
        << "  If --sphere and --lights-file are omitted, the first image opens in a window.\n"
        << "  Click-drag from sphere center to sphere edge, then press Enter or Space.\n\n"
        << "Options:\n"
        << "  --out dir                    Output directory. Default: out\n"
        << "  --mask path                  Optional object mask; white pixels are solved.\n"
        << "  --sphere cx cy radius        Reuse a known sphere circle without the GUI.\n"
        << "  --lights-file path           CSV/text file with one x,y,z light vector per image.\n"
        << "  --no-gui                     Disable interactive selection. Requires --sphere or --lights-file.\n"
        << "  --srgb                       Linearize sRGB image intensities.\n"
        << "  --highlight-percentile p     Bright percentile inside sphere. Default: 99.8\n"
        << "  --min-highlight v            Minimum highlight intensity. Default: 0.05\n"
        << "  --shadow-threshold v         Per-observation shadow cutoff. Default: 0.02\n"
        << "  --integration-iterations n   Height preview iterations. Default: 800\n"
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

    if (!opt.guiMode && opt.imagePaths.size() != 4 && opt.imagePaths.size() != 8) {
        die("Use exactly 4 or 8 --image arguments.");
    }
    if (opt.guiMode && !opt.imagePaths.empty() && opt.imagePaths.size() != 4 && opt.imagePaths.size() != 8) {
        die("Use exactly 4 or 8 --image arguments.");
    }
    if (opt.noGui && opt.lightsFile.empty() && !opt.hasSphere) {
        die("--no-gui requires --sphere or --lights-file.");
    }
    if (opt.hasSphere && opt.sphere.radius <= 0.0) {
        die("Sphere radius must be positive.");
    }
    if (opt.highlightPercentile <= 0.0 || opt.highlightPercentile >= 100.0) {
        die("--highlight-percentile must be between 0 and 100.");
    }
    if (opt.integrationIterations < 0) {
        die("--integration-iterations must be non-negative.");
    }

    const float viewNorm = std::sqrt(opt.viewDir.dot(opt.viewDir));
    if (viewNorm <= 0.0f) {
        die("--view-dir must be non-zero.");
    }
    opt.viewDir /= viewNorm;

    return opt;
}
