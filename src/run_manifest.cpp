#include "run_manifest.hpp"

#include "checked_io.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#ifndef WHAT_A_RELIEF_VERSION
#define WHAT_A_RELIEF_VERSION "0.2.1"
#endif

namespace {

std::string utcNow() {
    const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &now);
#else
    gmtime_r(&now, &utc);
#endif
    std::ostringstream out;
    out << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

std::string jsonEscape(const std::string& value) {
    std::ostringstream out;
    for (const unsigned char c : value) {
        switch (c) {
        case '"':
            out << "\\\"";
            break;
        case '\\':
            out << "\\\\";
            break;
        case '\b':
            out << "\\b";
            break;
        case '\f':
            out << "\\f";
            break;
        case '\n':
            out << "\\n";
            break;
        case '\r':
            out << "\\r";
            break;
        case '\t':
            out << "\\t";
            break;
        default:
            if (c < 0x20) {
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c)
                    << std::dec << std::setfill(' ');
            } else {
                out << static_cast<char>(c);
            }
        }
    }
    return out.str();
}

std::string normalizedAbsolutePath(const fs::path& path) {
    std::error_code error;
    const fs::path absolute = fs::absolute(path, error);
    return (error ? path : absolute).lexically_normal().string();
}

const char* solverName(NormalSolverMode mode) {
    return mode == NormalSolverMode::Robust ? "robust" : "standard";
}

const char* lightingName(const Options& opt) {
    if (opt.uncalibratedLighting) {
        return "uncalibrated";
    }
    return opt.lightingModel == LightingModel::NearFieldRing ? "near_field_ring" : "directional";
}

const char* flattenName(FlattenMode mode) {
    switch (mode) {
    case FlattenMode::Gentle:
        return "gentle";
    case FlattenMode::Strong:
        return "strong";
    case FlattenMode::None:
    default:
        return "none";
    }
}

const char* heightSolverName(HeightSolverMode mode) {
    return mode == HeightSolverMode::FastDct ? "fast_dct" : "robust_masked";
}

const char* heightFlattenName(HeightFlattenMode mode) {
    switch (mode) {
    case HeightFlattenMode::Plane:
        return "plane";
    case HeightFlattenMode::Radial:
        return "radial";
    case HeightFlattenMode::Quadratic:
        return "quadratic";
    case HeightFlattenMode::None:
    default:
        return "none";
    }
}

const char* rtiLayoutName(RtiLayoutMode mode) {
    switch (mode) {
    case RtiLayoutMode::DeepZoom:
        return "deepzoom";
    case RtiLayoutMode::WebRtiViewer:
        return "webrti";
    case RtiLayoutMode::Image:
    default:
        return "image";
    }
}

void requireNonemptyFile(const fs::path& path, const std::string& description) {
    std::error_code error;
    if (!fs::is_regular_file(path, error) || error || fs::file_size(path, error) == 0 || error) {
        throw std::runtime_error("Requested " + description + " was not written: " + path.string());
    }
}

bool sameExistingFile(const fs::path& a, const std::string& b) {
    if (b.empty()) {
        return false;
    }
    std::error_code error;
    return fs::equivalent(a, fs::path(b), error) && !error;
}

void rejectInputImageCollision(const fs::path& output, const Options& opt) {
    for (const std::string& input : opt.imagePaths) {
        if (sameExistingFile(output, input)) {
            throw std::runtime_error(
                "Output would overwrite an input image; choose a different output folder: " + output.string());
        }
    }
}

void removeKnownRunFiles(const Options& opt) {
    const fs::path outputDir(opt.outputDir);
    static const std::vector<std::string> names = {
        "lights.csv",
        "light_vectors.csv",
        "normal_rgb.png",
        "normal_x.png",
        "normal_y.png",
        "normal_z.png",
        "hillshade_ul.png",
        "albedo.png",
        "residual.png",
        "valid_mask.png",
        "liquid_metal.png",
        "fused_normal_rgb.png",
        "fused_normal_x.png",
        "fused_normal_y.png",
        "fused_normal_z.png",
        "fused_hillshade_ul.png",
        "classical_normal_rgb.png",
        "classical_normal_x.png",
        "classical_normal_y.png",
        "classical_normal_z.png",
        "classical_hillshade_ul.png",
        "neural_normal_rgb.png",
        "neural_normal_x.png",
        "neural_normal_y.png",
        "neural_normal_z.png",
        "neural_hillshade_ul.png",
        "neural_valid_mask.png",
        "fused_classical_confidence.png",
        "robust_weight.png",
        "robust_fallback_mask.png",
        "shadow_count.png",
        "highlight_outlier_count.png",
        "specular_cue_mask.png",
        "height.png",
        "height.pfm",
        "height_mask.png",
        "surface.ply",
        "printable_surface.ply"};
    for (const std::string& name : names) {
        rejectInputImageCollision(outputDir / name, opt);
    }
    for (const std::string& name : names) {
        const fs::path path = outputDir / name;
        // A repeat run can read calibration or masks from its previous output.
        if (sameExistingFile(path, opt.lightsFile) || sameExistingFile(path, opt.maskPath) ||
            sameExistingFile(path, opt.heightMaskPath)) {
            continue;
        }
        std::error_code error;
        fs::remove(path, error);
        if (error) {
            throw std::system_error(error, "Could not remove stale output " + path.string());
        }
    }
}

std::vector<fs::path> expectedOutputFiles(const Options& opt) {
    const fs::path outputDir(opt.outputDir);
    std::vector<fs::path> paths = {
        outputDir / "lights.csv",
        outputDir / "light_vectors.csv",
        outputDir / "normal_rgb.png",
        outputDir / "normal_x.png",
        outputDir / "normal_y.png",
        outputDir / "normal_z.png",
        outputDir / "hillshade_ul.png",
        outputDir / "albedo.png",
        outputDir / "residual.png",
        outputDir / "valid_mask.png",
        outputDir / "liquid_metal.png"};
    if (!opt.uncalibratedLighting && opt.solverMode == NormalSolverMode::Robust) {
        paths.push_back(outputDir / "robust_weight.png");
        paths.push_back(outputDir / "robust_fallback_mask.png");
        paths.push_back(outputDir / "shadow_count.png");
        paths.push_back(outputDir / "highlight_outlier_count.png");
    }
    if (!opt.uncalibratedLighting && opt.specularDiagnostics) {
        paths.push_back(outputDir / "specular_cue_mask.png");
    }
    if (opt.neuralFusion) {
        for (const std::string& prefix : {"fused", "classical", "neural"}) {
            paths.push_back(outputDir / (prefix + "_normal_rgb.png"));
            paths.push_back(outputDir / (prefix + "_normal_x.png"));
            paths.push_back(outputDir / (prefix + "_normal_y.png"));
            paths.push_back(outputDir / (prefix + "_normal_z.png"));
            paths.push_back(outputDir / (prefix + "_hillshade_ul.png"));
        }
        paths.push_back(outputDir / "neural_valid_mask.png");
        paths.push_back(outputDir / "fused_classical_confidence.png");
    }
    if (opt.calculateHeight) {
        paths.push_back(outputDir / "height.png");
        paths.push_back(outputDir / "height.pfm");
        paths.push_back(outputDir / "height_mask.png");
    }
    if (!opt.meshPath.empty()) {
        paths.emplace_back(opt.meshPath);
    }
    if (!opt.printableMeshPath.empty()) {
        paths.emplace_back(opt.printableMeshPath);
    }
    return paths;
}

fs::path rtiOutputPath(const Options& opt) {
    return opt.rtiPath.empty() ? fs::path(opt.outputDir) / "rti" : fs::path(opt.rtiPath);
}

std::vector<fs::path> collectVerifiedOutputs(const Options& opt) {
    std::vector<fs::path> outputs = expectedOutputFiles(opt);
    for (const fs::path& path : outputs) {
        requireNonemptyFile(path, "output");
    }
    if (opt.exportRti) {
        const fs::path rtiPath = rtiOutputPath(opt);
        const fs::path descriptor = opt.rtiLayoutMode == RtiLayoutMode::WebRtiViewer
            ? rtiPath / "info.xml"
            : rtiPath / "info.json";
        requireNonemptyFile(descriptor, "RTI descriptor");
        requireNonemptyFile(rtiPath / "rti_manifest.json", "RTI package manifest");
        for (const fs::directory_entry& entry : fs::recursive_directory_iterator(rtiPath)) {
            if (entry.is_regular_file()) {
                outputs.push_back(entry.path());
            }
        }
    }

    std::set<std::string> seen;
    std::vector<fs::path> unique;
    unique.reserve(outputs.size());
    for (const fs::path& path : outputs) {
        const std::string key = normalizedAbsolutePath(path);
        if (seen.insert(key).second) {
            unique.push_back(path);
        }
    }
    std::sort(unique.begin(), unique.end(), [](const fs::path& a, const fs::path& b) {
        return normalizedAbsolutePath(a) < normalizedAbsolutePath(b);
    });
    return unique;
}

void writeInputs(std::ostream& out, const Options& opt) {
    out << "  \"inputs\": [\n";
    for (size_t i = 0; i < opt.imagePaths.size(); ++i) {
        const fs::path path(opt.imagePaths[i]);
        std::error_code sizeError;
        const std::uintmax_t size = fs::is_regular_file(path, sizeError)
            ? fs::file_size(path, sizeError)
            : 0;
        std::error_code timeError;
        const auto modified = fs::last_write_time(path, timeError);
        const auto ticks = timeError ? 0 : modified.time_since_epoch().count();
        out << "    {\"path\": \"" << jsonEscape(normalizedAbsolutePath(path))
            << "\", \"size_bytes\": " << size
            << ", \"last_write_time_ticks\": " << ticks << "}";
        out << (i + 1 == opt.imagePaths.size() ? "\n" : ",\n");
    }
    out << "  ],\n";
}

void writePathOrNull(std::ostream& out, const std::string& path) {
    if (path.empty()) {
        out << "null";
    } else {
        out << '"' << jsonEscape(normalizedAbsolutePath(path)) << '"';
    }
}

void writeParameters(std::ostream& out, const Options& opt) {
    out << std::setprecision(17);
    out << "  \"parameters\": {\n";
    out << "    \"output_directory\": \""
        << jsonEscape(normalizedAbsolutePath(opt.outputDir)) << "\",\n";
    out << "    \"lights_file\": ";
    writePathOrNull(out, opt.lightsFile);
    out << ",\n";
    out << "    \"solve_mask_file\": ";
    writePathOrNull(out, opt.maskPath);
    out << ",\n";
    out << "    \"height_mask_file\": ";
    writePathOrNull(out, opt.heightMaskPath);
    out << ",\n";
    out << "    \"interactive_height_mask\": " << (opt.hasHeightMask ? "true" : "false") << ",\n";
    out << "    \"crop\": ";
    if (opt.hasCrop) {
        out << "{\"x\": " << opt.crop.x << ", \"y\": " << opt.crop.y
            << ", \"width\": " << opt.crop.width << ", \"height\": " << opt.crop.height << "}";
    } else {
        out << "null";
    }
    out << ",\n";
    out << "    \"sphere\": ";
    if (opt.hasSphere) {
        out << "{\"cx\": " << opt.sphere.cx << ", \"cy\": " << opt.sphere.cy
            << ", \"radius\": " << opt.sphere.radius << "}";
    } else {
        out << "null";
    }
    out << ",\n";
    out << "    \"keep_sphere_in_solve\": " << (opt.keepSphere ? "true" : "false") << ",\n";
    out << "    \"lighting\": \"" << lightingName(opt) << "\",\n";
    out << "    \"view_direction\": [" << opt.viewDir[0] << ", " << opt.viewDir[1]
        << ", " << opt.viewDir[2] << "],\n";
    out << "    \"normal_solver\": \"" << solverName(opt.solverMode) << "\",\n";
    out << "    \"srgb_decode\": " << (opt.srgb ? "true" : "false") << ",\n";
    out << "    \"highlight_percentile\": " << opt.highlightPercentile << ",\n";
    out << "    \"minimum_highlight\": " << opt.minHighlight << ",\n";
    out << "    \"shadow_threshold\": " << opt.shadowThreshold << ",\n";
    out << "    \"high_outlier_threshold\": " << opt.highOutlierThreshold << ",\n";
    out << "    \"normal_flattening\": \"" << flattenName(opt.flattenMode) << "\",\n";
    out << "    \"calculate_height\": " << (opt.calculateHeight ? "true" : "false") << ",\n";
    out << "    \"height_solver\": \"" << heightSolverName(opt.heightSolverMode) << "\",\n";
    out << "    \"integration_iterations\": " << opt.integrationIterations << ",\n";
    out << "    \"height_flattening\": \"" << heightFlattenName(opt.heightFlattenMode) << "\",\n";
    out << "    \"height_slope_cap\": " << opt.heightSlopeCap << ",\n";
    out << "    \"pixel_scale_mm_per_pixel\": " << opt.pixelScaleMm << ",\n";
    out << "    \"ring_radius_mm\": " << opt.ringLightRadiusMm << ",\n";
    out << "    \"ring_height_mm\": " << opt.ringLightHeightMm << ",\n";
    out << "    \"mesh_step\": " << opt.meshStep << ",\n";
    out << "    \"height_scale\": " << opt.heightScale << ",\n";
    out << "    \"mesh_path\": ";
    writePathOrNull(out, opt.meshPath);
    out << ",\n";
    out << "    \"printable_mesh_path\": ";
    writePathOrNull(out, opt.printableMeshPath);
    out << ",\n";
    out << "    \"printable_base_thickness_mm\": " << opt.printableThicknessMm << ",\n";
    out << "    \"neural_fusion\": " << (opt.neuralFusion ? "true" : "false") << ",\n";
    out << "    \"neural_model_override\": ";
    writePathOrNull(out, opt.neuralModelPath);
    out << ",\n";
    out << "    \"neural_max_side\": " << opt.neuralMaxSide << ",\n";
    out << "    \"specular_diagnostics\": " << (opt.specularDiagnostics ? "true" : "false") << ",\n";
    out << "    \"export_rti\": " << (opt.exportRti ? "true" : "false") << ",\n";
    out << "    \"rti_path\": ";
    writePathOrNull(out, opt.exportRti
        ? (opt.rtiPath.empty() ? (fs::path(opt.outputDir) / "rti").string() : opt.rtiPath)
        : std::string());
    out << ",\n";
    out << "    \"rti_layout\": \"" << rtiLayoutName(opt.rtiLayoutMode) << "\",\n";
    out << "    \"rti_color\": \"" << (opt.rtiColorMode == RtiColorMode::Lrgb ? "lrgb" : "rgb") << "\"\n";
    out << "  }";
}

void writeInProgressManifest(const Options& opt, const RunManifestContext& context) {
    CheckedOutputFile checked(fs::path(opt.outputDir) / "run_manifest.json");
    std::ostream& out = checked.stream();
    out << "{\n";
    out << "  \"schema_version\": 1,\n";
    out << "  \"status\": \"in_progress\",\n";
    out << "  \"application\": {\"name\": \"What A Relief\", \"version\": \""
        << WHAT_A_RELIEF_VERSION << "\"},\n";
    out << "  \"started_utc\": \"" << context.startedUtc << "\",\n";
    writeInputs(out, opt);
    writeParameters(out, opt);
    out << "\n}\n";
    checked.commit();
}

} // namespace

RunManifestContext beginRunManifest(const Options& opt) {
    if (opt.outputDir.empty()) {
        throw std::runtime_error("Output directory must not be empty.");
    }
    fs::create_directories(opt.outputDir);
    for (const fs::path& output : expectedOutputFiles(opt)) {
        rejectInputImageCollision(output, opt);
    }
    RunManifestContext context;
    context.startedUtc = utcNow();
    writeInProgressManifest(opt, context);
    removeKnownRunFiles(opt);
    const fs::path defaultRti = fs::path(opt.outputDir) / "rti";
    if (!opt.exportRti && fs::is_regular_file(defaultRti / "rti_manifest.json")) {
        fs::remove_all(defaultRti);
    }
    return context;
}

void completeRunManifest(
    const Options& opt,
    const RunManifestContext& context,
    const std::vector<cv::Vec3f>& lights,
    const PhotometricDiagnostics& diagnostics) {
    const std::vector<fs::path> outputs = collectVerifiedOutputs(opt);
    CheckedOutputFile checked(fs::path(opt.outputDir) / "run_manifest.json");
    std::ostream& out = checked.stream();
    out << std::setprecision(17);
    out << "{\n";
    out << "  \"schema_version\": 1,\n";
    out << "  \"status\": \"complete\",\n";
    out << "  \"application\": {\"name\": \"What A Relief\", \"version\": \""
        << WHAT_A_RELIEF_VERSION << "\"},\n";
    out << "  \"started_utc\": \"" << context.startedUtc << "\",\n";
    out << "  \"completed_utc\": \"" << utcNow() << "\",\n";
    writeInputs(out, opt);
    writeParameters(out, opt);
    out << ",\n";
    out << "  \"diagnostics\": {\"lighting_condition_number\": ";
    if (opt.uncalibratedLighting) {
        out << "null";
    } else {
        out << diagnostics.lightingConditionNumber;
    }
    out << ", \"solved_fraction\": " << diagnostics.solvedFraction << "},\n";
    out << "  \"lights\": [";
    for (size_t i = 0; i < lights.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << "[" << lights[i][0] << ", " << lights[i][1] << ", " << lights[i][2] << "]";
    }
    out << "],\n";
    out << "  \"outputs\": [\n";
    for (size_t i = 0; i < outputs.size(); ++i) {
        std::error_code error;
        const std::uintmax_t bytes = fs::file_size(outputs[i], error);
        if (error || bytes == 0) {
            throw std::runtime_error("Output disappeared while writing manifest: " + outputs[i].string());
        }
        out << "    {\"path\": \"" << jsonEscape(normalizedAbsolutePath(outputs[i]))
            << "\", \"size_bytes\": " << bytes << "}";
        out << (i + 1 == outputs.size() ? "\n" : ",\n");
    }
    out << "  ]\n";
    out << "}\n";
    checked.commit();
}
