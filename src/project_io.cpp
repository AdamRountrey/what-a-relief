#include "project_io.hpp"

#include "image_io.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <nlohmann/json.hpp>
#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace fs = std::filesystem;
using Json = nlohmann::json;

namespace {

std::string readProjectJson(std::ifstream& stream) {
    std::string text((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    if (stream.bad()) throw std::runtime_error("Could not read project JSON.");
#ifdef _WIN32
    // Older manifests used the Windows path code page rather than UTF-8.
    if (!text.empty() && !MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), static_cast<int>(text.size()), nullptr, 0)) {
        const int count = MultiByteToWideChar(CP_ACP, 0, text.data(), static_cast<int>(text.size()), nullptr, 0);
        if (!count) throw std::runtime_error("Unknown project text encoding.");
        std::wstring wide(count, L'\0');
        MultiByteToWideChar(CP_ACP, 0, text.data(), static_cast<int>(text.size()), wide.data(), count);
        const int bytes = WideCharToMultiByte(CP_UTF8, 0, wide.data(), count, nullptr, 0, nullptr, nullptr);
        text.assign(bytes, '\0');
        WideCharToMultiByte(CP_UTF8, 0, wide.data(), count, text.data(), bytes, nullptr, nullptr);
    }
#endif
    return text;
}

std::runtime_error invalid(const std::string& field) {
    return std::runtime_error("Invalid project field: " + field);
}

std::string stringValue(const Json& value, const std::string& field) {
    if (!value.is_string()) throw invalid(field);
    const std::string text = value.get<std::string>();
    if (text.find('\0') != std::string::npos) throw invalid(field);
    return text;
}

double numberValue(const Json& value, const std::string& field,
    double minimum = -std::numeric_limits<double>::max(),
    double maximum = std::numeric_limits<double>::max()) {
    if (!value.is_number()) throw invalid(field);
    const double number = value.get<double>();
    if (!std::isfinite(number) || number < minimum || number > maximum) throw invalid(field);
    return number;
}

void readNumber(const Json& object, const char* key, double& target,
    double minimum = -std::numeric_limits<double>::max(),
    double maximum = std::numeric_limits<double>::max()) {
    if (object.contains(key)) target = numberValue(object.at(key), key, minimum, maximum);
}

void readInteger(const Json& object, const char* key, int& target, int minimum) {
    if (!object.contains(key)) return;
    const auto& value = object.at(key);
    if (!value.is_number_integer()) throw invalid(key);
    target = static_cast<int>(numberValue(value, key, minimum, std::numeric_limits<int>::max()));
}

void readBool(const Json& object, const char* key, bool& target) {
    if (!object.contains(key)) return;
    if (!object.at(key).is_boolean()) throw invalid(key);
    target = object.at(key).get<bool>();
}

template<class T>
void readEnum(const Json& object, const char* key, T& target,
    std::initializer_list<std::pair<const char*, T>> choices) {
    if (!object.contains(key)) return;
    const std::string text = stringValue(object.at(key), key);
    for (const auto& choice : choices) {
        if (text == choice.first) { target = choice.second; return; }
    }
    throw invalid(key);
}

std::string optionalPath(const Json& object, const char* key) {
    if (!object.contains(key) || object.at(key).is_null()) return {};
    return stringValue(object.at(key), key);
}

} // namespace

std::string freshProjectOutputDirectory(const fs::path& requested) {
    if (requested.empty()) throw std::runtime_error("Choose an output folder.");
    const fs::path base = fs::absolute(requested).lexically_normal();
    for (unsigned int attempt = 0; attempt < 10000; ++attempt) {
        const fs::path candidate = attempt == 0 ? base : base.parent_path() /
            (base.filename().string() + "-" + std::to_string(attempt + 1));
        if (!fs::exists(candidate) || (fs::is_directory(candidate) && fs::is_empty(candidate))) {
            return candidate.string();
        }
    }
    throw std::runtime_error("Could not choose an unused output folder.");
}

LoadedProject loadCompletedProject(const fs::path& manifestOrDirectory) {
    const fs::path manifest = fs::absolute(fs::is_directory(manifestOrDirectory)
        ? manifestOrDirectory / "run_manifest.json" : manifestOrDirectory).lexically_normal();
    if (!fs::is_regular_file(manifest) || fs::file_size(manifest) > 32 * 1024 * 1024) {
        throw std::runtime_error("Choose a completed run_manifest.json (maximum 32 MiB).");
    }
    std::ifstream stream(manifest, std::ios::binary);
    if (!stream) throw std::runtime_error("Could not open project: " + manifest.string());
    const Json root = Json::parse(readProjectJson(stream), [](int depth, Json::parse_event_t, Json&) {
        if (depth > 32) throw std::runtime_error("Project JSON is nested too deeply.");
        return true;
    });
    if (!root.is_object() || root.at("schema_version") != 1 ||
        root.at("application").at("name") != "what-a-relief" || root.at("status") != "complete") {
        throw std::runtime_error("This is not a supported, completed what-a-relief project.");
    }
    const Json& p = root.at("parameters");
    const Json& inputs = root.at("inputs");
    if (!p.is_object() || !inputs.is_array() || inputs.size() < 3) throw invalid("inputs/parameters");
    LoadedProject project;
    project.manifestPath = manifest.string();
    project.completedOutputDir = manifest.parent_path().string();
    Options& opt = project.options;
    opt.guiMode = true;
    const fs::path previousRoot = fs::u8path(optionalPath(p, "output_directory"));
    auto dataPath = [&](const std::string& name) -> std::string {
        if (name.empty()) return {};
        const fs::path recorded = fs::u8path(name);
        if (!previousRoot.empty() && previousRoot.is_absolute() && recorded.is_absolute()) {
            const fs::path relative = recorded.lexically_relative(previousRoot);
            if (!relative.empty()) {
                const fs::path moved = (manifest.parent_path() / relative).lexically_normal();
                if (fs::is_regular_file(moved)) return moved.string();
            }
        }
        const fs::path resolved = recorded.is_absolute() ? recorded : manifest.parent_path() / recorded;
        if (fs::is_regular_file(resolved)) return resolved.lexically_normal().string();
        throw std::runtime_error("Project input is missing: " + name +
            "\nRestore the file or keep the original image/output folder layout when moving a project.");
    };
    std::set<std::string> uniqueImages;
    for (const Json& input : inputs) {
        const std::string path = dataPath(stringValue(input.at("path"), "input path"));
        if (!uniqueImages.insert(path).second) throw invalid("duplicate input image");
        opt.imagePaths.push_back(path);
    }
    opt.outputDir = freshProjectOutputDirectory(manifest.parent_path());
    readBool(p, "keep_sphere_in_solve", opt.keepSphere);
    readBool(p, "lights_file_order_override", opt.lightsFileByOrder);
    readBool(p, "calculate_height", opt.calculateHeight);
    readBool(p, "open_relight_viewer", opt.openRelightViewer);
    readBool(p, "printable_fill_holes", opt.printableFillHoles);
    readBool(p, "neural_fusion", opt.neuralFusion);
    readBool(p, "specular_diagnostics", opt.specularDiagnostics);
    readBool(p, "shadow_height_refinement", opt.shadowHeightRefinement);
    readBool(p, "mitsuba_inverse_refinement", opt.mitsubaInverseRefinement);
    readBool(p, "export_rti", opt.exportRti);
    const std::string lighting = stringValue(p.at("lighting"), "lighting");
    if (lighting == "uncalibrated") opt.uncalibratedLighting = true;
    else if (lighting == "near_field_ring") opt.lightingModel = LightingModel::NearFieldRing;
    else if (lighting != "directional") throw invalid("lighting");
    readEnum(p, "normal_solver", opt.solverMode,
        {{"robust", NormalSolverMode::Robust}, {"standard", NormalSolverMode::Standard}});
    readEnum(p, "normal_flattening", opt.flattenMode,
        {{"none", FlattenMode::None}, {"gentle", FlattenMode::Gentle}, {"strong", FlattenMode::Strong}});
    readEnum(p, "height_solver", opt.heightSolverMode,
        {{"robust_masked", HeightSolverMode::RobustMasked}, {"fast_dct", HeightSolverMode::FastDct}});
    readEnum(p, "height_flattening", opt.heightFlattenMode,
        {{"none", HeightFlattenMode::None}, {"plane", HeightFlattenMode::Plane},
         {"radial", HeightFlattenMode::Radial}, {"quadratic", HeightFlattenMode::Quadratic}});
    readEnum(p, "rti_layout", opt.rtiLayoutMode,
        {{"image", RtiLayoutMode::Image}, {"deepzoom", RtiLayoutMode::DeepZoom}, {"webrti", RtiLayoutMode::WebRtiViewer}});
    readEnum(p, "rti_color", opt.rtiColorMode, {{"rgb", RtiColorMode::Rgb}, {"lrgb", RtiColorMode::Lrgb}});
    readEnum(p, "mitsuba_backend_requested", opt.mitsubaBackendMode,
        {{"auto", MitsubaBackendMode::Auto}, {"cpu", MitsubaBackendMode::Cpu}, {"cuda", MitsubaBackendMode::Cuda}});
    readEnum(p, "mitsuba_quality", opt.mitsubaQualityMode,
        {{"preview", MitsubaQualityMode::Preview}, {"standard", MitsubaQualityMode::Standard}, {"research", MitsubaQualityMode::Research}});
    if (p.contains("input_response_mode")) {
        readEnum(p, "input_response_mode", opt.inputResponseMode,
            {{"auto", InputResponseMode::Auto}, {"linear", InputResponseMode::Linear}, {"srgb", InputResponseMode::Srgb}});
    } else if (p.contains("srgb_decode") && p.at("srgb_decode").is_boolean()) {
        opt.inputResponseMode = p.at("srgb_decode").get<bool>() ? InputResponseMode::Srgb : InputResponseMode::Linear;
    }
    readNumber(p, "highlight_percentile", opt.highlightPercentile, 0.000001, 99.999999);
    readNumber(p, "minimum_highlight", opt.minHighlight, 0);
    readNumber(p, "shadow_threshold", opt.shadowThreshold, 0, 1);
    readNumber(p, "high_outlier_threshold", opt.highOutlierThreshold, 0, 1);
    if (opt.highOutlierThreshold <= opt.shadowThreshold) throw invalid("exposure thresholds");
    readNumber(p, "height_slope_cap", opt.heightSlopeCap, 0);
    readNumber(p, "pixel_scale_mm_per_pixel", opt.pixelScaleMm, 0);
    readNumber(p, "ring_radius_mm", opt.ringLightRadiusMm, 0);
    readNumber(p, "ring_height_mm", opt.ringLightHeightMm, 0);
    readNumber(p, "shadow_reference_surface_z_mm", opt.shadowReferenceZMm);
    readNumber(p, "shadow_led_diameter_mm", opt.shadowLedDiameterMm, 0);
    readNumber(p, "mitsuba_light_angle_degrees", opt.mitsubaLightAngleDegrees, 0.1, 10);
    readNumber(p, "height_scale", opt.heightScale);
    readNumber(p, "printable_base_thickness_mm", opt.printableThicknessMm, 0);
    if (opt.heightScale == 0 || opt.printableThicknessMm == 0) throw invalid("height scale/base thickness");
    readInteger(p, "integration_iterations", opt.integrationIterations, 0);
    readInteger(p, "mesh_step", opt.meshStep, 1);
    readInteger(p, "neural_max_side", opt.neuralMaxSide, 0);
    if (p.contains("crop") && !p.at("crop").is_null()) {
        const Json& crop = p.at("crop");
        for (const char* key : {"x", "y", "width", "height"}) if (!crop.contains(key)) throw invalid("crop");
        readInteger(crop, "x", opt.crop.x, 0);
        readInteger(crop, "y", opt.crop.y, 0);
        readInteger(crop, "width", opt.crop.width, 1);
        readInteger(crop, "height", opt.crop.height, 1);
        if (static_cast<int64_t>(opt.crop.x) + opt.crop.width > std::numeric_limits<int>::max() ||
            static_cast<int64_t>(opt.crop.y) + opt.crop.height > std::numeric_limits<int>::max()) throw invalid("crop extent");
        opt.hasCrop = true;
    }
    if (p.contains("sphere") && !p.at("sphere").is_null()) {
        const Json& sphere = p.at("sphere");
        opt.sphere = {numberValue(sphere.at("cx"), "sphere cx", 0),
            numberValue(sphere.at("cy"), "sphere cy", 0), numberValue(sphere.at("radius"), "sphere radius", 0)};
        if (opt.sphere.radius == 0) throw invalid("sphere radius");
        opt.hasSphere = true;
    }
    if (p.contains("view_direction")) {
        const auto& v = p.at("view_direction");
        if (!v.is_array() || v.size() != 3) throw invalid("view_direction");
        cv::Vec3d view;
        for (int i = 0; i < 3; ++i) view[i] = numberValue(v.at(i), "view_direction", -1, 1);
        const double norm = cv::norm(view);
        if (norm < 1e-12) throw invalid("view_direction");
        opt.viewDir = cv::Vec3f(view / norm);
    }
    opt.maskPath = dataPath(optionalPath(p, "solve_mask_file"));
    opt.neuralModelPath = dataPath(optionalPath(p, "neural_model_override"));
    opt.heightMaskPath = dataPath(optionalPath(p, "height_mask_file"));
    bool interactive = false;
    readBool(p, "interactive_height_mask", interactive);
    if (interactive) {
        std::string selection = optionalPath(p, "project_height_mask_file");
        if (selection.empty()) {
            selection = "height_mask.png";
            if (fs::is_regular_file(manifest.parent_path() / selection)) {
                project.warnings.push_back("This older run saved only the final height mask. It is restored for height/mesh only; redraw it to reconsider excluded pixels.");
            } else {
                interactive = false;
                project.warnings.push_back("This older run did not save its specimen mask. Redraw the outline on the Geometry tab before calculating masked height.");
            }
        }
        if (interactive) {
            const std::string path = dataPath(selection);
            opt.heightMask = cv::imread(path, cv::IMREAD_GRAYSCALE);
            if (opt.heightMask.empty()) throw std::runtime_error("Could not read specimen mask: " + path);
            cv::threshold(opt.heightMask, opt.heightMask, 0, 255, cv::THRESH_BINARY);
            opt.hasHeightMask = true;
            opt.heightMaskPath.clear();
        }
    }
    if (!opt.uncalibratedLighting) {
        const std::string calibration = optionalPath(p, "lights_file");
        if (!calibration.empty()) {
            try { opt.lightsFile = dataPath(calibration); }
            catch (const std::exception&) {
                opt.lightsFile = dataPath("lights.csv");
                project.warnings.push_back("The original calibration is unavailable; the completed run's lights.csv is being reused.");
            }
        } else if (!opt.hasSphere) {
            opt.lightsFile = dataPath("lights.csv");
        }
    }
    if (!optionalPath(p, "mesh_path").empty()) opt.meshPath = (fs::path(opt.outputDir) / "surface.ply").string();
    if (!optionalPath(p, "printable_mesh_path").empty()) opt.printableMeshPath = (fs::path(opt.outputDir) / "printable_surface.ply").string();
    if (opt.exportRti) opt.rtiPath = (fs::path(opt.outputDir) / "rti").string();
    // A project is data, not authorization to run an arbitrary Python executable/script.
    if (!optionalPath(p, "mitsuba_python_override").empty() || !optionalPath(p, "mitsuba_worker_override").empty()) {
        project.warnings.push_back("Custom backend executable/script paths were not restored. Use this computer's backend configuration.");
    }
    return project;
}
