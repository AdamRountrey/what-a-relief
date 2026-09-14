#include "project_io.hpp"
#include "checked_io.hpp"
#include "image_io.hpp"
#include "run_manifest.hpp"
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;
using Json = nlohmann::json;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void writeJson(const fs::path& path, const Json& document) {
    CheckedOutputFile file(path);
    file.stream() << document.dump(2);
    file.commit();
}

int main() {
    try {
        const fs::path root = fs::absolute("project-io-test-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(root / "images");
        Options original;
        for (const char* name : {"light_c.png", "light_a.png", "light_b.png", "light_d.png"}) {
            original.imagePaths.push_back((root / "images" / name).string());
            writeImageChecked(original.imagePaths.back(), cv::Mat(16, 16, CV_8U, cv::Scalar(128)));
        }
        original.outputDir = (root / "images" / "what-a-relief").string();
        original.hasSphere = true;
        original.sphere = {2.5, 3.5, 1.25};
        original.hasCrop = true;
        original.crop = {1, 2, 12, 11};
        original.pixelScaleMm = 0.125;
        original.hasHeightMask = true;
        original.heightMask = cv::Mat::zeros(16, 16, CV_8U);
        original.heightMask(cv::Rect(2, 2, 12, 12)).setTo(255);
        original.meshStep = 2;
        original.heightScale = 1.75;
        original.printableThicknessMm = 1.25;
        original.printableFillHoles = true;
        original.meshPath = (fs::path(original.outputDir) / "surface.ply").string();
        original.printableMeshPath = (fs::path(original.outputDir) / "printable_surface.ply").string();
        original.heightFlattenMode = HeightFlattenMode::Radial;
        original.heightSolverMode = HeightSolverMode::FastDct;
        original.inputResponseMode = InputResponseMode::Linear;
        original.openRelightViewer = true;
        original.minHighlight = 0.07;
        original.viewDir = cv::Vec3f(0, 0.6f, 0.8f);
        const std::vector<cv::Vec3f> lights{{0.6f, 0, 0.8f}, {-0.6f, 0, 0.8f}, {0, 0.6f, 0.8f}, {0, -0.6f, 0.8f}};
        cv::Mat valid(16, 16, CV_8U, cv::Scalar(255));
        cv::Mat geometry = original.heightMask.clone();
        geometry.at<uchar>(7, 7) = 0;
        cv::Mat normals(16, 16, CV_32FC3, cv::Scalar(0, 0, 1));
        cv::Mat albedo(16, 16, CV_32F, cv::Scalar(0.5));
        cv::Mat height = cv::Mat::zeros(16, 16, CV_32F);
        PhotometricDiagnostics diagnostics;
        const auto context = beginRunManifest(original);
        saveOutputs(original, lights, {}, normals, albedo, height, valid, diagnostics, height, geometry);
        completeRunManifest(original, context, lights, diagnostics);
        const fs::path manifest = fs::path(original.outputDir) / "run_manifest.json";
        std::ifstream input(manifest);
        const Json recorded = Json::parse(input);
        input.close();
        const auto loaded = loadCompletedProject(manifest);
        const Options& opt = loaded.options;
        require(opt.imagePaths == original.imagePaths, "Image order changed");
        require(opt.guiMode && !opt.noGui, "Loaded project is not an editable GUI project");
        require(opt.hasSphere && opt.sphere.radius == 1.25 && opt.hasCrop && opt.crop == original.crop, "Sphere/crop did not round-trip");
        require(opt.hasHeightMask && cv::countNonZero(opt.heightMask != original.heightMask) == 0, "Original specimen selection was not preserved");
        require(opt.maskPath.empty() && opt.heightMaskPath.empty(), "Specimen mask became a solve mask");
        require(opt.heightMask.at<uchar>(7, 7) == 255, "Final validity exclusions leaked into stored specimen selection");
        require(opt.pixelScaleMm == 0.125 && opt.heightScale == 1.75 && opt.meshStep == 2, "Scale/sampling changed");
        require(opt.printableFillHoles && opt.printableThicknessMm == 1.25, "Print settings changed");
        require(opt.heightSolverMode == HeightSolverMode::FastDct && opt.heightFlattenMode == HeightFlattenMode::Radial, "Height settings changed");
        require(opt.inputResponseMode == InputResponseMode::Linear && opt.openRelightViewer, "Response/viewer settings changed");
        require(opt.outputDir != original.outputDir && !fs::exists(opt.outputDir), "Loading modified or reused the old output folder");
        require(fs::path(opt.meshPath).parent_path() == fs::path(opt.outputDir), "Mesh still targets old output");
        require(loaded.warnings.empty(), "New-format project unexpectedly needs a warning");

        Json legacy = recorded;
        legacy["parameters"].erase("project_height_mask_file");
        legacy["parameters"].erase("input_response_mode");
        legacy["parameters"]["srgb_decode"] = true;
        writeJson(manifest, legacy);
        auto older = loadCompletedProject(manifest.parent_path());
        require(!older.warnings.empty() && older.options.heightMask.at<uchar>(7, 7) == 0, "Legacy height-mask fallback was not explicit");
        require(older.options.maskPath.empty() && older.options.inputResponseMode == InputResponseMode::Srgb, "Legacy options misinterpreted");
        const fs::path noHeightFolder = root / "legacy-no-height";
        fs::create_directory(noHeightFolder);
        Json noHeight = legacy;
        noHeight["parameters"]["calculate_height"] = false;
        noHeight["parameters"]["mesh_path"] = nullptr;
        noHeight["parameters"]["printable_mesh_path"] = nullptr;
        noHeight["parameters"]["printable_fill_holes"] = false;
        writeJson(noHeightFolder / "run_manifest.json", noHeight);
        const auto missingLegacyMask = loadCompletedProject(noHeightFolder);
        require(!missingLegacyMask.options.hasHeightMask && !missingLegacyMask.options.calculateHeight &&
            !missingLegacyMask.warnings.empty(), "Older no-height project did not explain its unavailable mask");

        Json advanced = recorded;
        auto& p = advanced["parameters"];
        p["sphere"] = nullptr;
        p["lighting"] = "near_field_ring";
        p["lights_file"] = "missing-original-calibration.csv";
        p["ring_radius_mm"] = 37.5;
        p["ring_height_mm"] = 25.0;
        p["neural_fusion"] = true;
        p["mitsuba_inverse_refinement"] = true;
        p["mitsuba_quality"] = "research";
        p["mitsuba_backend_requested"] = "cpu";
        p["mitsuba_python_override"] = "C:/untrusted/python.exe";
        p["mitsuba_worker_override"] = "C:/untrusted/worker.py";
        p["export_rti"] = true;
        p["rti_layout"] = "webrti";
        p["rti_color"] = "lrgb";
        writeJson(manifest, advanced);
        auto restored = loadCompletedProject(manifest);
        require(!restored.options.lightsFile.empty() && restored.options.lightingModel == LightingModel::NearFieldRing, "Saved calibration fallback lost ring mode");
        require(restored.options.ringLightRadiusMm == 37.5 && restored.options.ringLightHeightMm == 25, "Ring geometry changed");
        require(restored.options.mitsubaInverseRefinement && restored.options.mitsubaQualityMode == MitsubaQualityMode::Research, "Inverse settings changed");
        require(restored.options.mitsubaPythonPath.empty() && restored.options.mitsubaWorkerPath.empty(), "Project restored executable overrides");
        require(restored.options.exportRti && restored.options.rtiLayoutMode == RtiLayoutMode::WebRtiViewer && restored.options.rtiColorMode == RtiColorMode::Lrgb, "RTI settings changed");
        require(restored.options.neuralFusion && restored.warnings.size() == 2, "Expected restoration warnings missing");

        Json uncalibrated = recorded;
        uncalibrated["parameters"]["lighting"] = "uncalibrated";
        uncalibrated["parameters"]["sphere"] = nullptr;
        uncalibrated["diagnostics"]["lighting_condition_number"] = nullptr;
        writeJson(manifest, uncalibrated);
        require(loadCompletedProject(manifest).options.uncalibratedLighting, "Uncalibrated/null JSON failed");

        for (int bad = 0; bad < 7; ++bad) {
            Json broken = recorded;
            switch (bad) {
            case 0: broken["status"] = "in_progress"; break;
            case 1: broken["schema_version"] = 99; break;
            case 2: broken["parameters"]["height_solver"] = "guess"; break;
            case 3: broken["parameters"]["mesh_step"] = 0; break;
            case 4: broken["inputs"][0]["path"] = "missing.png"; break;
            case 5: broken["inputs"][1] = broken["inputs"][0]; break;
            case 6: broken["parameters"]["calculate_height"] = "false"; break;
            }
            writeJson(manifest, broken);
            bool rejected = false;
            try { (void)loadCompletedProject(manifest); } catch (const std::exception&) { rejected = true; }
            require(rejected, "Malformed project accepted: " + std::to_string(bad));
        }
        writeJson(manifest, recorded);
        const fs::path moved = root / "moved";
        fs::copy(root / "images", moved, fs::copy_options::recursive);
        const auto relocated = loadCompletedProject(moved / "what-a-relief");
        require(fs::path(relocated.options.imagePaths.front()).parent_path() == moved, "Moved project ignored adjacent input copies");
        require(!fs::exists(relocated.options.outputDir), "Loading a moved project wrote outputs");
        const fs::path accented = root / "images" / fs::u8path(u8"\u00e9chantillon.png");
        writeImageChecked(accented, cv::Mat(16, 16, CV_8U, cv::Scalar(128)));
        Json unicode = recorded;
        unicode["inputs"][0]["path"] = accented.u8string();
        writeJson(manifest, unicode);
        require(loadCompletedProject(manifest).options.imagePaths[0] == accented.string(), "UTF-8 image path failed");
#ifdef _WIN32
        Options legacyEncoding = original;
        legacyEncoding.outputDir = (root / "legacy-path-encoding").string();
        legacyEncoding.imagePaths[0] = accented.string();
        legacyEncoding.meshPath.clear();
        legacyEncoding.printableMeshPath.clear();
        legacyEncoding.printableFillHoles = false;
        const auto legacyContext = beginRunManifest(legacyEncoding);
        saveOutputs(legacyEncoding, lights, {}, normals, albedo, height, valid, diagnostics, height, geometry);
        completeRunManifest(legacyEncoding, legacyContext, lights, diagnostics);
        require(loadCompletedProject(legacyEncoding.outputDir).options.imagePaths[0] == accented.string(), "Windows legacy path encoding failed");
#endif
        std::cout << "Project round-trip, legacy, relocation, safety, and mask tests passed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
