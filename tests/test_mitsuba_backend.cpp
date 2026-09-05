#include "mitsuba_backend.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: test_mitsuba_backend <fake-python-executable> <worker-file>\n";
        return 2;
    }
    const fs::path relativeOutput = "mitsuba-contract-output";
    const fs::path output = fs::current_path() / relativeOutput;
    std::error_code error;
    fs::remove_all(output, error);
    try {
        Options options;
        options.outputDir = relativeOutput.string();
        options.solverMode = NormalSolverMode::Robust;
        options.calculateHeight = true;
        options.mitsubaInverseRefinement = true;
        options.mitsubaBackendMode = MitsubaBackendMode::Cuda;
        options.mitsubaQualityMode = MitsubaQualityMode::Preview;
        options.mitsubaPythonPath = argv[1];
        options.mitsubaWorkerPath = argv[2];
        for (int i = 0; i < 6; ++i) {
            options.imagePaths.push_back("nonexistent_contract_image_" + std::to_string(i) + ".tif");
        }
        fs::create_directories(output);

        const cv::Mat albedo(8, 8, CV_32F, cv::Scalar(0.5f));
        const cv::Mat height(8, 8, CV_32F, cv::Scalar(0.0f));
        const cv::Mat mask(8, 8, CV_8U, cv::Scalar(255));
        std::vector<cv::Vec3f> lights;
        std::vector<cv::Mat> images;
        for (int i = 0; i < 6; ++i) {
            const float angle = static_cast<float>(2.0 * CV_PI * i / 6.0);
            lights.emplace_back(0.6f * std::cos(angle), 0.6f * std::sin(angle), 0.8f);
            images.emplace_back(8, 8, CV_32F, cv::Scalar(0.15f + 0.1f * static_cast<float>(i)));
        }
        PhotometricDiagnostics diagnostics;
        diagnostics.robustWeight = cv::Mat(8, 8, CV_32F, cv::Scalar(1.0f));
        int lastProgress = -1;
        runMitsubaInverseRefinement(
            options,
            lights,
            images,
            albedo,
            height,
            mask,
            diagnostics,
            [&](const std::string&, int percent) { lastProgress = percent; },
            []() { return false; });

        require(lastProgress == 100, "Backend contract did not report completion");
        require(diagnostics.mitsuba.attempted, "Backend attempt was not recorded");
        require(diagnostics.mitsuba.succeeded, "Completed backend result was not recognized");
        require(!diagnostics.mitsuba.accepted, "Fake rejected candidate was marked accepted");
        require(diagnostics.mitsuba.decision == "fake_contract_rejection", "Decision was not parsed");
        require(diagnostics.mitsuba.iterationsCompleted == 3, "Iteration count was not parsed");
        require(fs::is_regular_file(output / "inverse" / "result.json"), "Result was not committed");
        require(
            !fs::exists(output / "inverse" / "input_observations"),
            "Temporary linear observation handoff was retained in completed outputs");
        require(!fs::exists(output / "inverse.part"), "A partial inverse directory was left behind");

        fs::remove_all(output, error);
        std::cout << "Mitsuba process and output contract passed.\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        fs::remove_all(output, error);
        return 1;
    }
}
