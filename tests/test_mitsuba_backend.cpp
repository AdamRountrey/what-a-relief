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
        cv::Mat normals(8, 8, CV_32FC3);
        for (int y = 0; y < 8; ++y) {
            for (int x = 0; x < 8; ++x) {
                const float nx = 0.02f * (x - 3);
                const float ny = 0.03f * (y - 4);
                normals.at<cv::Vec3f>(y, x) = cv::Vec3f(nx, ny, std::sqrt(1.0f - nx * nx - ny * ny));
            }
        }
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
        std::string lastMessage;
        const auto run = [&]() { runMitsubaInverseRefinement(
            options,
            lights,
            images,
            albedo,
            height,
            mask,
            normals,
            std::vector<cv::Mat>(6, cv::Mat(8, 8, CV_8U, cv::Scalar(0))),
            diagnostics,
            [&](const std::string& message, int percent) { lastProgress = percent; lastMessage = message; },
            []() { return false; }); };
        run();

        require(lastProgress == 100, "Backend contract did not report completion");
        require(diagnostics.mitsuba.attempted, "Backend attempt was not recorded");
        require(diagnostics.mitsuba.succeeded, "Completed backend result was not recognized");
        require(!diagnostics.mitsuba.accepted, "Fake rejected candidate was marked accepted");
        require(diagnostics.mitsuba.decision == "fake_contract_rejection", "Decision was not parsed");
        require(diagnostics.mitsuba.candidateSaved, "Unvalidated candidate was not recorded");
        require(diagnostics.mitsuba.candidateExportStatus == "saved_unvalidated", "Candidate export status was lost");
        require(fs::is_regular_file(diagnostics.mitsuba.candidateReviewPath), "Review path did not survive staging rename");
        require(lastMessage.find("fake_contract_rejection") != std::string::npos, "Progress omitted rejection reason");
        require(lastMessage.find("Unvalidated candidate review:") != std::string::npos, "Progress omitted review path");
        require(diagnostics.mitsuba.iterationsCompleted == 3, "Iteration count was not parsed");
        require(fs::is_regular_file(output / "inverse" / "result.json"), "Result was not committed");
        require(
            !fs::exists(output / "inverse" / "input_observations"),
            "Temporary linear observation handoff was retained in completed outputs");
        require(!fs::exists(output / "inverse.part"), "A partial inverse directory was left behind");

        // The fake worker uses Standard to return an accepted result on the next run.
        options.mitsubaQualityMode = MitsubaQualityMode::Standard;
        run();
        require(diagnostics.mitsuba.accepted, "Accepted rerun was not recognized");
        require(!diagnostics.mitsuba.candidateSaved && diagnostics.mitsuba.candidateReviewPath.empty(),
                "Accepted rerun advertised a stale candidate");
        require(!fs::exists(output / "inverse" / "unvalidated_candidate"),
                "Accepted rerun retained stale rejected candidate files");

        fs::remove_all(output, error);
        std::cout << "Mitsuba process and output contract passed.\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        fs::remove_all(output, error);
        return 1;
    }
}
