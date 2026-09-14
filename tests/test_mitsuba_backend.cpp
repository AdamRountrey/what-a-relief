#include "mitsuba_backend.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <queue>
#include <sstream>
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

std::string fileText(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(input), {});
}

void checkInverseSolid(const fs::path& path, bool accepted) {
    std::ifstream input(path, std::ios::binary);
    std::string line, header;
    size_t vertexCount = 0, faceCount = 0;
    while (std::getline(input, line)) {
        header += line + "\n";
        if (line.rfind("element vertex ", 0) == 0) vertexCount = std::stoul(line.substr(15));
        if (line.rfind("element face ", 0) == 0) faceCount = std::stoul(line.substr(13));
        if (line == "end_header") break;
    }
    require(vertexCount > 0 && vertexCount % 2 == 0 && faceCount > 0, "Inverse solid has no topology");
    require(header.find("units millimeters") != std::string::npos, "Inverse printable units are not mm");
    require(header.find(accepted ? "accepted inverse" : "inverse refinement rejected") != std::string::npos,
            "Printable source/acceptance annotation missing");
    const size_t topCount = vertexCount / 2;
    for (size_t index = 0; index < vertexCount; ++index) {
        std::array<float, 3> point{};
        std::array<unsigned char, 3> color{};
        input.read(reinterpret_cast<char*>(point.data()), sizeof(point));
        input.read(reinterpret_cast<char*>(color.data()), sizeof(color));
        require(input.good(), "Truncated inverse printable vertex");
        // Scale 0.25 mm/pixel, mesh step 2, height exaggeration 3, base 1.5 mm.
        require(std::abs(point[0] / 0.5f - std::round(point[0] / 0.5f)) < 1e-5f &&
                std::abs(point[1] / 0.5f - std::round(point[1] / 0.5f)) < 1e-5f,
                "Inverse mesh did not inherit mm scale and sampling step");
        const float expected = index >= topCount ? -1.5f :
            accepted ? 3.0f * (0.1f * point[0] - 0.2f * point[1]) : 0.0f;
        require(std::abs(point[2] - expected) < 1e-5f, "Inverse solid used wrong height, PFM row order, Z scale, or base");
        require(color[0] > 0 && color[0] == color[1] && color[1] == color[2], "Inverse albedo colors missing");
    }
    std::map<std::pair<int, int>, std::pair<int, int>> edges;
    std::vector<std::vector<int>> neighbors(vertexCount);
    for (size_t face = 0; face < faceCount; ++face) {
        unsigned char count = 0;
        input.read(reinterpret_cast<char*>(&count), 1);
        require(input.good() && (count == 3 || count == 4), "Inverse solid face is invalid");
        std::vector<std::int32_t> faceVertices(count);
        input.read(reinterpret_cast<char*>(faceVertices.data()), count * sizeof(std::int32_t));
        require(input.good(), "Truncated inverse solid face");
        for (size_t edge = 0; edge < count; ++edge) {
            const int a = faceVertices[edge], b = faceVertices[(edge + 1) % count];
            require(a >= 0 && b >= 0 && a != b && static_cast<size_t>(a) < vertexCount &&
                    static_cast<size_t>(b) < vertexCount, "Inverse solid index out of range");
            auto& incidence = edges[{std::min(a, b), std::max(a, b)}];
            ++incidence.first;
            incidence.second += a < b ? 1 : -1;
            neighbors[a].push_back(b);
            neighbors[b].push_back(a);
        }
    }
    for (const auto& edge : edges) {
        require(edge.second.first == 2 && edge.second.second == 0, "Inverse printable mesh is not closed and oriented");
    }
    std::vector<bool> visited(vertexCount, false);
    std::queue<int> queue;
    queue.push(0);
    visited[0] = true;
    size_t reached = 0;
    while (!queue.empty()) {
        const int current = queue.front();
        queue.pop();
        ++reached;
        for (int next : neighbors[current]) {
            if (!visited[next]) { visited[next] = true; queue.push(next); }
        }
    }
    require(reached == vertexCount, "Inverse printable mesh has disconnected components");
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
        const fs::path printable = output / "inverse" / "inverse_printable_surface.ply";

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
        bool receivedLive = false;
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
            []() { return false; },
            [&](const ProgressUpdate& update) {
                require(update.stage == "Mitsuba optimization" && update.completed == 5 && update.total == 12 &&
                    update.remainingSeconds == 23, "Backend live progress lost iteration/ETA metadata");
                receivedLive = true;
            }); };
        run();

        require(lastProgress == 100, "Backend contract did not report completion");
        require(receivedLive, "Backend did not forward live progress");
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
        require(!fs::exists(printable), "Inverse solid exported without printable option");

        options.printableMeshPath = (output / "printable_surface.ply").string();
        options.pixelScaleMm = 0.25;
        options.printableThicknessMm = 1.5;
        options.meshStep = 2;
        options.heightScale = 3;
        options.printableFillHoles = true;
        { std::ofstream original(options.printableMeshPath); original << "baseline printable sentinel\n"; }
        const std::string baselinePrintable = fileText(options.printableMeshPath);
        run();
        checkInverseSolid(printable, false);
        require(fs::is_regular_file(output / "inverse" / "printable_fill_mask.png"), "Inverse fill audit missing");
        require(!fs::exists(output / "inverse" / "unvalidated_candidate" / "inverse_printable_surface.ply"),
                "Rejected candidate was silently made printable");
        require(fileText(output / "inverse" / "inverse_printable_surface.json").find("retained_baseline") != std::string::npos,
                "Rejected inverse solid provenance lost");

        // The fake worker uses Standard to return an accepted result on the next run.
        options.mitsubaQualityMode = MitsubaQualityMode::Standard;
        run();
        require(diagnostics.mitsuba.accepted, "Accepted rerun was not recognized");
        require(!diagnostics.mitsuba.candidateSaved && diagnostics.mitsuba.candidateReviewPath.empty(),
                "Accepted rerun advertised a stale candidate");
        require(!fs::exists(output / "inverse" / "unvalidated_candidate"),
                "Accepted rerun retained stale rejected candidate files");
        checkInverseSolid(printable, true);
        require(fileText(options.printableMeshPath) == baselinePrintable, "Baseline printable was overwritten");
        require(cv::norm(height, cv::NORM_INF) == 0, "Inverse export mutated baseline heights");
        const std::string previousSolid = fileText(printable);
        options.mitsubaQualityMode = MitsubaQualityMode::Research;
        bool rejectedMalformed = false;
        try { run(); } catch (const std::exception& exception) {
            rejectedMalformed = std::string(exception.what()).find("full-resolution float height") != std::string::npos;
        }
        require(rejectedMalformed, "Malformed inverse PFM was allowed into printable export");
        require(fileText(printable) == previousSolid, "Failed printable export replaced previous inverse results");

        options.mitsubaQualityMode = MitsubaQualityMode::Standard;
        options.printableMeshPath.clear();
        run();
        require(!fs::exists(printable) && !fs::exists(output / "inverse" / "printable_fill_mask.png"),
                "Disabled printable option retained stale inverse solid products");

        fs::remove_all(output, error);
        std::cout << "Mitsuba process and output contract passed.\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        fs::remove_all(output, error);
        return 1;
    }
}
