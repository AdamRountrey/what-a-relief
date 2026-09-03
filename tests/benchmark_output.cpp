#include "image_io.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

std::vector<cv::Vec3f> makeLights() {
    std::vector<cv::Vec3f> lights;
    constexpr int count = 8;
    constexpr double pi = 3.14159265358979323846;
    for (int i = 0; i < count; ++i) {
        const double angle = 2.0 * pi * static_cast<double>(i) / count;
        cv::Vec3f light(
            static_cast<float>(0.62 * std::cos(angle)),
            static_cast<float>(0.62 * std::sin(angle)),
            0.78f);
        light /= std::sqrt(light.dot(light));
        lights.push_back(light);
    }
    return lights;
}

std::uintmax_t directoryBytes(const fs::path& root) {
    std::uintmax_t total = 0;
    for (const fs::directory_entry& entry : fs::recursive_directory_iterator(root)) {
        if (entry.is_regular_file()) {
            total += entry.file_size();
        }
    }
    return total;
}

} // namespace

int main(int argc, char** argv) {
    const int rows = argc > 1 ? std::max(32, std::atoi(argv[1])) : 768;
    const int cols = argc > 2 ? std::max(32, std::atoi(argv[2])) : 1024;
    const int step = argc > 3 ? std::max(1, std::atoi(argv[3])) : 1;
    const int requestedThreads = argc > 4 ? std::max(1, std::atoi(argv[4])) : 0;
    if (requestedThreads > 0) {
        cv::setNumThreads(requestedThreads);
    }
    const fs::path root = fs::temp_directory_path() / "what-a-relief-output-benchmark";
    fs::remove_all(root);

    cv::Mat normals(rows, cols, CV_32FC3);
    cv::Mat albedo(rows, cols, CV_32F);
    cv::Mat residual(rows, cols, CV_32F, cv::Scalar(0.01f));
    cv::Mat height(rows, cols, CV_32F);
    const cv::Mat mask(rows, cols, CV_8U, cv::Scalar(255));
    for (int y = 0; y < rows; ++y) {
        cv::Vec3f* normalRow = normals.ptr<cv::Vec3f>(y);
        float* albedoRow = albedo.ptr<float>(y);
        float* heightRow = height.ptr<float>(y);
        for (int x = 0; x < cols; ++x) {
            const float fx = static_cast<float>(x) / std::max(1, cols - 1);
            const float fy = static_cast<float>(y) / std::max(1, rows - 1);
            cv::Vec3f normal(-0.12f * std::cos(12.0f * fx), 0.10f * std::sin(10.0f * fy), 1.0f);
            normal /= std::sqrt(normal.dot(normal));
            normalRow[x] = normal;
            albedoRow[x] = 0.55f + 0.15f * std::sin(8.0f * fx) * std::cos(7.0f * fy);
            heightRow[x] = 5.0f * std::sin(5.0f * fx) * std::cos(4.0f * fy);
        }
    }

    Options opt;
    opt.outputDir = root.string();
    opt.calculateHeight = true;
    opt.solverMode = NormalSolverMode::Standard;
    opt.meshPath = (root / "surface.ply").string();
    opt.printableMeshPath = (root / "printable_surface.ply").string();
    opt.pixelScaleMm = 0.01;
    opt.meshStep = step;
    const std::vector<cv::Vec3f> lights = makeLights();
    for (int i = 0; i < static_cast<int>(lights.size()); ++i) {
        opt.imagePaths.push_back("benchmark_image_" + std::to_string(i) + ".tif");
    }

    const auto start = std::chrono::steady_clock::now();
    saveOutputs(opt, lights, {}, normals, albedo, residual, mask, {}, height, mask);
    const auto end = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(end - start).count();
    const std::uintmax_t bytes = directoryBytes(root);
    std::cout << std::fixed << std::setprecision(3)
              << "output_benchmark rows=" << rows
              << " cols=" << cols
              << " step=" << step
              << " threads=" << cv::getNumThreads()
              << " seconds=" << seconds
              << " output_megabytes=" << static_cast<double>(bytes) / 1.0e6
              << " megabytes_per_second=" << static_cast<double>(bytes) / 1.0e6 / seconds
              << '\n';
    fs::remove_all(root);
    return 0;
}
