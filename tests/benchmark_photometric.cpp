#include "photometric.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

std::vector<cv::Vec3f> makeRingLights(int count) {
    std::vector<cv::Vec3f> lights;
    lights.reserve(static_cast<size_t>(count));
    constexpr float z = 0.70f;
    const float radial = std::sqrt(1.0f - z * z);
    for (int i = 0; i < count; ++i) {
        const double angle = 2.0 * kPi * static_cast<double>(i) / static_cast<double>(count);
        lights.emplace_back(
            radial * static_cast<float>(std::cos(angle)),
            radial * static_cast<float>(std::sin(angle)),
            z);
    }
    return lights;
}

std::vector<cv::Mat> renderFixture(
    int rows,
    int cols,
    const std::vector<cv::Vec3f>& lights) {
    std::vector<cv::Mat> images;
    images.reserve(lights.size());
    for (const cv::Vec3f& light : lights) {
        cv::Mat image(rows, cols, CV_32F);
        for (int y = 0; y < rows; ++y) {
            float* row = image.ptr<float>(y);
            const float fy = 2.0f * static_cast<float>(y) / static_cast<float>(rows - 1) - 1.0f;
            for (int x = 0; x < cols; ++x) {
                const float fx = 2.0f * static_cast<float>(x) / static_cast<float>(cols - 1) - 1.0f;
                cv::Vec3f normal(-0.24f * fx, 0.18f * fy, 1.0f);
                normal /= std::sqrt(normal.dot(normal));
                const float albedo = 0.55f + 0.12f * std::sin(9.0f * fx) * std::cos(7.0f * fy);
                row[x] = albedo * std::max(0.0f, normal.dot(light));
            }
        }
        images.push_back(std::move(image));
    }

    // Deterministic corruption exercises both robust rejection paths.
    images[1](cv::Rect(cols / 7, rows / 6, cols / 5, rows / 4)).setTo(0.0f);
    images[5](cv::Rect(cols / 2, rows / 3, cols / 6, rows / 5)).setTo(1.0f);
    return images;
}

}  // namespace

int main(int argc, char** argv) {
    const int rows = argc > 1 ? std::max(32, std::atoi(argv[1])) : 768;
    const int cols = argc > 2 ? std::max(32, std::atoi(argv[2])) : 1024;
    constexpr int imageCount = 8;
    const std::vector<cv::Vec3f> lights = makeRingLights(imageCount);
    const std::vector<cv::Mat> images = renderFixture(rows, cols, lights);
    const cv::Mat inputMask(rows, cols, CV_8U, cv::Scalar(255));

    cv::Mat normals;
    cv::Mat albedo;
    cv::Mat residual;
    cv::Mat validMask;
    PhotometricDiagnostics diagnostics;
    const auto start = std::chrono::steady_clock::now();
    solvePhotometricStereo(
        images,
        lights,
        inputMask,
        0.02f,
        NormalSolverMode::Robust,
        0.98f,
        LightingModel::Directional,
        10.0,
        10.0,
        0.01,
        cv::Point2d((cols - 1) * 0.5, (rows - 1) * 0.5),
        cv::Vec3f(0.0f, 0.0f, 1.0f),
        normals,
        albedo,
        residual,
        validMask,
        diagnostics);
    const auto end = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(end - start).count();
    const cv::Scalar normalSum = cv::sum(normals);
    const double checksum = normalSum[0] + 3.0 * normalSum[1] + 7.0 * normalSum[2] + cv::sum(albedo)[0];

    std::cout << std::fixed << std::setprecision(6)
              << "photometric_benchmark rows=" << rows
              << " cols=" << cols
              << " images=" << imageCount
              << " seconds=" << seconds
              << " megapixels_per_second="
              << (static_cast<double>(rows) * static_cast<double>(cols) / 1.0e6 / seconds)
              << " solved_fraction=" << diagnostics.solvedFraction
              << " checksum=" << checksum << '\n';
    return diagnostics.solvedFraction > 0.99 ? 0 : 1;
}
