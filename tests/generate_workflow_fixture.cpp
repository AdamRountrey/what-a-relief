#include "checked_io.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

cv::Vec3f normalized(const cv::Vec3f& value) {
    return value / std::sqrt(value.dot(value));
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::runtime_error("Usage: fixture-generator output-directory");
        }
        const fs::path root(argv[1]);
        fs::remove_all(root);
        fs::create_directories(root / "inputs");

        constexpr int rows = 36;
        constexpr int cols = 48;
        constexpr int lightCount = 8;
        std::vector<cv::Vec3f> lights;
        lights.reserve(lightCount);
        for (int i = 0; i < lightCount; ++i) {
            const double angle = 2.0 * 3.14159265358979323846 * static_cast<double>(i) / lightCount;
            lights.push_back(normalized(cv::Vec3f(
                0.66f * static_cast<float>(std::cos(angle)),
                0.66f * static_cast<float>(std::sin(angle)),
                0.7512656f)));
        }

        for (int i = 0; i < lightCount; ++i) {
            cv::Mat image(rows, cols, CV_16U);
            for (int y = 0; y < rows; ++y) {
                unsigned short* row = image.ptr<unsigned short>(y);
                for (int x = 0; x < cols; ++x) {
                    const double xf = static_cast<double>(x) / static_cast<double>(cols - 1);
                    const double yf = static_cast<double>(y) / static_cast<double>(rows - 1);
                    const float p = static_cast<float>(0.18 * std::sin(6.283185307179586 * xf));
                    const float q = static_cast<float>(0.13 * std::cos(6.283185307179586 * yf));
                    const cv::Vec3f normal = normalized(cv::Vec3f(-p, q, 1.0f));
                    const double albedo = 0.52 + 0.16 * xf + 0.08 * yf;
                    double intensity = albedo * std::max(0.0f, normal.dot(lights[static_cast<size_t>(i)]));
                    if (i == 1 && x >= 8 && x < 16 && y >= 10 && y < 26) {
                        intensity = 0.0;
                    }
                    if (i == 5 && x >= 27 && x < 34 && y >= 12 && y < 20) {
                        intensity = 1.0;
                    }
                    row[x] = static_cast<unsigned short>(std::clamp(std::round(intensity * 60000.0), 0.0, 65535.0));
                }
            }
            writeImageChecked(root / "inputs" / ("image_" + std::to_string(i) + ".png"), image);
        }

        CheckedOutputFile lightsFile(root / "light_vectors.csv");
        std::ostream& out = lightsFile.stream();
        out << "x,y,z\n";
        for (const cv::Vec3f& light : lights) {
            out << light[0] << ',' << light[1] << ',' << light[2] << '\n';
        }
        lightsFile.commit();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
