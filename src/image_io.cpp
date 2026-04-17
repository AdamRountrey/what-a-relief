#include "image_io.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace {

[[noreturn]] void die(const std::string& message) {
    throw std::runtime_error(message);
}

void reportProgress(
    const std::function<void(const std::string&)>& progress,
    const std::string& message) {
    if (progress) {
        progress(message);
    }
}

bool insideSphere(const Sphere& sphere, int x, int y, double margin = 0.0) {
    const double dx = static_cast<double>(x) - sphere.cx;
    const double dy = static_cast<double>(y) - sphere.cy;
    const double r = sphere.radius + margin;
    return dx * dx + dy * dy <= r * r;
}

cv::Mat toFloatLuminance(const cv::Mat& input, bool srgb) {
    const int depth = input.depth();
    double scale = 1.0;
    if (depth == CV_8U) {
        scale = 1.0 / 255.0;
    } else if (depth == CV_16U) {
        scale = 1.0 / 65535.0;
    } else if (depth == CV_16S) {
        scale = 1.0 / 32767.0;
    }

    cv::Mat f;
    input.convertTo(f, CV_MAKETYPE(CV_32F, input.channels()), scale);

    cv::Mat gray;
    if (f.channels() == 1) {
        gray = f;
    } else {
        std::vector<cv::Mat> channels;
        cv::split(f, channels);
        if (channels.size() < 3) {
            gray = channels[0];
        } else {
            gray = 0.0722f * channels[0] + 0.7152f * channels[1] + 0.2126f * channels[2];
        }
    }

    cv::max(gray, 0.0f, gray);
    if (srgb) {
        cv::pow(gray, 2.2, gray);
    }
    return gray;
}

void minMaxMasked(const cv::Mat& src, const cv::Mat& mask, double& minValue, double& maxValue) {
    minValue = std::numeric_limits<double>::infinity();
    maxValue = -std::numeric_limits<double>::infinity();
    for (int y = 0; y < src.rows; ++y) {
        const float* row = src.ptr<float>(y);
        const uchar* mrow = mask.ptr<uchar>(y);
        for (int x = 0; x < src.cols; ++x) {
            if (mrow[x] == 0 || !std::isfinite(row[x])) {
                continue;
            }
            minValue = std::min(minValue, static_cast<double>(row[x]));
            maxValue = std::max(maxValue, static_cast<double>(row[x]));
        }
    }
    if (!std::isfinite(minValue) || !std::isfinite(maxValue) || minValue == maxValue) {
        minValue = 0.0;
        maxValue = 1.0;
    }
}

cv::Mat normalizeFloatTo8U(const cv::Mat& src, const cv::Mat& mask, bool forceZeroMin) {
    double minValue = 0.0;
    double maxValue = 1.0;
    minMaxMasked(src, mask, minValue, maxValue);
    if (forceZeroMin) {
        minValue = 0.0;
    }
    const double scale = 255.0 / std::max(1.0e-12, maxValue - minValue);
    cv::Mat out(src.rows, src.cols, CV_8U, cv::Scalar(0));
    for (int y = 0; y < src.rows; ++y) {
        const float* row = src.ptr<float>(y);
        const uchar* mrow = mask.ptr<uchar>(y);
        uchar* orow = out.ptr<uchar>(y);
        for (int x = 0; x < src.cols; ++x) {
            if (mrow[x] == 0 || !std::isfinite(row[x])) {
                continue;
            }
            const double v = (static_cast<double>(row[x]) - minValue) * scale;
            orow[x] = static_cast<uchar>(std::clamp(v, 0.0, 255.0));
        }
    }
    return out;
}

cv::Mat normalComponentTo8U(const cv::Mat& normalMap, const cv::Mat& mask, int component, bool signedComponent) {
    cv::Mat out(normalMap.rows, normalMap.cols, CV_8U, cv::Scalar(0));
    for (int y = 0; y < normalMap.rows; ++y) {
        const cv::Vec3f* nrow = normalMap.ptr<cv::Vec3f>(y);
        const uchar* mrow = mask.ptr<uchar>(y);
        uchar* orow = out.ptr<uchar>(y);
        for (int x = 0; x < normalMap.cols; ++x) {
            if (mrow[x] == 0) {
                continue;
            }
            double value = nrow[x][component];
            if (signedComponent) {
                value = value * 0.5 + 0.5;
            }
            orow[x] = static_cast<uchar>(std::clamp(value * 255.0, 0.0, 255.0));
        }
    }
    return out;
}

cv::Mat normalRgbTo8U(const cv::Mat& normalMap, const cv::Mat& mask) {
    cv::Mat out(normalMap.rows, normalMap.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int y = 0; y < normalMap.rows; ++y) {
        const cv::Vec3f* nrow = normalMap.ptr<cv::Vec3f>(y);
        const uchar* mrow = mask.ptr<uchar>(y);
        cv::Vec3b* orow = out.ptr<cv::Vec3b>(y);
        for (int x = 0; x < normalMap.cols; ++x) {
            if (mrow[x] == 0) {
                continue;
            }
            const cv::Vec3f n = nrow[x];
            const uchar r = static_cast<uchar>(std::clamp((n[0] * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
            const uchar g = static_cast<uchar>(std::clamp((n[1] * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
            const uchar b = static_cast<uchar>(std::clamp((n[2] * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
            orow[x] = cv::Vec3b(b, g, r);
        }
    }
    return out;
}

cv::Vec3f normalizeVector(const cv::Vec3f& v) {
    const float length = std::sqrt(v.dot(v));
    if (length <= 1.0e-6f) {
        return cv::Vec3f(0.0f, 0.0f, 1.0f);
    }
    return v / length;
}

float smoothstep(float edge0, float edge1, float x) {
    const float t = std::clamp((x - edge0) / std::max(1.0e-6f, edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

cv::Vec3f mixColor(const cv::Vec3f& a, const cv::Vec3f& b, float t) {
    return a * (1.0f - t) + b * t;
}

cv::Vec3f reflectVector(const cv::Vec3f& incident, const cv::Vec3f& normal) {
    return incident - 2.0f * incident.dot(normal) * normal;
}

float maskedAbsPercentile(const cv::Mat& src, const cv::Mat& mask, float percentileValue) {
    std::vector<float> values;
    values.reserve(static_cast<size_t>(src.rows * src.cols / 4));
    for (int y = 0; y < src.rows; ++y) {
        const float* row = src.ptr<float>(y);
        const uchar* mrow = mask.ptr<uchar>(y);
        for (int x = 0; x < src.cols; ++x) {
            if (mrow[x] != 0 && std::isfinite(row[x])) {
                values.push_back(std::abs(row[x]));
            }
        }
    }
    if (values.empty()) {
        return 1.0f;
    }
    const size_t index = static_cast<size_t>(
        std::clamp(percentileValue, 0.0f, 1.0f) * static_cast<float>(values.size() - 1));
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(index), values.end());
    return std::max(values[index], 1.0e-6f);
}

cv::Mat maskedGaussianBlur(const cv::Mat& src, const cv::Mat& mask, double sigma) {
    cv::Mat maskFloat;
    mask.convertTo(maskFloat, CV_32F, 1.0 / 255.0);

    cv::Mat weighted(src.size(), CV_32F, cv::Scalar(0));
    for (int y = 0; y < src.rows; ++y) {
        const float* srcRow = src.ptr<float>(y);
        const float* maskRow = maskFloat.ptr<float>(y);
        float* weightedRow = weighted.ptr<float>(y);
        for (int x = 0; x < src.cols; ++x) {
            weightedRow[x] = std::isfinite(srcRow[x]) ? srcRow[x] * maskRow[x] : 0.0f;
        }
    }

    cv::Mat numerator;
    cv::Mat denominator;
    cv::GaussianBlur(weighted, numerator, cv::Size(0, 0), sigma);
    cv::GaussianBlur(maskFloat, denominator, cv::Size(0, 0), sigma);

    cv::Mat out(src.size(), CV_32F, cv::Scalar(0));
    for (int y = 0; y < src.rows; ++y) {
        const float* numRow = numerator.ptr<float>(y);
        const float* denRow = denominator.ptr<float>(y);
        float* outRow = out.ptr<float>(y);
        for (int x = 0; x < src.cols; ++x) {
            outRow[x] = denRow[x] > 1.0e-6f ? numRow[x] / denRow[x] : 0.0f;
        }
    }
    return out;
}

cv::Mat maskedGaussianBlurVec3(const cv::Mat& src, const cv::Mat& mask, double sigma) {
    std::vector<cv::Mat> channels;
    cv::split(src, channels);
    for (cv::Mat& channel : channels) {
        channel = maskedGaussianBlur(channel, mask, sigma);
    }
    cv::Mat out;
    cv::merge(channels, out);
    return out;
}

cv::Mat liquidMetalTo8U(const cv::Mat& normalMap, const cv::Mat& mask) {
    cv::Mat smoothNormals = maskedGaussianBlurVec3(normalMap, mask, 0.45);
    const cv::Vec3f view(0.0f, 0.0f, 1.0f);
    const cv::Vec3f keyLight = normalizeVector(cv::Vec3f(-0.45f, 0.35f, 0.82f));
    const cv::Vec3f fillLight = normalizeVector(cv::Vec3f(0.58f, -0.42f, 0.70f));
    const cv::Vec3f darkSteel(0.035f, 0.045f, 0.050f);
    const cv::Vec3f silver(0.72f, 0.76f, 0.78f);
    const cv::Vec3f coolCyan(0.42f, 0.72f, 0.80f);
    const cv::Vec3f warmGold(0.95f, 0.76f, 0.42f);

    cv::Mat out(normalMap.rows, normalMap.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int y = 0; y < normalMap.rows; ++y) {
        const cv::Vec3f* rawRow = normalMap.ptr<cv::Vec3f>(y);
        const cv::Vec3f* nrow = smoothNormals.ptr<cv::Vec3f>(y);
        const uchar* mrow = mask.ptr<uchar>(y);
        cv::Vec3b* outRow = out.ptr<cv::Vec3b>(y);
        for (int x = 0; x < normalMap.cols; ++x) {
            if (mrow[x] == 0) {
                continue;
            }

            const cv::Vec3f n = normalizeVector(mixColor(rawRow[x], nrow[x], 0.18f));
            const cv::Vec3f reflectedView = normalizeVector(reflectVector(-view, n));
            const float ndv = std::clamp(n.dot(view), 0.0f, 1.0f);
            const float key = std::max(0.0f, n.dot(keyLight));
            const float fill = std::max(0.0f, n.dot(fillLight));
            const float stripe = smoothstep(0.32f, 0.86f, 0.5f + 0.5f * (reflectedView[0] * 0.70f + reflectedView[1] * 0.30f));

            cv::Vec3f base = mixColor(darkSteel, silver, stripe);
            base = mixColor(base, coolCyan, 0.20f * smoothstep(0.05f, 0.90f, reflectedView[1] * 0.5f + 0.5f));
            base = mixColor(base, warmGold, 0.16f * smoothstep(0.25f, 1.00f, reflectedView[0] * -0.5f + 0.5f));

            const cv::Vec3f keyReflection = reflectVector(-keyLight, n);
            const cv::Vec3f fillReflection = reflectVector(-fillLight, n);
            const float specKey = std::pow(std::max(0.0f, keyReflection.dot(view)), 60.0f);
            const float specFill = std::pow(std::max(0.0f, fillReflection.dot(view)), 32.0f);
            const float rim = std::pow(1.0f - ndv, 2.0f);

            cv::Vec3f rgb = base * (0.42f + 0.46f * key + 0.18f * fill);
            rgb += cv::Vec3f(1.0f, 0.96f, 0.88f) * (1.00f * specKey + 0.36f * specFill);
            rgb += coolCyan * (0.20f * rim);

            for (int c = 0; c < 3; ++c) {
                rgb[c] = 1.0f - std::exp(-std::max(0.0f, rgb[c]) * 1.35f);
                rgb[c] = std::pow(std::clamp(rgb[c], 0.0f, 1.0f), 1.0f / 2.2f);
            }
            outRow[x] = cv::Vec3b(
                static_cast<uchar>(std::clamp(rgb[2] * 255.0f, 0.0f, 255.0f)),
                static_cast<uchar>(std::clamp(rgb[1] * 255.0f, 0.0f, 255.0f)),
                static_cast<uchar>(std::clamp(rgb[0] * 255.0f, 0.0f, 255.0f)));
        }
    }
    return out;
}

void writePfm(const fs::path& path, const cv::Mat& image, const cv::Mat& mask) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        die("Failed to write PFM: " + path.string());
    }
    out << "Pf\n" << image.cols << " " << image.rows << "\n-1.0\n";
    for (int y = image.rows - 1; y >= 0; --y) {
        const float* row = image.ptr<float>(y);
        const uchar* mrow = mask.ptr<uchar>(y);
        for (int x = 0; x < image.cols; ++x) {
            float v = (mrow[x] != 0 && std::isfinite(row[x])) ? row[x] : 0.0f;
            out.write(reinterpret_cast<const char*>(&v), sizeof(float));
        }
    }
}

void writeLightsCsv(
    const fs::path& path,
    const Options& opt,
    const std::vector<cv::Vec3f>& lights,
    const std::vector<HighlightEstimate>& estimates) {
    std::ofstream out(path);
    if (!out) {
        die("Failed to write lights CSV: " + path.string());
    }
    if (opt.uncalibratedLighting) {
        out << "mode,uncalibrated_unknown_lighting\n";
        out << "note,No sphere or calibrated light vectors were used. Normals and height are relative and ambiguous.\n";
        out << "image\n";
        for (const std::string& imagePath : opt.imagePaths) {
            out << '"' << imagePath << '"' << '\n';
        }
        return;
    }
    out << "sphere_cx," << opt.sphere.cx << "\n";
    out << "sphere_cy," << opt.sphere.cy << "\n";
    out << "sphere_radius," << opt.sphere.radius << "\n";
    out << "image,highlight_x,highlight_y,light_x,light_y,light_z,threshold,peak,selected_pixels\n";
    out << std::fixed << std::setprecision(8);
    for (size_t i = 0; i < lights.size(); ++i) {
        out << '"' << opt.imagePaths[i] << '"';
        if (i < estimates.size()) {
            out << ',' << estimates[i].point.x << ',' << estimates[i].point.y;
        } else {
            out << ",,";
        }
        out << ',' << lights[i][0] << ',' << lights[i][1] << ',' << lights[i][2];
        if (i < estimates.size()) {
            out << ',' << estimates[i].threshold << ',' << estimates[i].peak << ',' << estimates[i].selectedPixels;
        } else {
            out << ",,,";
        }
        out << '\n';
    }
}

void writeLightVectorsCsv(const fs::path& path, const std::vector<cv::Vec3f>& lights) {
    std::ofstream out(path);
    if (!out) {
        die("Failed to write light vector CSV: " + path.string());
    }
    out << "x,y,z\n";
    out << std::fixed << std::setprecision(8);
    for (const cv::Vec3f& light : lights) {
        out << light[0] << ',' << light[1] << ',' << light[2] << '\n';
    }
}

void writePlyMesh(
    const fs::path& path,
    const cv::Mat& height,
    const cv::Mat& mask,
    int step,
    double heightScale,
    const std::function<void(const std::string&)>& progress) {
    const int rows = height.rows;
    const int cols = height.cols;
    cv::Mat index(rows, cols, CV_32S, cv::Scalar(-1));

    reportProgress(progress, "PLY: indexing vertices...");
    int vertexCount = 0;
    for (int y = 0; y < rows; y += step) {
        const float* hrow = height.ptr<float>(y);
        const uchar* mrow = mask.ptr<uchar>(y);
        int* irow = index.ptr<int>(y);
        for (int x = 0; x < cols; x += step) {
            if (mrow[x] != 0 && std::isfinite(hrow[x])) {
                irow[x] = vertexCount++;
            }
        }
    }

    reportProgress(progress, "PLY: counting faces...");
    int faceCount = 0;
    for (int y = 0; y + step < rows; y += step) {
        const int* row0 = index.ptr<int>(y);
        const int* row1 = index.ptr<int>(y + step);
        for (int x = 0; x + step < cols; x += step) {
            const int a = row0[x];
            const int b = row0[x + step];
            const int c = row1[x];
            const int d = row1[x + step];
            if (a >= 0 && b >= 0 && c >= 0) {
                ++faceCount;
            }
            if (b >= 0 && d >= 0 && c >= 0) {
                ++faceCount;
            }
        }
    }

    fs::create_directories(path.parent_path().empty() ? fs::path(".") : path.parent_path());
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        die("Failed to write PLY mesh: " + path.string());
    }

    out << "ply\n";
    out << "format binary_little_endian 1.0\n";
    out << "comment generated by What A Relief\n";
    out << "element vertex " << vertexCount << "\n";
    out << "property float x\n";
    out << "property float y\n";
    out << "property float z\n";
    out << "element face " << faceCount << "\n";
    out << "property list uchar int vertex_indices\n";
    out << "end_header\n";

    reportProgress(progress, "PLY: writing binary vertices...");
    for (int y = 0; y < rows; y += step) {
        const float* hrow = height.ptr<float>(y);
        const int* irow = index.ptr<int>(y);
        for (int x = 0; x < cols; x += step) {
            if (irow[x] >= 0) {
                const float vx = static_cast<float>(x);
                const float vy = static_cast<float>(-y);
                const float vz = static_cast<float>(static_cast<double>(hrow[x]) * heightScale);
                out.write(reinterpret_cast<const char*>(&vx), sizeof(vx));
                out.write(reinterpret_cast<const char*>(&vy), sizeof(vy));
                out.write(reinterpret_cast<const char*>(&vz), sizeof(vz));
            }
        }
    }

    reportProgress(progress, "PLY: writing binary faces...");
    for (int y = 0; y + step < rows; y += step) {
        const int* row0 = index.ptr<int>(y);
        const int* row1 = index.ptr<int>(y + step);
        for (int x = 0; x + step < cols; x += step) {
            const int a = row0[x];
            const int b = row0[x + step];
            const int c = row1[x];
            const int d = row1[x + step];
            if (a >= 0 && b >= 0 && c >= 0) {
                const std::uint8_t count = 3;
                const std::int32_t ia = static_cast<std::int32_t>(a);
                const std::int32_t ib = static_cast<std::int32_t>(c);
                const std::int32_t ic = static_cast<std::int32_t>(b);
                out.write(reinterpret_cast<const char*>(&count), sizeof(count));
                out.write(reinterpret_cast<const char*>(&ia), sizeof(ia));
                out.write(reinterpret_cast<const char*>(&ib), sizeof(ib));
                out.write(reinterpret_cast<const char*>(&ic), sizeof(ic));
            }
            if (b >= 0 && d >= 0 && c >= 0) {
                const std::uint8_t count = 3;
                const std::int32_t ia = static_cast<std::int32_t>(b);
                const std::int32_t ib = static_cast<std::int32_t>(c);
                const std::int32_t ic = static_cast<std::int32_t>(d);
                out.write(reinterpret_cast<const char*>(&count), sizeof(count));
                out.write(reinterpret_cast<const char*>(&ia), sizeof(ia));
                out.write(reinterpret_cast<const char*>(&ib), sizeof(ib));
                out.write(reinterpret_cast<const char*>(&ic), sizeof(ic));
            }
        }
    }

    if (!out) {
        die("Failed while writing PLY mesh: " + path.string());
    }
    reportProgress(
        progress,
        "PLY: done (" + std::to_string(vertexCount) + " vertices, " +
            std::to_string(faceCount) + " faces).");
}

} // namespace

std::vector<cv::Mat> loadLuminanceImages(const std::vector<std::string>& paths, bool srgb) {
    std::vector<cv::Mat> images;
    images.reserve(paths.size());
    cv::Size expected;
    for (const std::string& path : paths) {
        cv::Mat raw = cv::imread(path, cv::IMREAD_UNCHANGED);
        if (raw.empty()) {
            die("Failed to read image: " + path);
        }
        cv::Mat gray = toFloatLuminance(raw, srgb);
        if (expected.empty()) {
            expected = gray.size();
        } else if (gray.size() != expected) {
            die("All input images must have identical dimensions. Mismatch at: " + path);
        }
        images.push_back(gray);
    }
    return images;
}

cv::Mat loadDisplayImage(const std::string& path) {
    cv::Mat raw = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (raw.empty()) {
        die("Failed to read image: " + path);
    }
    cv::Mat normalized;
    if (raw.depth() == CV_8U) {
        normalized = raw;
    } else {
        cv::normalize(raw, normalized, 0, 255, cv::NORM_MINMAX, CV_MAKETYPE(CV_8U, raw.channels()));
    }
    if (normalized.channels() == 1) {
        cv::Mat bgr;
        cv::cvtColor(normalized, bgr, cv::COLOR_GRAY2BGR);
        return bgr;
    }
    if (normalized.channels() == 4) {
        cv::Mat bgr;
        cv::cvtColor(normalized, bgr, cv::COLOR_BGRA2BGR);
        return bgr;
    }
    return normalized;
}

cv::Mat loadMask(const std::string& path, const cv::Size& size) {
    if (path.empty()) {
        return cv::Mat(size, CV_8U, cv::Scalar(255));
    }
    cv::Mat raw = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (raw.empty()) {
        die("Failed to read mask: " + path);
    }
    if (raw.size() != size) {
        die("Mask dimensions must match input images.");
    }
    cv::Mat mask;
    cv::threshold(raw, mask, 0, 255, cv::THRESH_BINARY);
    return mask;
}

void applyCropToMask(cv::Mat& mask, const cv::Rect& crop) {
    const cv::Rect imageRect(0, 0, mask.cols, mask.rows);
    const cv::Rect clamped = crop & imageRect;
    if (clamped.empty()) {
        die("Crop rectangle does not overlap the image.");
    }

    cv::Mat cropMask(mask.size(), CV_8U, cv::Scalar(0));
    cropMask(clamped).setTo(cv::Scalar(255));
    cv::bitwise_and(mask, cropMask, mask);
}

void removeSphereFromMask(cv::Mat& mask, const Sphere& sphere) {
    const int x0 = std::max(0, static_cast<int>(std::floor(sphere.cx - sphere.radius - 2.0)));
    const int y0 = std::max(0, static_cast<int>(std::floor(sphere.cy - sphere.radius - 2.0)));
    const int x1 = std::min(mask.cols - 1, static_cast<int>(std::ceil(sphere.cx + sphere.radius + 2.0)));
    const int y1 = std::min(mask.rows - 1, static_cast<int>(std::ceil(sphere.cy + sphere.radius + 2.0)));
    for (int y = y0; y <= y1; ++y) {
        uchar* row = mask.ptr<uchar>(y);
        for (int x = x0; x <= x1; ++x) {
            if (insideSphere(sphere, x, y, 2.0)) {
                row[x] = 0;
            }
        }
    }
}

void saveOutputs(
    const Options& opt,
    const std::vector<cv::Vec3f>& lights,
    const std::vector<HighlightEstimate>& estimates,
    const cv::Mat& normalMap,
    const cv::Mat& albedo,
    const cv::Mat& residual,
    const cv::Mat& validMask,
    const cv::Mat& height,
    const std::function<void(const std::string&)>& progress) {
    const fs::path outDir = opt.outputDir;
    fs::create_directories(outDir);
    reportProgress(progress, "Writing light CSV files...");
    writeLightsCsv(outDir / "lights.csv", opt, lights, estimates);
    writeLightVectorsCsv(outDir / "light_vectors.csv", lights);
    reportProgress(progress, "Writing image outputs...");
    cv::imwrite((outDir / "normal_rgb.png").string(), normalRgbTo8U(normalMap, validMask));
    cv::imwrite((outDir / "normal_x.png").string(), normalComponentTo8U(normalMap, validMask, 0, true));
    cv::imwrite((outDir / "normal_y.png").string(), normalComponentTo8U(normalMap, validMask, 1, true));
    cv::imwrite((outDir / "normal_z.png").string(), normalComponentTo8U(normalMap, validMask, 2, false));
    cv::imwrite((outDir / "albedo.png").string(), normalizeFloatTo8U(albedo, validMask, true));
    cv::imwrite((outDir / "residual.png").string(), normalizeFloatTo8U(residual, validMask, true));
    cv::imwrite((outDir / "valid_mask.png").string(), validMask);
    cv::imwrite((outDir / "liquid_metal.png").string(), liquidMetalTo8U(normalMap, validMask));
    if (!height.empty()) {
        reportProgress(progress, "Writing height outputs...");
        cv::imwrite((outDir / "height.png").string(), normalizeFloatTo8U(height, validMask, false));
        writePfm(outDir / "height.pfm", height, validMask);
        if (!opt.meshPath.empty()) {
            writePlyMesh(opt.meshPath, height, validMask, opt.meshStep, opt.heightScale, progress);
        }
    }
}
