#include "image_io.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>

namespace fs = std::filesystem;

namespace {

[[noreturn]] void die(const std::string& message) {
    throw std::runtime_error(message);
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
    const cv::Mat& height) {
    const fs::path outDir = opt.outputDir;
    fs::create_directories(outDir);
    writeLightsCsv(outDir / "lights.csv", opt, lights, estimates);
    writeLightVectorsCsv(outDir / "light_vectors.csv", lights);
    cv::imwrite((outDir / "normal_rgb.png").string(), normalRgbTo8U(normalMap, validMask));
    cv::imwrite((outDir / "normal_x.png").string(), normalComponentTo8U(normalMap, validMask, 0, true));
    cv::imwrite((outDir / "normal_y.png").string(), normalComponentTo8U(normalMap, validMask, 1, true));
    cv::imwrite((outDir / "normal_z.png").string(), normalComponentTo8U(normalMap, validMask, 2, false));
    cv::imwrite((outDir / "albedo.png").string(), normalizeFloatTo8U(albedo, validMask, true));
    cv::imwrite((outDir / "residual.png").string(), normalizeFloatTo8U(residual, validMask, true));
    cv::imwrite((outDir / "valid_mask.png").string(), validMask);
    cv::imwrite((outDir / "height.png").string(), normalizeFloatTo8U(height, validMask, false));
    writePfm(outDir / "height.pfm", height, validMask);
}
