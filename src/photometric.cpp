#include "photometric.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <array>
#include <functional>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

[[noreturn]] void die(const std::string& message) {
    throw std::runtime_error(message);
}

bool insideSphere(const Sphere& sphere, int x, int y) {
    const double dx = static_cast<double>(x) - sphere.cx;
    const double dy = static_cast<double>(y) - sphere.cy;
    return dx * dx + dy * dy <= sphere.radius * sphere.radius;
}

float percentile(std::vector<float>& values, double pct) {
    if (values.empty()) {
        die("Cannot compute percentile of an empty value set.");
    }
    const double rank = (pct / 100.0) * static_cast<double>(values.size() - 1);
    const size_t index = static_cast<size_t>(std::clamp(rank, 0.0, static_cast<double>(values.size() - 1)));
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(index), values.end());
    return values[index];
}

bool solveSymmetric3x3(
    double a00,
    double a01,
    double a02,
    double a11,
    double a12,
    double a22,
    const cv::Vec3d& b,
    cv::Vec3d& x) {
    const double c00 = a11 * a22 - a12 * a12;
    const double c01 = a02 * a12 - a01 * a22;
    const double c02 = a01 * a12 - a02 * a11;
    const double c11 = a00 * a22 - a02 * a02;
    const double c12 = a01 * a02 - a00 * a12;
    const double c22 = a00 * a11 - a01 * a01;
    const double det = a00 * c00 + a01 * c01 + a02 * c02;
    if (!std::isfinite(det) || std::abs(det) < 1.0e-8) {
        return false;
    }

    const double invDet = 1.0 / det;
    x[0] = (c00 * b[0] + c01 * b[1] + c02 * b[2]) * invDet;
    x[1] = (c01 * b[0] + c11 * b[1] + c12 * b[2]) * invDet;
    x[2] = (c02 * b[0] + c12 * b[1] + c22 * b[2]) * invDet;
    return std::isfinite(x[0]) && std::isfinite(x[1]) && std::isfinite(x[2]);
}

void reportProgress(
    const std::function<void(const std::string&)>& progress,
    const std::string& message) {
    if (progress) {
        progress(message);
    }
}

std::string trimCopy(const std::string& s) {
    const size_t first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const size_t last = s.find_last_not_of(" \t\r\n");
    std::string out = s.substr(first, last - first + 1);
    if (out.size() >= 2 && out.front() == '"' && out.back() == '"') {
        out = out.substr(1, out.size() - 2);
    }
    return out;
}

std::vector<std::string> splitCsvLine(const std::string& line) {
    std::vector<std::string> fields;
    std::string current;
    bool quoted = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '"') {
            quoted = !quoted;
            current.push_back(c);
        } else if (c == ',' && !quoted) {
            fields.push_back(trimCopy(current));
            current.clear();
        } else {
            current.push_back(c);
        }
    }
    fields.push_back(trimCopy(current));
    return fields;
}

std::vector<std::string> splitLightLine(const std::string& line) {
    if (line.find(',') != std::string::npos) {
        return splitCsvLine(line);
    }
    std::istringstream iss(line);
    std::vector<std::string> fields;
    std::string field;
    while (iss >> field) {
        fields.push_back(field);
    }
    return fields;
}

bool parseDoubleField(const std::string& s, double& value) {
    try {
        size_t end = 0;
        value = std::stod(s, &end);
        return end == s.size();
    } catch (const std::exception&) {
        return false;
    }
}

bool parseVec3Fields(const std::vector<std::string>& fields, size_t first, cv::Vec3f& v) {
    if (fields.size() < first + 3) {
        return false;
    }
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    if (!parseDoubleField(fields[first], x) ||
        !parseDoubleField(fields[first + 1], y) ||
        !parseDoubleField(fields[first + 2], z)) {
        return false;
    }
    v = cv::Vec3f(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
    return true;
}

cv::Mat maskedGaussianBlurFloat(const cv::Mat& src, const cv::Mat& mask, double sigma) {
    cv::Mat weighted = cv::Mat::zeros(src.size(), CV_32F);
    cv::Mat weights = cv::Mat::zeros(src.size(), CV_32F);
    for (int y = 0; y < src.rows; ++y) {
        const float* srow = src.ptr<float>(y);
        const uchar* mrow = mask.ptr<uchar>(y);
        float* wrow = weighted.ptr<float>(y);
        float* weightRow = weights.ptr<float>(y);
        for (int x = 0; x < src.cols; ++x) {
            if (mrow[x] == 0 || !std::isfinite(srow[x])) {
                continue;
            }
            wrow[x] = srow[x];
            weightRow[x] = 1.0f;
        }
    }

    cv::GaussianBlur(weighted, weighted, cv::Size(), sigma, sigma, cv::BORDER_REPLICATE);
    cv::GaussianBlur(weights, weights, cv::Size(), sigma, sigma, cv::BORDER_REPLICATE);
    cv::Mat result = cv::Mat::zeros(src.size(), CV_32F);
    for (int y = 0; y < src.rows; ++y) {
        const float* wrow = weighted.ptr<float>(y);
        const float* weightRow = weights.ptr<float>(y);
        float* rrow = result.ptr<float>(y);
        for (int x = 0; x < src.cols; ++x) {
            if (weightRow[x] > 1.0e-6f) {
                rrow[x] = wrow[x] / weightRow[x];
            }
        }
    }
    return result;
}

double medianValue(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    const size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid), values.end());
    return values[mid];
}

cv::Vec3d normalizeVec3d(const cv::Vec3d& v) {
    const double length = std::sqrt(v.dot(v));
    if (length <= 1.0e-12 || !std::isfinite(length)) {
        return cv::Vec3d(0.0, 0.0, 1.0);
    }
    return v / length;
}

cv::Matx33d rotationBetween(const cv::Vec3d& fromInput, const cv::Vec3d& toInput) {
    const cv::Vec3d from = normalizeVec3d(fromInput);
    const cv::Vec3d to = normalizeVec3d(toInput);
    const cv::Vec3d v = from.cross(to);
    const double c = std::clamp(from.dot(to), -1.0, 1.0);
    const double s = std::sqrt(v.dot(v));

    if (s <= 1.0e-10) {
        if (c > 0.0) {
            return cv::Matx33d::eye();
        }
        cv::Vec3d axis = std::abs(from[0]) < 0.9 ? cv::Vec3d(1.0, 0.0, 0.0) : cv::Vec3d(0.0, 1.0, 0.0);
        axis = normalizeVec3d(from.cross(axis));
        const double x = axis[0];
        const double y = axis[1];
        const double z = axis[2];
        return cv::Matx33d(
            2.0 * x * x - 1.0, 2.0 * x * y, 2.0 * x * z,
            2.0 * y * x, 2.0 * y * y - 1.0, 2.0 * y * z,
            2.0 * z * x, 2.0 * z * y, 2.0 * z * z - 1.0);
    }

    const cv::Matx33d k(
        0.0, -v[2], v[1],
        v[2], 0.0, -v[0],
        -v[1], v[0], 0.0);
    return cv::Matx33d::eye() + k + (k * k) * ((1.0 - c) / (s * s));
}

cv::Matx33d rotationAroundZ(double radians) {
    const double c = std::cos(radians);
    const double s = std::sin(radians);
    return cv::Matx33d(
        c, -s, 0.0,
        s, c, 0.0,
        0.0, 0.0, 1.0);
}

cv::Vec3f nearFieldRingLightDirection(
    const cv::Vec3f& reference,
    int index,
    int count,
    int x,
    int y,
    double radiusMm,
    double heightMm,
    double pixelScaleMm,
    const cv::Point2d& center) {
    double ax = reference[0];
    double ay = reference[1];
    const double xy = std::sqrt(ax * ax + ay * ay);
    if (xy > 1.0e-8) {
        ax /= xy;
        ay /= xy;
    } else {
        const double theta = 2.0 * CV_PI * static_cast<double>(index) / static_cast<double>(std::max(1, count));
        ax = std::cos(theta);
        ay = std::sin(theta);
    }

    const cv::Vec3d lightPosition(radiusMm * ax, radiusMm * ay, heightMm);
    const cv::Vec3d surfacePoint(
        (static_cast<double>(x) - center.x) * pixelScaleMm,
        (center.y - static_cast<double>(y)) * pixelScaleMm,
        0.0);
    const cv::Vec3d direction = normalizeVec3d(lightPosition - surfacePoint);
    return cv::Vec3f(
        static_cast<float>(direction[0]),
        static_cast<float>(direction[1]),
        static_cast<float>(direction[2]));
}

double curlCostForNormals(const cv::Mat& normalMap, const cv::Mat& validMask, const cv::Matx33d& rotation) {
    const int rows = normalMap.rows;
    const int cols = normalMap.cols;
    cv::Mat p(rows, cols, CV_32F, cv::Scalar(0));
    cv::Mat q(rows, cols, CV_32F, cv::Scalar(0));

    for (int y = 0; y < rows; ++y) {
        const cv::Vec3f* nrow = normalMap.ptr<cv::Vec3f>(y);
        const uchar* mrow = validMask.ptr<uchar>(y);
        float* prow = p.ptr<float>(y);
        float* qrow = q.ptr<float>(y);
        for (int x = 0; x < cols; ++x) {
            if (mrow[x] == 0) {
                continue;
            }
            const cv::Vec3d r = rotation * cv::Vec3d(nrow[x][0], nrow[x][1], nrow[x][2]);
            if (r[2] <= 1.0e-4) {
                continue;
            }
            prow[x] = static_cast<float>(-r[0] / r[2]);
            qrow[x] = static_cast<float>(r[1] / r[2]);
        }
    }

    double cost = 0.0;
    int count = 0;
    for (int y = 1; y < rows; ++y) {
        const uchar* mrow = validMask.ptr<uchar>(y);
        const uchar* prevMrow = validMask.ptr<uchar>(y - 1);
        const float* prow = p.ptr<float>(y);
        const float* prevProw = p.ptr<float>(y - 1);
        const float* qrow = q.ptr<float>(y);
        for (int x = 1; x < cols; ++x) {
            if (mrow[x] == 0 || mrow[x - 1] == 0 || prevMrow[x] == 0) {
                continue;
            }
            const double dpdy = static_cast<double>(prow[x] - prevProw[x]);
            const double dqdx = static_cast<double>(qrow[x] - qrow[x - 1]);
            const double curl = dpdy - dqdx;
            cost += curl * curl;
            ++count;
        }
    }
    if (count == 0) {
        return std::numeric_limits<double>::infinity();
    }
    return cost / static_cast<double>(count);
}

void applyNormalRotation(cv::Mat& normalMap, cv::Mat& validMask, const cv::Matx33d& rotation) {
    for (int y = 0; y < normalMap.rows; ++y) {
        cv::Vec3f* nrow = normalMap.ptr<cv::Vec3f>(y);
        uchar* mrow = validMask.ptr<uchar>(y);
        for (int x = 0; x < normalMap.cols; ++x) {
            if (mrow[x] == 0) {
                continue;
            }
            cv::Vec3d r = normalizeVec3d(rotation * cv::Vec3d(nrow[x][0], nrow[x][1], nrow[x][2]));
            if (r[2] < 0.0) {
                r = -r;
            }
            nrow[x] = cv::Vec3f(static_cast<float>(r[0]), static_cast<float>(r[1]), static_cast<float>(r[2]));
        }
    }
}

float finitePercentile(std::vector<float>& values, float percentileValue, float fallback) {
    if (values.empty()) {
        return fallback;
    }
    const size_t index = static_cast<size_t>(
        std::clamp(percentileValue, 0.0f, 1.0f) * static_cast<float>(values.size() - 1));
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(index), values.end());
    return std::isfinite(values[index]) ? values[index] : fallback;
}

void stabilizeVisualReliefNormals(
    cv::Mat& normalMap,
    const cv::Mat& validMask,
    const std::function<void(const std::string&)>& progress) {
    std::vector<float> absP;
    std::vector<float> absQ;
    std::vector<float> slopeMag;
    absP.reserve(static_cast<size_t>(normalMap.rows * normalMap.cols / 2));
    absQ.reserve(absP.capacity());
    slopeMag.reserve(absP.capacity());

    constexpr float kMinNz = 0.20f;
    for (int y = 0; y < normalMap.rows; ++y) {
        const cv::Vec3f* nrow = normalMap.ptr<cv::Vec3f>(y);
        const uchar* mrow = validMask.ptr<uchar>(y);
        for (int x = 0; x < normalMap.cols; ++x) {
            if (mrow[x] == 0) {
                continue;
            }
            const cv::Vec3f n = nrow[x];
            const float nz = std::max(std::abs(n[2]), kMinNz);
            const float p = -n[0] / nz;
            const float q = n[1] / nz;
            if (std::isfinite(p) && std::isfinite(q)) {
                absP.push_back(std::abs(p));
                absQ.push_back(std::abs(q));
                slopeMag.push_back(std::sqrt(p * p + q * q));
            }
        }
    }

    const float p95 = finitePercentile(absP, 0.95f, 1.0f);
    const float q95 = finitePercentile(absQ, 0.95f, 1.0f);
    const float mag98 = finitePercentile(slopeMag, 0.98f, 1.0f);
    const float pScale = std::min(1.0f, 0.70f / std::max(p95, 1.0e-6f));
    const float qScale = std::min(1.0f, 0.70f / std::max(q95, 1.0e-6f));
    const float magScale = std::min(1.0f, 1.10f / std::max(mag98, 1.0e-6f));
    const float pTotalScale = pScale * magScale;
    const float qTotalScale = qScale * magScale;

    for (int y = 0; y < normalMap.rows; ++y) {
        cv::Vec3f* nrow = normalMap.ptr<cv::Vec3f>(y);
        const uchar* mrow = validMask.ptr<uchar>(y);
        for (int x = 0; x < normalMap.cols; ++x) {
            if (mrow[x] == 0) {
                continue;
            }
            const cv::Vec3f n = nrow[x];
            const float nz = std::max(std::abs(n[2]), kMinNz);
            float p = (-n[0] / nz) * pTotalScale;
            float q = (n[1] / nz) * qTotalScale;
            const float mag = std::sqrt(p * p + q * q);
            if (mag > 1.50f) {
                const float s = 1.50f / mag;
                p *= s;
                q *= s;
            }
            const float invLength = 1.0f / std::sqrt(p * p + q * q + 1.0f);
            nrow[x] = cv::Vec3f(-p * invLength, q * invLength, invLength);
        }
    }

    reportProgress(
        progress,
        "Unknown lighting: stabilized visual relief slopes (p95=" +
            std::to_string(p95) + ", q95=" + std::to_string(q95) + ").");
}

void solveUncalibratedPcaFallback(
    const std::vector<cv::Mat>& images,
    const cv::Mat& inputMask,
    float shadowThreshold,
    const cv::Mat& eigenVectors,
    cv::Mat& normalMap,
    cv::Mat& albedo,
    cv::Mat& residual,
    cv::Mat& validMask,
    const std::function<void(const std::string&)>& progress) {
    reportProgress(progress, "Unknown lighting: metric was degenerate; using rank-3 PCA relief fallback...");
    const int rows = images[0].rows;
    const int cols = images[0].cols;
    const int n = static_cast<int>(images.size());

    normalMap = cv::Mat(rows, cols, CV_32FC3, cv::Scalar(0, 0, 0));
    albedo = cv::Mat(rows, cols, CV_32F, cv::Scalar(0));
    residual = cv::Mat(rows, cols, CV_32F, cv::Scalar(0));
    validMask = cv::Mat(rows, cols, CV_8U, cv::Scalar(0));

    cv::Vec3d averageNormal(0.0, 0.0, 0.0);
    std::vector<const float*> imageRows;
    imageRows.reserve(images.size());
    for (int y = 0; y < rows; ++y) {
        const uchar* maskRow = inputMask.ptr<uchar>(y);
        cv::Vec3f* normalRow = normalMap.ptr<cv::Vec3f>(y);
        float* albedoRow = albedo.ptr<float>(y);
        float* residualRow = residual.ptr<float>(y);
        uchar* validRow = validMask.ptr<uchar>(y);
        imageRows.clear();
        for (const cv::Mat& image : images) {
            imageRows.push_back(image.ptr<float>(y));
        }
        for (int x = 0; x < cols; ++x) {
            if (maskRow[x] == 0) {
                continue;
            }

            cv::Vec3d b(0.0, 0.0, 0.0);
            bool usable = false;
            for (int r = 0; r < 3; ++r) {
                const double* urow = eigenVectors.ptr<double>(r);
                for (int i = 0; i < n; ++i) {
                    const float intensity = imageRows[i][x];
                    const double vi = std::isfinite(intensity) ? std::max(0.0f, intensity) : 0.0f;
                    b[r] += urow[i] * vi;
                    usable = usable || vi > shadowThreshold;
                }
            }
            const double rho = std::sqrt(b.dot(b));
            if (!usable || !std::isfinite(rho) || rho <= 1.0e-8) {
                continue;
            }

            double fit = 0.0;
            for (int i = 0; i < n; ++i) {
                const double* u0 = eigenVectors.ptr<double>(0);
                const double* u1 = eigenVectors.ptr<double>(1);
                const double* u2 = eigenVectors.ptr<double>(2);
                const double predicted = u0[i] * b[0] + u1[i] * b[1] + u2[i] * b[2];
                const float intensity = imageRows[i][x];
                const double observed = std::isfinite(intensity) ? std::max(0.0f, intensity) : 0.0;
                const double d = predicted - observed;
                fit += d * d;
            }

            const cv::Vec3d nrm = b / rho;
            normalRow[x] = cv::Vec3f(static_cast<float>(nrm[0]), static_cast<float>(nrm[1]), static_cast<float>(nrm[2]));
            albedoRow[x] = static_cast<float>(rho);
            residualRow[x] = static_cast<float>(std::sqrt(fit / static_cast<double>(n)));
            validRow[x] = 255;
            averageNormal += nrm * rho;
        }
    }

    const cv::Matx33d faceCamera = rotationBetween(averageNormal, cv::Vec3d(0.0, 0.0, 1.0));
    applyNormalRotation(normalMap, validMask, faceCamera);

    double bestTheta = 0.0;
    double bestCost = std::numeric_limits<double>::infinity();
    constexpr int kSearchSteps = 72;
    for (int i = 0; i < kSearchSteps; ++i) {
        const double theta = (CV_PI * static_cast<double>(i)) / static_cast<double>(kSearchSteps);
        const double cost = curlCostForNormals(normalMap, validMask, rotationAroundZ(theta));
        if (cost < bestCost) {
            bestCost = cost;
            bestTheta = theta;
        }
    }
    applyNormalRotation(normalMap, validMask, rotationAroundZ(bestTheta));
    stabilizeVisualReliefNormals(normalMap, validMask, progress);
    reportProgress(progress, "Unknown lighting: done.");
}

} // namespace

HighlightEstimate estimateHighlight(const cv::Mat& image, const Sphere& sphere, const Options& opt) {
    const int x0 = std::max(0, static_cast<int>(std::floor(sphere.cx - sphere.radius)));
    const int y0 = std::max(0, static_cast<int>(std::floor(sphere.cy - sphere.radius)));
    const int x1 = std::min(image.cols - 1, static_cast<int>(std::ceil(sphere.cx + sphere.radius)));
    const int y1 = std::min(image.rows - 1, static_cast<int>(std::ceil(sphere.cy + sphere.radius)));

    std::vector<float> samples;
    samples.reserve(static_cast<size_t>((x1 - x0 + 1) * (y1 - y0 + 1)));
    float peak = 0.0f;
    for (int y = y0; y <= y1; ++y) {
        const float* row = image.ptr<float>(y);
        for (int x = x0; x <= x1; ++x) {
            if (!insideSphere(sphere, x, y)) {
                continue;
            }
            const float value = row[x];
            if (std::isfinite(value)) {
                samples.push_back(value);
                peak = std::max(peak, value);
            }
        }
    }
    if (samples.empty()) {
        die("Sphere ROI did not overlap any valid image pixels.");
    }

    const float threshold = percentile(samples, opt.highlightPercentile);
    const float minValue = std::max(threshold, static_cast<float>(opt.minHighlight));
    double sumW = 0.0;
    double sumX = 0.0;
    double sumY = 0.0;
    int selected = 0;

    for (int y = y0; y <= y1; ++y) {
        const float* row = image.ptr<float>(y);
        for (int x = x0; x <= x1; ++x) {
            if (!insideSphere(sphere, x, y)) {
                continue;
            }
            const float value = row[x];
            if (!std::isfinite(value) || value < minValue) {
                continue;
            }
            const double w = static_cast<double>(value - minValue) + 1.0e-6;
            sumW += w;
            sumX += w * static_cast<double>(x);
            sumY += w * static_cast<double>(y);
            ++selected;
        }
    }

    if (sumW <= 0.0 || selected == 0) {
        die("Could not find a bright highlight inside the selected sphere.");
    }

    const double hx = sumX / sumW;
    const double hy = sumY / sumW;
    double nx = (hx - sphere.cx) / sphere.radius;
    double ny = -(hy - sphere.cy) / sphere.radius;
    const double rr = nx * nx + ny * ny;
    if (rr >= 1.0) {
        const double s = 0.999 / std::sqrt(rr);
        nx *= s;
        ny *= s;
    }
    const double nz = std::sqrt(std::max(0.0, 1.0 - nx * nx - ny * ny));
    cv::Vec3f normal(static_cast<float>(nx), static_cast<float>(ny), static_cast<float>(nz));

    const float ndotv = normal.dot(opt.viewDir);
    if (ndotv <= 0.0f) {
        die("Highlight normal points away from the camera; check the selected sphere.");
    }
    cv::Vec3f light = 2.0f * ndotv * normal - opt.viewDir;
    const float lightNorm = std::sqrt(light.dot(light));
    if (lightNorm <= 0.0f) {
        die("Computed a zero light vector from the highlight sphere.");
    }
    light /= lightNorm;

    HighlightEstimate estimate;
    estimate.point = cv::Point2f(static_cast<float>(hx), static_cast<float>(hy));
    estimate.light = light;
    estimate.threshold = threshold;
    estimate.peak = peak;
    estimate.selectedPixels = selected;
    return estimate;
}

bool loadLightsFileMetadata(const std::string& path, Options& opt) {
    std::ifstream in(path);
    if (!in) {
        die("Failed to open lights file: " + path);
    }

    bool found = false;
    std::string line;
    while (std::getline(in, line)) {
        const size_t comment = line.find('#');
        if (comment != std::string::npos) {
            line = line.substr(0, comment);
        }
        line = trimCopy(line);
        if (line.empty()) {
            continue;
        }

        const std::vector<std::string> fields = splitLightLine(line);
        if (fields.empty()) {
            continue;
        }
        if (fields.size() >= 2) {
            if (fields[0] == "lighting_model") {
                if (fields[1] == "near_field_ring") {
                    opt.lightingModel = LightingModel::NearFieldRing;
                    found = true;
                } else if (fields[1] == "directional") {
                    opt.lightingModel = LightingModel::Directional;
                    found = true;
                }
                continue;
            }
            double value = 0.0;
            if (parseDoubleField(fields[1], value)) {
                if (fields[0] == "ring_light_radius_mm") {
                    opt.ringLightRadiusMm = value;
                    found = true;
                    continue;
                }
                if (fields[0] == "ring_light_height_mm") {
                    opt.ringLightHeightMm = value;
                    found = true;
                    continue;
                }
                if (fields[0] == "pixel_scale_mm_per_pixel") {
                    opt.pixelScaleMm = value;
                    found = true;
                    continue;
                }
            }
        }
    }
    return found;
}

std::vector<cv::Vec3f> loadLightsFile(const std::string& path, size_t expectedCount, Options* opt) {
    if (opt != nullptr) {
        loadLightsFileMetadata(path, *opt);
    }

    std::ifstream in(path);
    if (!in) {
        die("Failed to open lights file: " + path);
    }

    std::vector<cv::Vec3f> lights;
    std::string line;
    int lineNumber = 0;
    while (std::getline(in, line)) {
        ++lineNumber;
        const size_t comment = line.find('#');
        if (comment != std::string::npos) {
            line = line.substr(0, comment);
        }
        line = trimCopy(line);
        if (line.empty()) {
            continue;
        }

        const std::vector<std::string> fields = splitLightLine(line);
        cv::Vec3f v(0.0f, 0.0f, 0.0f);
        if (!parseVec3Fields(fields, 0, v) && !parseVec3Fields(fields, 3, v)) {
            continue;
        }
        const float norm = std::sqrt(v.dot(v));
        if (norm <= 0.0f) {
            die("Zero light vector in lights file at line " + std::to_string(lineNumber));
        }
        lights.push_back(v / norm);
    }
    if (lights.size() != expectedCount) {
        die("Lights file must contain one x,y,z vector per image.");
    }
    return lights;
}

void solvePhotometricStereo(
    const std::vector<cv::Mat>& images,
    const std::vector<cv::Vec3f>& lights,
    const cv::Mat& inputMask,
    float shadowThreshold,
    NormalSolverMode solverMode,
    float highOutlierThreshold,
    LightingModel lightingModel,
    double ringLightRadiusMm,
    double ringLightHeightMm,
    double pixelScaleMm,
    cv::Point2d lightingCenter,
    cv::Mat& normalMap,
    cv::Mat& albedo,
    cv::Mat& residual,
    cv::Mat& validMask,
    PhotometricDiagnostics& diagnostics) {
    const int rows = images[0].rows;
    const int cols = images[0].cols;
    const int n = static_cast<int>(images.size());
    if (n < 3) {
        die("Photometric stereo requires at least 3 images.");
    }
    if (n > 25) {
        die("Photometric stereo supports at most 25 images.");
    }
    if (lights.size() != images.size()) {
        die("Light vector count must match image count.");
    }

    normalMap = cv::Mat(rows, cols, CV_32FC3, cv::Scalar(0, 0, 0));
    albedo = cv::Mat(rows, cols, CV_32F, cv::Scalar(0));
    residual = cv::Mat(rows, cols, CV_32F, cv::Scalar(0));
    validMask = cv::Mat(rows, cols, CV_8U, cv::Scalar(0));
    diagnostics.robustWeight = cv::Mat(rows, cols, CV_32F, cv::Scalar(0));
    diagnostics.shadowCount = cv::Mat(rows, cols, CV_32F, cv::Scalar(0));
    diagnostics.highlightOutlierCount = cv::Mat(rows, cols, CV_32F, cv::Scalar(0));
    diagnostics.specularCueMask = cv::Mat(rows, cols, CV_8U, cv::Scalar(0));

    for (int y = 0; y < rows; ++y) {
        const uchar* maskRow = inputMask.ptr<uchar>(y);
        cv::Vec3f* normalRow = normalMap.ptr<cv::Vec3f>(y);
        float* albedoRow = albedo.ptr<float>(y);
        float* residualRow = residual.ptr<float>(y);
        uchar* validRow = validMask.ptr<uchar>(y);
        float* weightRow = diagnostics.robustWeight.ptr<float>(y);
        float* shadowRow = diagnostics.shadowCount.ptr<float>(y);
        float* highRow = diagnostics.highlightOutlierCount.ptr<float>(y);
        uchar* specularRow = diagnostics.specularCueMask.ptr<uchar>(y);
        std::vector<const float*> imageRows;
        imageRows.reserve(images.size());
        for (const cv::Mat& image : images) {
            imageRows.push_back(image.ptr<float>(y));
        }

        for (int x = 0; x < cols; ++x) {
            if (maskRow[x] == 0) {
                continue;
            }

            std::vector<int> observations;
            std::vector<float> observationIntensities;
            observations.reserve(n);
            observationIntensities.reserve(n);
            int shadowObservations = 0;
            for (int i = 0; i < n; ++i) {
                const float intensity = imageRows[i][x];
                if (!std::isfinite(intensity) || intensity <= shadowThreshold) {
                    ++shadowObservations;
                    continue;
                }
                observations.push_back(i);
                observationIntensities.push_back(intensity);
            }
            shadowRow[x] = static_cast<float>(shadowObservations);

            if (observations.size() < 3) {
                continue;
            }

            const bool robust = solverMode == NormalSolverMode::Robust && observations.size() >= 5;
            double adaptiveHighThreshold = highOutlierThreshold;
            if (robust) {
                std::vector<double> values;
                values.reserve(observationIntensities.size());
                for (const float value : observationIntensities) {
                    values.push_back(value);
                }
                const double medianIntensity = medianValue(values);
                std::vector<double> deviations;
                deviations.reserve(values.size());
                for (const double value : values) {
                    deviations.push_back(std::abs(value - medianIntensity));
                }
                const double mad = medianValue(deviations);
                const double robustScale = std::max(0.03, mad / 0.6745);
                adaptiveHighThreshold = std::min(static_cast<double>(highOutlierThreshold), medianIntensity + 2.5 * robustScale);
            }

            std::vector<int> fitIndices;
            fitIndices.reserve(observations.size());
            int highOutliers = 0;
            for (const int i : observations) {
                const float intensity = imageRows[i][x];
                if (robust && intensity > adaptiveHighThreshold) {
                    ++highOutliers;
                    continue;
                }
                fitIndices.push_back(i);
            }
            if (fitIndices.size() < 3) {
                fitIndices = observations;
                highOutliers = 0;
            }
            highRow[x] = static_cast<float>(highOutliers);
            if (highOutliers > 0) {
                specularRow[x] = 255;
            }

            if (fitIndices.size() < 3) {
                continue;
            }

            auto lightAt = [&](int i) {
                if (lightingModel == LightingModel::NearFieldRing) {
                    return nearFieldRingLightDirection(
                        lights[i],
                        i,
                        n,
                        x,
                        y,
                        ringLightRadiusMm,
                        ringLightHeightMm,
                        pixelScaleMm,
                        lightingCenter);
                }
                return lights[i];
            };

            auto weightedSolve = [&](const std::vector<double>& weights, cv::Vec3d& solution) {
                double a00 = 0.0;
                double a01 = 0.0;
                double a02 = 0.0;
                double a11 = 0.0;
                double a12 = 0.0;
                double a22 = 0.0;
                cv::Vec3d b(0.0, 0.0, 0.0);
                for (size_t k = 0; k < fitIndices.size(); ++k) {
                    const int i = fitIndices[k];
                    const double w = weights.empty() ? 1.0 : weights[k];
                    if (w <= 0.0) {
                        continue;
                    }
                    const float intensity = imageRows[i][x];
                    const cv::Vec3f l = lightAt(i);
                    a00 += w * static_cast<double>(l[0]) * l[0];
                    a01 += w * static_cast<double>(l[0]) * l[1];
                    a02 += w * static_cast<double>(l[0]) * l[2];
                    a11 += w * static_cast<double>(l[1]) * l[1];
                    a12 += w * static_cast<double>(l[1]) * l[2];
                    a22 += w * static_cast<double>(l[2]) * l[2];
                    b[0] += w * static_cast<double>(l[0]) * intensity;
                    b[1] += w * static_cast<double>(l[1]) * intensity;
                    b[2] += w * static_cast<double>(l[2]) * intensity;
                }
                return solveSymmetric3x3(a00, a01, a02, a11, a12, a22, b, solution);
            };

            cv::Vec3d solution;
            if (!weightedSolve({}, solution)) {
                continue;
            }
            if (robust) {
                std::vector<double> weights(fitIndices.size(), 1.0);
                for (int iter = 0; iter < 8; ++iter) {
                    if (!weightedSolve(weights, solution)) {
                        break;
                    }
                    std::vector<double> absResiduals;
                    absResiduals.reserve(fitIndices.size());
                    for (const int i : fitIndices) {
                        const double predicted = lightAt(i).dot(cv::Vec3f(
                            static_cast<float>(solution[0]),
                            static_cast<float>(solution[1]),
                            static_cast<float>(solution[2])));
                        absResiduals.push_back(std::abs(predicted - static_cast<double>(imageRows[i][x])));
                    }
                    std::vector<double> sorted = absResiduals;
                    const size_t mid = sorted.size() / 2;
                    std::nth_element(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(mid), sorted.end());
                    const double medianAbs = sorted[mid];
                    const double sigma = std::max(0.01, medianAbs / 0.6745);
                    const double delta = 1.345 * sigma;
                    double maxChange = 0.0;
                    for (size_t k = 0; k < absResiduals.size(); ++k) {
                        const double r = absResiduals[k];
                        const double next = r <= delta ? 1.0 : delta / std::max(r, 1.0e-8);
                        maxChange = std::max(maxChange, std::abs(next - weights[k]));
                        weights[k] = next;
                    }
                    if (maxChange < 1.0e-3) {
                        break;
                    }
                }
                if (!weightedSolve(weights, solution)) {
                    continue;
                }
            }

            cv::Vec3f g(
                static_cast<float>(solution[0]),
                static_cast<float>(solution[1]),
                static_cast<float>(solution[2]));
            const float rho = std::sqrt(g.dot(g));
            if (!std::isfinite(rho) || rho <= 1.0e-6f) {
                continue;
            }
            const cv::Vec3f normal = g / rho;
            if (normal[2] <= 0.0f) {
                continue;
            }

            double err = 0.0;
            double weightSum = 0.0;
            int downweighted = 0;
            for (const int i : fitIndices) {
                const float intensity = imageRows[i][x];
                const double d = static_cast<double>(lightAt(i).dot(g) - intensity);
                err += d * d;
            }
            err = std::sqrt(err / static_cast<double>(fitIndices.size()));
            if (robust) {
                std::vector<double> absResiduals;
                absResiduals.reserve(fitIndices.size());
                for (const int i : fitIndices) {
                    absResiduals.push_back(std::abs(static_cast<double>(lightAt(i).dot(g) - imageRows[i][x])));
                }
                const double medianAbs = medianValue(absResiduals);
                const double sigma = std::max(0.01, medianAbs / 0.6745);
                const double delta = 1.345 * sigma;
                for (const double r : absResiduals) {
                    const double w = r <= delta ? 1.0 : delta / std::max(r, 1.0e-8);
                    weightSum += w;
                    if (w < 0.75) {
                        ++downweighted;
                    }
                }
                if (downweighted > 0) {
                    specularRow[x] = 255;
                }
                weightRow[x] = static_cast<float>(weightSum / static_cast<double>(fitIndices.size()));
            } else {
                weightRow[x] = 1.0f;
            }

            normalRow[x] = normal;
            albedoRow[x] = rho;
            residualRow[x] = static_cast<float>(err);
            validRow[x] = 255;
        }
    }
}

void solveUncalibratedPhotometricStereo(
    const std::vector<cv::Mat>& images,
    const cv::Mat& inputMask,
    float shadowThreshold,
    cv::Mat& normalMap,
    cv::Mat& albedo,
    cv::Mat& residual,
    cv::Mat& validMask,
    const std::function<void(const std::string&)>& progress) {
    const int rows = images[0].rows;
    const int cols = images[0].cols;
    const int n = static_cast<int>(images.size());
    constexpr int kRank = 4;
    if (n < kRank) {
        die("Uncalibrated no-sphere mode requires at least 4 images.");
    }

    reportProgress(progress, "Unknown lighting: building image covariance...");
    cv::Mat covariance(n, n, CV_64F, cv::Scalar(0));
    int validPixels = 0;
    std::vector<const float*> imageRows;
    imageRows.reserve(images.size());
    for (int y = 0; y < rows; ++y) {
        const uchar* maskRow = inputMask.ptr<uchar>(y);
        imageRows.clear();
        for (const cv::Mat& image : images) {
            imageRows.push_back(image.ptr<float>(y));
        }
        for (int x = 0; x < cols; ++x) {
            if (maskRow[x] == 0) {
                continue;
            }
            bool usable = false;
            for (int i = 0; i < n; ++i) {
                const float value = imageRows[i][x];
                if (std::isfinite(value) && value > shadowThreshold) {
                    usable = true;
                    break;
                }
            }
            if (!usable) {
                continue;
            }
            for (int i = 0; i < n; ++i) {
                const double vi = std::isfinite(imageRows[i][x]) ? std::max(0.0f, imageRows[i][x]) : 0.0f;
                double* crow = covariance.ptr<double>(i);
                for (int j = 0; j < n; ++j) {
                    const double vj = std::isfinite(imageRows[j][x]) ? std::max(0.0f, imageRows[j][x]) : 0.0f;
                    crow[j] += vi * vj;
                }
            }
            ++validPixels;
        }
    }
    if (validPixels < 10) {
        die("Not enough valid pixels for uncalibrated unknown-lighting solve.");
    }

    cv::Mat eigenValues;
    cv::Mat eigenVectors;
    if (!cv::eigen(covariance, eigenValues, eigenVectors)) {
        die("Could not decompose image covariance for uncalibrated solve.");
    }

    reportProgress(progress, "Unknown lighting: fitting first-order harmonic constraint...");
    cv::Mat constraint(10, 10, CV_64F, cv::Scalar(0));
    for (int y = 0; y < rows; ++y) {
        const uchar* maskRow = inputMask.ptr<uchar>(y);
        imageRows.clear();
        for (const cv::Mat& image : images) {
            imageRows.push_back(image.ptr<float>(y));
        }
        for (int x = 0; x < cols; ++x) {
            if (maskRow[x] == 0) {
                continue;
            }
            bool usable = false;
            std::array<double, kRank> b = {};
            for (int r = 0; r < kRank; ++r) {
                double value = 0.0;
                const double* urow = eigenVectors.ptr<double>(r);
                for (int i = 0; i < n; ++i) {
                    const float intensity = imageRows[i][x];
                    const double vi = std::isfinite(intensity) ? std::max(0.0f, intensity) : 0.0f;
                    value += urow[i] * vi;
                    if (vi > shadowThreshold) {
                        usable = true;
                    }
                }
                b[static_cast<size_t>(r)] = value;
            }
            if (!usable) {
                continue;
            }

            const std::array<double, 10> phi = {
                b[0] * b[0],
                2.0 * b[0] * b[1],
                2.0 * b[0] * b[2],
                2.0 * b[0] * b[3],
                b[1] * b[1],
                2.0 * b[1] * b[2],
                2.0 * b[1] * b[3],
                b[2] * b[2],
                2.0 * b[2] * b[3],
                b[3] * b[3]};
            for (int i = 0; i < 10; ++i) {
                double* row = constraint.ptr<double>(i);
                for (int j = 0; j < 10; ++j) {
                    row[j] += phi[static_cast<size_t>(i)] * phi[static_cast<size_t>(j)];
                }
            }
        }
    }

    cv::Mat qValues;
    cv::Mat qVectors;
    if (!cv::eigen(constraint, qValues, qVectors)) {
        die("Could not solve first-order harmonic constraint.");
    }
    cv::Matx44d q(
        qVectors.at<double>(9, 0), qVectors.at<double>(9, 1), qVectors.at<double>(9, 2), qVectors.at<double>(9, 3),
        qVectors.at<double>(9, 1), qVectors.at<double>(9, 4), qVectors.at<double>(9, 5), qVectors.at<double>(9, 6),
        qVectors.at<double>(9, 2), qVectors.at<double>(9, 5), qVectors.at<double>(9, 7), qVectors.at<double>(9, 8),
        qVectors.at<double>(9, 3), qVectors.at<double>(9, 6), qVectors.at<double>(9, 8), qVectors.at<double>(9, 9));

    cv::Mat qMat(4, 4, CV_64F);
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            qMat.at<double>(r, c) = q(r, c);
        }
    }

    cv::Mat lorentzValues;
    cv::Mat lorentzVectors;
    if (!cv::eigen(qMat, lorentzValues, lorentzVectors)) {
        die("Could not factor unknown-lighting metric.");
    }
    int positives = 0;
    int negatives = 0;
    for (int i = 0; i < 4; ++i) {
        const double value = lorentzValues.at<double>(i, 0);
        positives += value > 1.0e-10 ? 1 : 0;
        negatives += value < -1.0e-10 ? 1 : 0;
    }
    if (positives == 3 && negatives == 1) {
        qMat = -qMat;
        cv::eigen(qMat, lorentzValues, lorentzVectors);
    }

    std::vector<int> order;
    order.reserve(4);
    for (int i = 0; i < 4; ++i) {
        if (lorentzValues.at<double>(i, 0) > 1.0e-10) {
            order.push_back(i);
            break;
        }
    }
    for (int i = 0; i < 4; ++i) {
        if (lorentzValues.at<double>(i, 0) < -1.0e-10) {
            order.push_back(i);
        }
    }
    if (order.size() != 4) {
        solveUncalibratedPcaFallback(
            images,
            inputMask,
            shadowThreshold,
            eigenVectors,
            normalMap,
            albedo,
            residual,
            validMask,
            progress);
        return;
    }

    cv::Matx44d transform = cv::Matx44d::zeros();
    for (int r = 0; r < 4; ++r) {
        const int source = order[static_cast<size_t>(r)];
        const double scale = std::sqrt(std::abs(lorentzValues.at<double>(source, 0)));
        for (int c = 0; c < 4; ++c) {
            transform(r, c) = scale * lorentzVectors.at<double>(source, c);
        }
    }

    double albedoSignSum = 0.0;
    for (int y = 0; y < rows; ++y) {
        const uchar* maskRow = inputMask.ptr<uchar>(y);
        imageRows.clear();
        for (const cv::Mat& image : images) {
            imageRows.push_back(image.ptr<float>(y));
        }
        for (int x = 0; x < cols; ++x) {
            if (maskRow[x] == 0) {
                continue;
            }
            cv::Vec4d b(0.0, 0.0, 0.0, 0.0);
            bool usable = false;
            for (int r = 0; r < kRank; ++r) {
                const double* urow = eigenVectors.ptr<double>(r);
                for (int i = 0; i < n; ++i) {
                    const float intensity = imageRows[i][x];
                    const double vi = std::isfinite(intensity) ? std::max(0.0f, intensity) : 0.0f;
                    b[r] += urow[i] * vi;
                    usable = usable || vi > shadowThreshold;
                }
            }
            if (usable) {
                const cv::Vec4d s = transform * b;
                albedoSignSum += s[0];
            }
        }
    }
    if (albedoSignSum < 0.0) {
        for (int c = 0; c < 4; ++c) {
            transform(0, c) = -transform(0, c);
        }
    }

    reportProgress(progress, "Unknown lighting: building relative normal map...");
    normalMap = cv::Mat(rows, cols, CV_32FC3, cv::Scalar(0, 0, 0));
    albedo = cv::Mat(rows, cols, CV_32F, cv::Scalar(0));
    residual = cv::Mat(rows, cols, CV_32F, cv::Scalar(0));
    validMask = cv::Mat(rows, cols, CV_8U, cv::Scalar(0));
    cv::Vec3d averageNormal(0.0, 0.0, 0.0);

    for (int y = 0; y < rows; ++y) {
        const uchar* maskRow = inputMask.ptr<uchar>(y);
        cv::Vec3f* normalRow = normalMap.ptr<cv::Vec3f>(y);
        float* albedoRow = albedo.ptr<float>(y);
        float* residualRow = residual.ptr<float>(y);
        uchar* validRow = validMask.ptr<uchar>(y);
        imageRows.clear();
        for (const cv::Mat& image : images) {
            imageRows.push_back(image.ptr<float>(y));
        }
        for (int x = 0; x < cols; ++x) {
            if (maskRow[x] == 0) {
                continue;
            }
            cv::Vec4d b(0.0, 0.0, 0.0, 0.0);
            bool usable = false;
            for (int r = 0; r < kRank; ++r) {
                const double* urow = eigenVectors.ptr<double>(r);
                for (int i = 0; i < n; ++i) {
                    const float intensity = imageRows[i][x];
                    const double vi = std::isfinite(intensity) ? std::max(0.0f, intensity) : 0.0f;
                    b[r] += urow[i] * vi;
                    usable = usable || vi > shadowThreshold;
                }
            }
            if (!usable) {
                continue;
            }

            const cv::Vec4d s = transform * b;
            const double rho = s[0];
            if (!std::isfinite(rho) || rho <= 1.0e-8) {
                continue;
            }
            const cv::Vec3d nrm = normalizeVec3d(cv::Vec3d(s[1] / rho, s[2] / rho, s[3] / rho));
            if (!std::isfinite(nrm[0]) || !std::isfinite(nrm[1]) || !std::isfinite(nrm[2])) {
                continue;
            }

            double fit = 0.0;
            for (int i = 0; i < n; ++i) {
                const double* u0 = eigenVectors.ptr<double>(0);
                const double* u1 = eigenVectors.ptr<double>(1);
                const double* u2 = eigenVectors.ptr<double>(2);
                const double* u3 = eigenVectors.ptr<double>(3);
                const double predicted = u0[i] * b[0] + u1[i] * b[1] + u2[i] * b[2] + u3[i] * b[3];
                const float intensity = imageRows[i][x];
                const double observed = std::isfinite(intensity) ? std::max(0.0f, intensity) : 0.0;
                const double d = predicted - observed;
                fit += d * d;
            }

            normalRow[x] = cv::Vec3f(static_cast<float>(nrm[0]), static_cast<float>(nrm[1]), static_cast<float>(nrm[2]));
            albedoRow[x] = static_cast<float>(rho);
            residualRow[x] = static_cast<float>(std::sqrt(fit / static_cast<double>(n)));
            validRow[x] = 255;
            averageNormal += nrm * rho;
        }
    }

    reportProgress(progress, "Unknown lighting: orienting normals toward camera...");
    const cv::Matx33d faceCamera = rotationBetween(averageNormal, cv::Vec3d(0.0, 0.0, 1.0));
    applyNormalRotation(normalMap, validMask, faceCamera);

    double bestTheta = 0.0;
    double bestCost = std::numeric_limits<double>::infinity();
    constexpr int kSearchSteps = 72;
    for (int i = 0; i < kSearchSteps; ++i) {
        const double theta = (CV_PI * static_cast<double>(i)) / static_cast<double>(kSearchSteps);
        const double cost = curlCostForNormals(normalMap, validMask, rotationAroundZ(theta));
        if (cost < bestCost) {
            bestCost = cost;
            bestTheta = theta;
        }
    }
    applyNormalRotation(normalMap, validMask, rotationAroundZ(bestTheta));
    stabilizeVisualReliefNormals(normalMap, validMask, progress);
    reportProgress(progress, "Unknown lighting: done.");
}

void flattenNormalField(cv::Mat& normalMap, const cv::Mat& validMask, FlattenMode mode) {
    if (mode == FlattenMode::None) {
        return;
    }
    const int rows = normalMap.rows;
    const int cols = normalMap.cols;
    cv::Mat p(rows, cols, CV_32F, cv::Scalar(0));
    cv::Mat q(rows, cols, CV_32F, cv::Scalar(0));

    for (int y = 0; y < rows; ++y) {
        const cv::Vec3f* nrow = normalMap.ptr<cv::Vec3f>(y);
        const uchar* mrow = validMask.ptr<uchar>(y);
        float* prow = p.ptr<float>(y);
        float* qrow = q.ptr<float>(y);
        for (int x = 0; x < cols; ++x) {
            if (mrow[x] == 0 || nrow[x][2] <= 1.0e-4f) {
                continue;
            }
            prow[x] = -nrow[x][0] / nrow[x][2];
            qrow[x] = nrow[x][1] / nrow[x][2];
        }
    }

    const double minDim = static_cast<double>(std::max(1, std::min(rows, cols)));
    const double sigma = mode == FlattenMode::Gentle ? minDim * 0.075 : minDim * 0.14;
    const float amount = mode == FlattenMode::Gentle ? 0.60f : 0.85f;
    const cv::Mat lowP = maskedGaussianBlurFloat(p, validMask, sigma);
    const cv::Mat lowQ = maskedGaussianBlurFloat(q, validMask, sigma);

    for (int y = 0; y < rows; ++y) {
        cv::Vec3f* nrow = normalMap.ptr<cv::Vec3f>(y);
        const uchar* mrow = validMask.ptr<uchar>(y);
        const float* prow = p.ptr<float>(y);
        const float* qrow = q.ptr<float>(y);
        const float* lprow = lowP.ptr<float>(y);
        const float* lqrow = lowQ.ptr<float>(y);
        for (int x = 0; x < cols; ++x) {
            if (mrow[x] == 0 || nrow[x][2] <= 1.0e-4f) {
                continue;
            }
            const float highP = prow[x] - amount * lprow[x];
            const float highQ = qrow[x] - amount * lqrow[x];
            cv::Vec3f next(-highP, highQ, 1.0f);
            const float norm = std::sqrt(next.dot(next));
            if (norm > 1.0e-6f && std::isfinite(norm)) {
                nrow[x] = next / norm;
            }
        }
    }
}

void clampSlopeMagnitude(cv::Mat& p, cv::Mat& q, const cv::Mat& validMask, double slopeCap) {
    if (!std::isfinite(slopeCap) || slopeCap <= 0.0) {
        return;
    }
    const float cap = static_cast<float>(slopeCap);
    for (int y = 0; y < p.rows; ++y) {
        float* prow = p.ptr<float>(y);
        float* qrow = q.ptr<float>(y);
        const uchar* mrow = validMask.ptr<uchar>(y);
        for (int x = 0; x < p.cols; ++x) {
            if (mrow[x] == 0) {
                continue;
            }
            const float px = prow[x];
            const float qx = qrow[x];
            if (!std::isfinite(px) || !std::isfinite(qx)) {
                prow[x] = 0.0f;
                qrow[x] = 0.0f;
                continue;
            }
            const float magnitude = std::sqrt(px * px + qx * qx);
            if (magnitude > cap && magnitude > 1.0e-8f) {
                const float scale = cap / magnitude;
                prow[x] *= scale;
                qrow[x] *= scale;
            }
        }
    }
}

cv::Mat extendMaskedFloatField(const cv::Mat& src, const cv::Mat& validMask) {
    cv::Mat filled = cv::Mat::zeros(src.size(), CV_32F);
    src.copyTo(filled, validMask);

    cv::Mat unknown;
    cv::bitwise_not(validMask, unknown);
    if (cv::countNonZero(unknown) == 0 || cv::countNonZero(validMask) == 0) {
        return filled;
    }

    cv::Mat knownWeight(src.size(), CV_32F, cv::Scalar(0));
    knownWeight.setTo(cv::Scalar(1), validMask);

    for (int pass = 0; pass < 7 && cv::countNonZero(unknown) > 0; ++pass) {
        cv::Mat blurredValue;
        cv::Mat blurredWeight;
        cv::GaussianBlur(filled.mul(knownWeight), blurredValue, cv::Size(), 2.0, 2.0, cv::BORDER_REPLICATE);
        cv::GaussianBlur(knownWeight, blurredWeight, cv::Size(), 2.0, 2.0, cv::BORDER_REPLICATE);

        cv::Mat newlyKnown = unknown & (blurredWeight > 1.0e-5f);
        for (int y = 0; y < src.rows; ++y) {
            float* frow = filled.ptr<float>(y);
            float* wrow = knownWeight.ptr<float>(y);
            const float* vrow = blurredValue.ptr<float>(y);
            const float* brow = blurredWeight.ptr<float>(y);
            const uchar* nrow = newlyKnown.ptr<uchar>(y);
            for (int x = 0; x < src.cols; ++x) {
                if (nrow[x] != 0) {
                    frow[x] = vrow[x] / std::max(brow[x], 1.0e-6f);
                    wrow[x] = 1.0f;
                }
            }
        }
        cv::bitwise_not(knownWeight > 0.5f, unknown);
    }

    return filled;
}

cv::Mat integrateHeightDct(
    const cv::Mat& normalMap,
    const cv::Mat& validMask,
    double slopeCap,
    int iterations,
    const std::function<void(int, int)>& progress) {
    (void)iterations;
    const int rows = normalMap.rows;
    const int cols = normalMap.cols;
    cv::Mat p(rows, cols, CV_32F, cv::Scalar(0));
    cv::Mat q(rows, cols, CV_32F, cv::Scalar(0));

    for (int y = 0; y < rows; ++y) {
        const cv::Vec3f* nrow = normalMap.ptr<cv::Vec3f>(y);
        const uchar* mrow = validMask.ptr<uchar>(y);
        float* prow = p.ptr<float>(y);
        float* qrow = q.ptr<float>(y);
        for (int x = 0; x < cols; ++x) {
            if (mrow[x] == 0 || nrow[x][2] <= 1.0e-4f) {
                continue;
            }
            prow[x] = -nrow[x][0] / nrow[x][2];
            qrow[x] = nrow[x][1] / nrow[x][2];
        }
    }
    clampSlopeMagnitude(p, q, validMask, slopeCap);
    p = extendMaskedFloatField(p, validMask);
    q = extendMaskedFloatField(q, validMask);
    if (progress) {
        progress(1, 3);
    }

    cv::Mat divergence(rows, cols, CV_32F, cv::Scalar(0));
    for (int y = 0; y < rows; ++y) {
        const float* prow = p.ptr<float>(y);
        const float* qrow = q.ptr<float>(y);
        const float* prevQ = y > 0 ? q.ptr<float>(y - 1) : nullptr;
        float* out = divergence.ptr<float>(y);
        for (int x = 0; x < cols; ++x) {
            const float leftP = x > 0 ? prow[x - 1] : 0.0f;
            const float upQ = y > 0 ? prevQ[x] : 0.0f;
            out[x] = (prow[x] - leftP) + (qrow[x] - upQ);
        }
    }

    const int dctRows = 2 * cv::getOptimalDFTSize((rows + 1) / 2);
    const int dctCols = 2 * cv::getOptimalDFTSize((cols + 1) / 2);
    cv::Mat padded;
    cv::copyMakeBorder(
        divergence,
        padded,
        0,
        dctRows - rows,
        0,
        dctCols - cols,
        cv::BORDER_CONSTANT,
        cv::Scalar(0));

    cv::Mat dctCoeffs;
    cv::dct(padded, dctCoeffs);
    if (progress) {
        progress(2, 3);
    }

    for (int y = 0; y < dctRows; ++y) {
        float* row = dctCoeffs.ptr<float>(y);
        const double cy = std::cos(CV_PI * static_cast<double>(y) / static_cast<double>(dctRows));
        for (int x = 0; x < dctCols; ++x) {
            if (x == 0 && y == 0) {
                row[x] = 0.0f;
                continue;
            }
            const double cx = std::cos(CV_PI * static_cast<double>(x) / static_cast<double>(dctCols));
            const double denom = 2.0 * cx + 2.0 * cy - 4.0;
            row[x] = static_cast<float>(static_cast<double>(row[x]) / denom);
        }
    }

    cv::Mat paddedZ;
    cv::dct(dctCoeffs, paddedZ, cv::DCT_INVERSE);
    cv::Mat z = paddedZ(cv::Rect(0, 0, cols, rows)).clone();

    double sum = 0.0;
    int countValid = 0;
    for (int y = 0; y < rows; ++y) {
        const uchar* mrow = validMask.ptr<uchar>(y);
        float* zrow = z.ptr<float>(y);
        for (int x = 0; x < cols; ++x) {
            if (mrow[x] == 0 || !std::isfinite(zrow[x])) {
                zrow[x] = 0.0f;
                continue;
            }
            sum += zrow[x];
            ++countValid;
        }
    }
    if (countValid > 0) {
        const float mean = static_cast<float>(sum / static_cast<double>(countValid));
        for (int y = 0; y < rows; ++y) {
            const uchar* mrow = validMask.ptr<uchar>(y);
            float* zrow = z.ptr<float>(y);
            for (int x = 0; x < cols; ++x) {
                if (mrow[x] != 0) {
                    zrow[x] -= mean;
                }
            }
        }
    }
    if (progress) {
        progress(3, 3);
    }
    return z;
}

void normalsToWeightedSlopes(
    const cv::Mat& normalMap,
    const cv::Mat& validMask,
    double slopeCap,
    cv::Mat& p,
    cv::Mat& q,
    cv::Mat& confidence) {
    const int rows = normalMap.rows;
    const int cols = normalMap.cols;
    p = cv::Mat(rows, cols, CV_32F, cv::Scalar(0));
    q = cv::Mat(rows, cols, CV_32F, cv::Scalar(0));
    confidence = cv::Mat(rows, cols, CV_32F, cv::Scalar(0));
    const float cap = static_cast<float>(slopeCap > 0.0 ? slopeCap : 0.0);

    for (int y = 0; y < rows; ++y) {
        const cv::Vec3f* nrow = normalMap.ptr<cv::Vec3f>(y);
        const uchar* mrow = validMask.ptr<uchar>(y);
        float* prow = p.ptr<float>(y);
        float* qrow = q.ptr<float>(y);
        float* crow = confidence.ptr<float>(y);
        for (int x = 0; x < cols; ++x) {
            if (mrow[x] == 0) {
                continue;
            }
            const cv::Vec3f normal = nrow[x];
            if (!std::isfinite(normal[0]) || !std::isfinite(normal[1]) || !std::isfinite(normal[2]) || normal[2] <= 1.0e-4f) {
                continue;
            }
            float px = -normal[0] / normal[2];
            float qx = normal[1] / normal[2];
            float magnitude = std::sqrt(px * px + qx * qx);
            if (!std::isfinite(magnitude)) {
                continue;
            }
            if (cap > 0.0f && magnitude > 1.0e-8f) {
                const float softMagnitude = cap * std::tanh(magnitude / cap);
                const float scale = softMagnitude / magnitude;
                px *= scale;
                qx *= scale;
                magnitude = softMagnitude;
            }
            const float nzWeight = std::clamp((normal[2] - 0.08f) / 0.72f, 0.0f, 1.0f);
            const float slopeWeight = cap > 0.0f
                ? 1.0f / std::sqrt(1.0f + (magnitude / std::max(cap, 1.0e-4f)) * (magnitude / std::max(cap, 1.0e-4f)))
                : 1.0f;
            prow[x] = px;
            qrow[x] = qx;
            crow[x] = std::max(0.03f, nzWeight * nzWeight * slopeWeight);
        }
    }
}

void centerHeightOverMask(cv::Mat& z, const cv::Mat& validMask) {
    double sum = 0.0;
    int count = 0;
    for (int y = 0; y < z.rows; ++y) {
        const uchar* mrow = validMask.ptr<uchar>(y);
        const float* zrow = z.ptr<float>(y);
        for (int x = 0; x < z.cols; ++x) {
            if (mrow[x] != 0 && std::isfinite(zrow[x])) {
                sum += zrow[x];
                ++count;
            }
        }
    }
    if (count == 0) {
        return;
    }
    const float mean = static_cast<float>(sum / static_cast<double>(count));
    for (int y = 0; y < z.rows; ++y) {
        const uchar* mrow = validMask.ptr<uchar>(y);
        float* zrow = z.ptr<float>(y);
        for (int x = 0; x < z.cols; ++x) {
            if (mrow[x] != 0 && std::isfinite(zrow[x])) {
                zrow[x] -= mean;
            } else {
                zrow[x] = 0.0f;
            }
        }
    }
}

void updateRobustEdgeWeights(
    const cv::Mat& z,
    const cv::Mat& validMask,
    const cv::Mat& p,
    const cv::Mat& q,
    const cv::Mat& confidence,
    cv::Mat& wx,
    cv::Mat& wy) {
    const int rows = z.rows;
    const int cols = z.cols;
    constexpr float kDelta = 0.55f;
    wx = cv::Mat(rows, cols, CV_32F, cv::Scalar(0));
    wy = cv::Mat(rows, cols, CV_32F, cv::Scalar(0));

    for (int y = 0; y < rows; ++y) {
        const uchar* mrow = validMask.ptr<uchar>(y);
        const uchar* nextMrow = y + 1 < rows ? validMask.ptr<uchar>(y + 1) : nullptr;
        const float* zrow = z.ptr<float>(y);
        const float* nextZrow = y + 1 < rows ? z.ptr<float>(y + 1) : nullptr;
        const float* prow = p.ptr<float>(y);
        const float* qrow = q.ptr<float>(y);
        const float* crow = confidence.ptr<float>(y);
        const float* nextCrow = y + 1 < rows ? confidence.ptr<float>(y + 1) : nullptr;
        float* wxrow = wx.ptr<float>(y);
        float* wyrow = wy.ptr<float>(y);
        for (int x = 0; x < cols; ++x) {
            if (mrow[x] == 0) {
                continue;
            }
            if (x + 1 < cols && mrow[x + 1] != 0) {
                const float base = 0.5f * (crow[x] + crow[x + 1]);
                const float residual = (zrow[x + 1] - zrow[x]) - prow[x];
                wxrow[x] = base / std::sqrt(1.0f + (residual / kDelta) * (residual / kDelta));
            }
            if (y + 1 < rows && nextMrow[x] != 0) {
                const float base = 0.5f * (crow[x] + nextCrow[x]);
                const float residual = (nextZrow[x] - zrow[x]) - qrow[x];
                wyrow[x] = base / std::sqrt(1.0f + (residual / kDelta) * (residual / kDelta));
            }
        }
    }
}

void sorHeightSweep(
    cv::Mat& z,
    const cv::Mat& validMask,
    const cv::Mat& p,
    const cv::Mat& q,
    const cv::Mat& wx,
    const cv::Mat& wy,
    double omega) {
    const int rows = z.rows;
    const int cols = z.cols;
    const float relaxation = static_cast<float>(omega);

    for (int y = 0; y < rows; ++y) {
        const uchar* mrow = validMask.ptr<uchar>(y);
        const uchar* prevMrow = y > 0 ? validMask.ptr<uchar>(y - 1) : nullptr;
        const uchar* nextMrow = y + 1 < rows ? validMask.ptr<uchar>(y + 1) : nullptr;
        float* zrow = z.ptr<float>(y);
        const float* prevZrow = y > 0 ? z.ptr<float>(y - 1) : nullptr;
        const float* nextZrow = y + 1 < rows ? z.ptr<float>(y + 1) : nullptr;
        const float* prow = p.ptr<float>(y);
        const float* qrow = q.ptr<float>(y);
        const float* prevQrow = y > 0 ? q.ptr<float>(y - 1) : nullptr;
        const float* wxrow = wx.ptr<float>(y);
        const float* wyrow = wy.ptr<float>(y);
        const float* prevWyrow = y > 0 ? wy.ptr<float>(y - 1) : nullptr;
        for (int x = 0; x < cols; ++x) {
            if (mrow[x] == 0) {
                zrow[x] = 0.0f;
                continue;
            }

            float numerator = 0.0f;
            float denominator = 0.0f;
            if (x + 1 < cols && mrow[x + 1] != 0) {
                const float w = wxrow[x];
                numerator += w * (zrow[x + 1] - prow[x]);
                denominator += w;
            }
            if (x > 0 && mrow[x - 1] != 0) {
                const float w = wxrow[x - 1];
                numerator += w * (zrow[x - 1] + prow[x - 1]);
                denominator += w;
            }
            if (y + 1 < rows && nextMrow[x] != 0) {
                const float w = wyrow[x];
                numerator += w * (nextZrow[x] - qrow[x]);
                denominator += w;
            }
            if (y > 0 && prevMrow[x] != 0) {
                const float w = prevWyrow[x];
                numerator += w * (prevZrow[x] + prevQrow[x]);
                denominator += w;
            }
            if (denominator > 1.0e-7f) {
                const float candidate = numerator / denominator;
                zrow[x] += relaxation * (candidate - zrow[x]);
            }
        }
    }
}

cv::Mat integrateHeightRobustMasked(
    const cv::Mat& normalMap,
    const cv::Mat& validMask,
    double slopeCap,
    int iterations,
    const std::function<void(int, int)>& progress) {
    cv::Mat p;
    cv::Mat q;
    cv::Mat confidence;
    normalsToWeightedSlopes(normalMap, validMask, slopeCap, p, q, confidence);
    if (progress) {
        progress(1, 6);
    }

    cv::Mat z = integrateHeightDct(normalMap, validMask, slopeCap, iterations, {});
    centerHeightOverMask(z, validMask);
    if (progress) {
        progress(2, 6);
    }

    cv::Mat wx;
    cv::Mat wy;
    const int sweepsPerRound = std::max(12, std::min(45, iterations > 0 ? iterations / 30 : 24));
    for (int round = 0; round < 4; ++round) {
        updateRobustEdgeWeights(z, validMask, p, q, confidence, wx, wy);
        for (int sweep = 0; sweep < sweepsPerRound; ++sweep) {
            sorHeightSweep(z, validMask, p, q, wx, wy, 1.45);
        }
        centerHeightOverMask(z, validMask);
        if (progress) {
            progress(3 + round, 6);
        }
    }

    return z;
}

cv::Mat integrateHeight(
    const cv::Mat& normalMap,
    const cv::Mat& validMask,
    HeightSolverMode solverMode,
    double slopeCap,
    int iterations,
    const std::function<void(int, int)>& progress) {
    if (solverMode == HeightSolverMode::FastDct) {
        return integrateHeightDct(normalMap, validMask, slopeCap, iterations, progress);
    }
    return integrateHeightRobustMasked(normalMap, validMask, slopeCap, iterations, progress);
}

void removeBestFitPlane(cv::Mat& height, const cv::Mat& validMask) {
    double a00 = 0.0;
    double a01 = 0.0;
    double a02 = 0.0;
    double a11 = 0.0;
    double a12 = 0.0;
    double a22 = 0.0;
    cv::Vec3d b(0.0, 0.0, 0.0);

    for (int y = 0; y < height.rows; ++y) {
        const float* hrow = height.ptr<float>(y);
        const uchar* mrow = validMask.ptr<uchar>(y);
        for (int x = 0; x < height.cols; ++x) {
            if (mrow[x] == 0 || !std::isfinite(hrow[x])) {
                continue;
            }
            const double xd = static_cast<double>(x);
            const double yd = static_cast<double>(y);
            const double z = static_cast<double>(hrow[x]);
            a00 += xd * xd;
            a01 += xd * yd;
            a02 += xd;
            a11 += yd * yd;
            a12 += yd;
            a22 += 1.0;
            b[0] += xd * z;
            b[1] += yd * z;
            b[2] += z;
        }
    }

    cv::Vec3d plane;
    if (!solveSymmetric3x3(a00, a01, a02, a11, a12, a22, b, plane)) {
        return;
    }

    for (int y = 0; y < height.rows; ++y) {
        float* hrow = height.ptr<float>(y);
        const uchar* mrow = validMask.ptr<uchar>(y);
        for (int x = 0; x < height.cols; ++x) {
            if (mrow[x] == 0 || !std::isfinite(hrow[x])) {
                continue;
            }
            const double trend =
                plane[0] * static_cast<double>(x) +
                plane[1] * static_cast<double>(y) +
                plane[2];
            hrow[x] = static_cast<float>(static_cast<double>(hrow[x]) - trend);
        }
    }
}

void removeHeightCurl(cv::Mat& height, const cv::Mat& validMask, HeightFlattenMode mode) {
    if (mode == HeightFlattenMode::None || height.empty() || validMask.empty()) {
        return;
    }

    const int nTerms = mode == HeightFlattenMode::Quadratic ? 6 : 4;
    cv::Mat normal = cv::Mat::zeros(nTerms, nTerms, CV_64F);
    cv::Mat rhs = cv::Mat::zeros(nTerms, 1, CV_64F);
    const double cx = 0.5 * static_cast<double>(height.cols - 1);
    const double cy = 0.5 * static_cast<double>(height.rows - 1);
    const double scale = std::max(1.0, 0.5 * static_cast<double>(std::max(height.cols, height.rows)));
    int count = 0;

    std::array<double, 6> terms{};
    for (int y = 0; y < height.rows; ++y) {
        const float* hrow = height.ptr<float>(y);
        const uchar* mrow = validMask.ptr<uchar>(y);
        const double yn = (static_cast<double>(y) - cy) / scale;
        for (int x = 0; x < height.cols; ++x) {
            if (mrow[x] == 0 || !std::isfinite(hrow[x])) {
                continue;
            }
            const double xn = (static_cast<double>(x) - cx) / scale;
            terms[0] = 1.0;
            terms[1] = xn;
            terms[2] = yn;
            if (mode == HeightFlattenMode::Quadratic) {
                terms[3] = xn * xn;
                terms[4] = yn * yn;
                terms[5] = xn * yn;
            } else {
                terms[3] = xn * xn + yn * yn;
            }
            const double z = static_cast<double>(hrow[x]);
            for (int r = 0; r < nTerms; ++r) {
                rhs.at<double>(r, 0) += terms[r] * z;
                for (int c = 0; c < nTerms; ++c) {
                    normal.at<double>(r, c) += terms[r] * terms[c];
                }
            }
            ++count;
        }
    }

    if (count < nTerms) {
        return;
    }

    cv::Mat coeffs;
    if (!cv::solve(normal, rhs, coeffs, cv::DECOMP_CHOLESKY) &&
        !cv::solve(normal, rhs, coeffs, cv::DECOMP_SVD)) {
        return;
    }

    for (int y = 0; y < height.rows; ++y) {
        float* hrow = height.ptr<float>(y);
        const uchar* mrow = validMask.ptr<uchar>(y);
        const double yn = (static_cast<double>(y) - cy) / scale;
        for (int x = 0; x < height.cols; ++x) {
            if (mrow[x] == 0 || !std::isfinite(hrow[x])) {
                continue;
            }
            const double xn = (static_cast<double>(x) - cx) / scale;
            terms[0] = 1.0;
            terms[1] = xn;
            terms[2] = yn;
            if (mode == HeightFlattenMode::Quadratic) {
                terms[3] = xn * xn;
                terms[4] = yn * yn;
                terms[5] = xn * yn;
            } else {
                terms[3] = xn * xn + yn * yn;
            }

            double trend = 0.0;
            for (int i = 0; i < nTerms; ++i) {
                trend += coeffs.at<double>(i, 0) * terms[i];
            }
            hrow[x] = static_cast<float>(static_cast<double>(hrow[x]) - trend);
        }
    }
}
