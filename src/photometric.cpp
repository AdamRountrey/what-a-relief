#include "photometric.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {

struct SubsetSolver {
    bool valid = false;
    int count = 0;
    float coeff[3][8] = {};
};

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

std::array<SubsetSolver, 256> buildSolvers(const std::vector<cv::Vec3f>& lights) {
    std::array<SubsetSolver, 256> solvers;
    const int n = static_cast<int>(lights.size());
    const int total = 1 << n;

    for (int mask = 0; mask < total; ++mask) {
        std::vector<int> indices;
        for (int i = 0; i < n; ++i) {
            if ((mask & (1 << i)) != 0) {
                indices.push_back(i);
            }
        }
        if (indices.size() < 3) {
            continue;
        }

        cv::Mat A(static_cast<int>(indices.size()), 3, CV_32F);
        for (int r = 0; r < static_cast<int>(indices.size()); ++r) {
            const cv::Vec3f& l = lights[indices[r]];
            A.at<float>(r, 0) = l[0];
            A.at<float>(r, 1) = l[1];
            A.at<float>(r, 2) = l[2];
        }

        cv::Mat inv;
        const double reciprocalCondition = cv::invert(A.t() * A, inv, cv::DECOMP_SVD);
        if (reciprocalCondition < 1.0e-6) {
            continue;
        }

        cv::Mat pinv = inv * A.t();
        SubsetSolver solver;
        solver.valid = true;
        solver.count = static_cast<int>(indices.size());
        for (int c = 0; c < static_cast<int>(indices.size()); ++c) {
            const int originalIndex = indices[c];
            for (int r = 0; r < 3; ++r) {
                solver.coeff[r][originalIndex] = pinv.at<float>(r, c);
            }
        }
        solvers[mask] = solver;
    }

    return solvers;
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

std::vector<cv::Vec3f> loadLightsFile(const std::string& path, size_t expectedCount) {
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
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream iss(line);
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        if (!(iss >> x >> y >> z)) {
            continue;
        }
        cv::Vec3f v(x, y, z);
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
    cv::Mat& normalMap,
    cv::Mat& albedo,
    cv::Mat& residual,
    cv::Mat& validMask) {
    const int rows = images[0].rows;
    const int cols = images[0].cols;
    const int n = static_cast<int>(images.size());
    const auto solvers = buildSolvers(lights);

    normalMap = cv::Mat(rows, cols, CV_32FC3, cv::Scalar(0, 0, 0));
    albedo = cv::Mat(rows, cols, CV_32F, cv::Scalar(0));
    residual = cv::Mat(rows, cols, CV_32F, cv::Scalar(0));
    validMask = cv::Mat(rows, cols, CV_8U, cv::Scalar(0));

    for (int y = 0; y < rows; ++y) {
        const uchar* maskRow = inputMask.ptr<uchar>(y);
        cv::Vec3f* normalRow = normalMap.ptr<cv::Vec3f>(y);
        float* albedoRow = albedo.ptr<float>(y);
        float* residualRow = residual.ptr<float>(y);
        uchar* validRow = validMask.ptr<uchar>(y);

        for (int x = 0; x < cols; ++x) {
            if (maskRow[x] == 0) {
                continue;
            }

            float intensity[8] = {};
            int subset = 0;
            for (int i = 0; i < n; ++i) {
                intensity[i] = images[i].at<float>(y, x);
                if (std::isfinite(intensity[i]) && intensity[i] > shadowThreshold) {
                    subset |= 1 << i;
                }
            }

            const SubsetSolver& solver = solvers[subset];
            if (!solver.valid) {
                continue;
            }

            cv::Vec3f g(0.0f, 0.0f, 0.0f);
            for (int r = 0; r < 3; ++r) {
                for (int i = 0; i < n; ++i) {
                    g[r] += solver.coeff[r][i] * intensity[i];
                }
            }

            const float rho = std::sqrt(g.dot(g));
            if (!std::isfinite(rho) || rho <= 1.0e-6f) {
                continue;
            }
            const cv::Vec3f normal = g / rho;
            if (normal[2] <= 0.0f) {
                continue;
            }

            double err = 0.0;
            for (int i = 0; i < n; ++i) {
                if ((subset & (1 << i)) == 0) {
                    continue;
                }
                const double d = static_cast<double>(lights[i].dot(g) - intensity[i]);
                err += d * d;
            }
            err = std::sqrt(err / static_cast<double>(solver.count));

            normalRow[x] = normal;
            albedoRow[x] = rho;
            residualRow[x] = static_cast<float>(err);
            validRow[x] = 255;
        }
    }
}

cv::Mat integrateHeight(const cv::Mat& normalMap, const cv::Mat& validMask, int iterations) {
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

    cv::Mat z(rows, cols, CV_32F, cv::Scalar(0));
    cv::Mat next = z.clone();
    for (int iter = 0; iter < iterations; ++iter) {
        double sum = 0.0;
        int countValid = 0;
        for (int y = 0; y < rows; ++y) {
            const uchar* mrow = validMask.ptr<uchar>(y);
            const float* zrow = z.ptr<float>(y);
            float* out = next.ptr<float>(y);
            for (int x = 0; x < cols; ++x) {
                if (mrow[x] == 0) {
                    out[x] = 0.0f;
                    continue;
                }
                double acc = 0.0;
                int terms = 0;
                if (x > 0 && mrow[x - 1] != 0) {
                    acc += zrow[x - 1] + p.at<float>(y, x - 1);
                    ++terms;
                }
                if (x + 1 < cols && mrow[x + 1] != 0) {
                    acc += zrow[x + 1] - p.at<float>(y, x);
                    ++terms;
                }
                if (y > 0 && validMask.at<uchar>(y - 1, x) != 0) {
                    acc += z.at<float>(y - 1, x) + q.at<float>(y - 1, x);
                    ++terms;
                }
                if (y + 1 < rows && validMask.at<uchar>(y + 1, x) != 0) {
                    acc += z.at<float>(y + 1, x) - q.at<float>(y, x);
                    ++terms;
                }
                out[x] = terms > 0 ? static_cast<float>(acc / static_cast<double>(terms)) : 0.0f;
                sum += out[x];
                ++countValid;
            }
        }

        if (countValid > 0 && (iter % 25 == 24 || iter + 1 == iterations)) {
            const float mean = static_cast<float>(sum / static_cast<double>(countValid));
            for (int y = 0; y < rows; ++y) {
                const uchar* mrow = validMask.ptr<uchar>(y);
                float* out = next.ptr<float>(y);
                for (int x = 0; x < cols; ++x) {
                    if (mrow[x] != 0) {
                        out[x] -= mean;
                    }
                }
            }
        }
        std::swap(z, next);
    }
    return z;
}
