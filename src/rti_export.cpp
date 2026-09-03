#include "rti_export.hpp"
#include "checked_io.hpp"
#include "radiometry.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

void validateRtiDestination(const fs::path& destination, const Options& opt) {
    if (!fs::exists(destination)) {
        return;
    }
    if (!fs::is_directory(destination) || fs::is_symlink(destination) ||
        fs::exists(destination / "run_manifest.json") ||
        (!fs::is_empty(destination) && !fs::is_regular_file(destination / "rti_manifest.json"))) {
        die("RTI output must be an empty folder or an existing What A Relief RTI package: " + destination.string());
    }
    std::vector<std::string> inputs = opt.imagePaths;
    inputs.insert(inputs.end(), {opt.lightsFile, opt.maskPath, opt.heightMaskPath, opt.neuralModelPath});
    for (const std::string& input : inputs) {
        if (input.empty()) {
            continue;
        }
        fs::path current = fs::absolute(input).lexically_normal();
        while (!current.empty()) {
            std::error_code error;
            if (fs::equivalent(current, destination, error) && !error) {
                die("RTI output folder contains a selected input; choose a separate folder: " + destination.string());
            }
            const fs::path parent = current.parent_path();
            if (parent == current) {
                break;
            }
            current = parent;
        }
    }
}

class RtiDirectoryTransaction {
public:
    explicit RtiDirectoryTransaction(const fs::path& finalPath)
        : finalPath_(finalPath),
          stagingPath_(makeUniqueSiblingPath(finalPath, "staging")) {
        fs::create_directories(stagingPath_);
    }

    ~RtiDirectoryTransaction() {
        if (!committed_) {
            std::error_code ignored;
            fs::remove_all(stagingPath_, ignored);
        }
    }

    const fs::path& stagingPath() const {
        return stagingPath_;
    }

    void commit() {
        replaceDirectoryTransactionally(stagingPath_, finalPath_);
        committed_ = true;
    }

private:
    fs::path finalPath_;
    fs::path stagingPath_;
    bool committed_ = false;
};

cv::Mat loadRtiColorImage(const std::string& path, const cv::Size& expectedSize, bool srgb) {
    cv::Mat raw = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (raw.empty()) {
        die("Failed to read RTI image: " + path);
    }
    if (raw.size() != expectedSize) {
        die("All RTI export images must match the solve dimensions. Mismatch at: " + path);
    }

    cv::Mat bgr;
    if (raw.channels() == 1) {
        cv::cvtColor(raw, bgr, cv::COLOR_GRAY2BGR);
    } else if (raw.channels() == 4) {
        cv::cvtColor(raw, bgr, cv::COLOR_BGRA2BGR);
    } else if (raw.channels() == 3) {
        bgr = raw;
    } else {
        die("Unsupported channel count for RTI image: " + path);
    }

    return convertToLinearColor(bgr, srgb);
}

bool solveSymmetric6x6(
    std::array<std::array<double, 6>, 6> a,
    std::array<double, 6> b,
    std::array<double, 6>& x,
    int basisCount) {
    for (int col = 0; col < basisCount; ++col) {
        int pivot = col;
        double best = std::abs(a[col][col]);
        for (int row = col + 1; row < basisCount; ++row) {
            const double candidate = std::abs(a[row][col]);
            if (candidate > best) {
                best = candidate;
                pivot = row;
            }
        }
        if (best < 1.0e-12) {
            return false;
        }
        if (pivot != col) {
            std::swap(a[pivot], a[col]);
            std::swap(b[pivot], b[col]);
        }
        const double inv = 1.0 / a[col][col];
        for (int j = col; j < basisCount; ++j) {
            a[col][j] *= inv;
        }
        b[col] *= inv;
        for (int row = 0; row < basisCount; ++row) {
            if (row == col) {
                continue;
            }
            const double factor = a[row][col];
            if (factor == 0.0) {
                continue;
            }
            for (int j = col; j < basisCount; ++j) {
                a[row][j] -= factor * a[col][j];
            }
            b[row] -= factor * b[col];
        }
    }
    x = b;
    return true;
}

std::array<double, 6> ptmBasis(const cv::Vec3f& light) {
    const double u = static_cast<double>(light[0]);
    const double v = static_cast<double>(light[1]);
    return {1.0, u, v, u * u, u * v, v * v};
}

std::array<std::array<double, 6>, 6> buildNormalMatrix(const std::vector<cv::Vec3f>& lights, int basisCount) {
    std::array<std::array<double, 6>, 6> normal{};
    for (const cv::Vec3f& light : lights) {
        const std::array<double, 6> basis = ptmBasis(light);
        for (int r = 0; r < basisCount; ++r) {
            for (int c = 0; c < basisCount; ++c) {
                normal[r][c] += basis[r] * basis[c];
            }
        }
    }
    return normal;
}

bool normalMatrixIsStable(std::array<std::array<double, 6>, 6> a, int basisCount) {
    double maxDiagonal = 0.0;
    for (int i = 0; i < basisCount; ++i) {
        maxDiagonal = std::max(maxDiagonal, std::abs(a[i][i]));
    }
    if (maxDiagonal <= 0.0) {
        return false;
    }

    double minPivot = std::numeric_limits<double>::infinity();
    for (int col = 0; col < basisCount; ++col) {
        int pivot = col;
        double best = std::abs(a[col][col]);
        for (int row = col + 1; row < basisCount; ++row) {
            const double candidate = std::abs(a[row][col]);
            if (candidate > best) {
                best = candidate;
                pivot = row;
            }
        }
        minPivot = std::min(minPivot, best);
        if (best < maxDiagonal * 1.0e-8) {
            return false;
        }
        if (pivot != col) {
            std::swap(a[pivot], a[col]);
        }
        const double inv = 1.0 / a[col][col];
        for (int j = col; j < basisCount; ++j) {
            a[col][j] *= inv;
        }
        for (int row = col + 1; row < basisCount; ++row) {
            const double factor = a[row][col];
            for (int j = col; j < basisCount; ++j) {
                a[row][j] -= factor * a[col][j];
            }
        }
    }
    return minPivot > maxDiagonal * 1.0e-7;
}

std::array<double, 6> solvePixelCoefficients(
    const std::vector<cv::Mat>& images,
    const std::vector<std::array<double, 6>>& bases,
    const std::array<std::array<double, 6>, 6>& normal,
    int basisCount,
    int x,
    int y,
    int channel) {
    std::array<double, 6> rhs{};
    for (size_t i = 0; i < images.size(); ++i) {
        const cv::Vec3f bgr = images[i].ptr<cv::Vec3f>(y)[x];
        const double value = static_cast<double>(bgr[2 - channel]);
        for (int k = 0; k < basisCount; ++k) {
            rhs[k] += bases[i][k] * value;
        }
    }

    std::array<double, 6> coeffs{};
    if (!solveSymmetric6x6(normal, rhs, coeffs, basisCount)) {
        coeffs[0] = static_cast<double>(images[0].ptr<cv::Vec3f>(y)[x][2 - channel]);
    }
    return coeffs;
}

double averageLuminance(const cv::Vec3f& bgr) {
    return (static_cast<double>(bgr[0]) + static_cast<double>(bgr[1]) + static_cast<double>(bgr[2])) / 3.0;
}

double lrgbBaseWeight(const cv::Vec3f& light) {
    const double z = std::clamp(static_cast<double>(light[2]), 0.0, 1.0);
    const double elevationUnit = std::asin(z) / (std::acos(-1.0) * 0.5);
    return std::max(0.0, 0.25 - std::pow(elevationUnit - 0.5, 2.0));
}

cv::Mat buildLrgbBaseImage(const std::vector<cv::Mat>& images, const std::vector<cv::Vec3f>& lights) {
    cv::Mat base(images.front().size(), CV_32FC3, cv::Scalar(0, 0, 0));
    std::vector<double> weights;
    weights.reserve(lights.size());
    double totalWeight = 0.0;
    for (const cv::Vec3f& light : lights) {
        const double weight = lrgbBaseWeight(light);
        weights.push_back(weight);
        totalWeight += weight;
    }
    if (totalWeight <= 1.0e-12) {
        std::fill(weights.begin(), weights.end(), 1.0);
        totalWeight = static_cast<double>(weights.size());
    }

    for (size_t i = 0; i < images.size(); ++i) {
        base += images[i] * static_cast<float>(weights[i] / totalWeight);
    }
    return base;
}

std::array<double, 6> solvePixelLuminanceCoefficients(
    const std::vector<cv::Mat>& images,
    const cv::Mat& baseImage,
    const std::vector<std::array<double, 6>>& bases,
    const std::array<std::array<double, 6>, 6>& normal,
    int basisCount,
    int x,
    int y) {
    const cv::Vec3f base = baseImage.ptr<cv::Vec3f>(y)[x];
    const double baseLum = std::max(averageLuminance(base), 1.0 / 65535.0);

    std::array<double, 6> rhs{};
    for (size_t i = 0; i < images.size(); ++i) {
        const double value = averageLuminance(images[i].ptr<cv::Vec3f>(y)[x]) / baseLum;
        for (int k = 0; k < basisCount; ++k) {
            rhs[k] += bases[i][k] * value;
        }
    }

    std::array<double, 6> coeffs{};
    if (!solveSymmetric6x6(normal, rhs, coeffs, basisCount)) {
        coeffs[0] = averageLuminance(images[0].ptr<cv::Vec3f>(y)[x]) / baseLum;
    }
    return coeffs;
}

std::pair<double, double> robustRange(const cv::Mat& coeff, int bgr) {
    std::vector<float> values;
    values.reserve(static_cast<size_t>(coeff.rows) * static_cast<size_t>(coeff.cols));
    for (int y = 0; y < coeff.rows; ++y) {
        const cv::Vec3f* row = coeff.ptr<cv::Vec3f>(y);
        for (int x = 0; x < coeff.cols; ++x) {
            const float value = row[x][bgr];
            if (std::isfinite(value)) {
                values.push_back(value);
            }
        }
    }
    if (values.empty()) {
        return {0.0, 1.0};
    }

    const size_t lowIndex = std::min(values.size() - 1, static_cast<size_t>(std::floor(values.size() * 0.005)));
    const size_t highIndex = std::min(values.size() - 1, static_cast<size_t>(std::floor(values.size() * 0.995)));
    std::nth_element(values.begin(), values.begin() + lowIndex, values.end());
    const double low = values[lowIndex];
    std::nth_element(values.begin(), values.begin() + highIndex, values.end());
    const double high = values[highIndex];

    if (!std::isfinite(low) || !std::isfinite(high) || high <= low) {
        return {0.0, 1.0};
    }
    return {low, high};
}

cv::Mat encodeCoefficientJpeg(const cv::Mat& coeff, int coefficientIndex, std::vector<double>& scales, std::vector<double>& biases) {
    cv::Mat encoded(coeff.size(), CV_8UC3);
    for (int rgb = 0; rgb < 3; ++rgb) {
        const int bgr = 2 - rgb;
        const auto [minValue, maxValue] = robustRange(coeff, bgr);
        const double scale = maxValue - minValue;
        const double bias = -minValue / scale;
        scales[coefficientIndex * 3 + rgb] = scale;
        biases[coefficientIndex * 3 + rgb] = bias;

        for (int y = 0; y < coeff.rows; ++y) {
            const cv::Vec3f* src = coeff.ptr<cv::Vec3f>(y);
            cv::Vec3b* dst = encoded.ptr<cv::Vec3b>(y);
            for (int x = 0; x < coeff.cols; ++x) {
                const double unit = (static_cast<double>(src[x][bgr]) - minValue) / scale;
                dst[x][bgr] = static_cast<uchar>(std::clamp(std::round(unit * 255.0), 0.0, 255.0));
            }
        }
    }
    return encoded;
}

cv::Mat encodeLrgbBaseJpeg(const cv::Mat& baseImage) {
    cv::Mat encoded(baseImage.size(), CV_8UC3);
    for (int y = 0; y < baseImage.rows; ++y) {
        const cv::Vec3f* src = baseImage.ptr<cv::Vec3f>(y);
        cv::Vec3b* dst = encoded.ptr<cv::Vec3b>(y);
        for (int x = 0; x < baseImage.cols; ++x) {
            for (int bgr = 0; bgr < 3; ++bgr) {
                dst[x][bgr] = static_cast<uchar>(std::clamp(std::round(src[x][bgr] * 255.0f), 0.0f, 255.0f));
            }
        }
    }
    return encoded;
}

std::pair<double, double> robustRangeFloat(const cv::Mat& coeff) {
    std::vector<float> values;
    values.reserve(static_cast<size_t>(coeff.rows) * static_cast<size_t>(coeff.cols));
    for (int y = 0; y < coeff.rows; ++y) {
        const float* row = coeff.ptr<float>(y);
        for (int x = 0; x < coeff.cols; ++x) {
            const float value = row[x];
            if (std::isfinite(value)) {
                values.push_back(value);
            }
        }
    }
    if (values.empty()) {
        return {0.0, 1.0};
    }

    const size_t lowIndex = std::min(values.size() - 1, static_cast<size_t>(std::floor(values.size() * 0.005)));
    const size_t highIndex = std::min(values.size() - 1, static_cast<size_t>(std::floor(values.size() * 0.995)));
    std::nth_element(values.begin(), values.begin() + lowIndex, values.end());
    const double low = values[lowIndex];
    std::nth_element(values.begin(), values.begin() + highIndex, values.end());
    const double high = values[highIndex];

    if (!std::isfinite(low) || !std::isfinite(high) || high <= low) {
        return {0.0, 1.0};
    }
    return {low, high};
}

cv::Mat encodeLrgbCoefficientJpeg(
    const std::vector<cv::Mat>& coeffImages,
    int groupIndex,
    std::vector<double>& scales,
    std::vector<double>& biases) {
    cv::Mat encoded(coeffImages.front().size(), CV_8UC3, cv::Scalar(0, 0, 0));
    for (int rgb = 0; rgb < 3; ++rgb) {
        const int coeffIndex = groupIndex * 3 + rgb;
        if (coeffIndex >= static_cast<int>(coeffImages.size())) {
            continue;
        }
        const int bgr = 2 - rgb;
        const int planeIndex = 3 + coeffIndex;
        const auto [minValue, maxValue] = robustRangeFloat(coeffImages[static_cast<size_t>(coeffIndex)]);
        const double scale = maxValue - minValue;
        const double bias = -minValue / scale;
        scales[static_cast<size_t>(planeIndex)] = scale;
        biases[static_cast<size_t>(planeIndex)] = bias;

        for (int y = 0; y < encoded.rows; ++y) {
            const float* src = coeffImages[static_cast<size_t>(coeffIndex)].ptr<float>(y);
            cv::Vec3b* dst = encoded.ptr<cv::Vec3b>(y);
            for (int x = 0; x < encoded.cols; ++x) {
                const double unit = (static_cast<double>(src[x]) - minValue) / scale;
                dst[x][bgr] = static_cast<uchar>(std::clamp(std::round(unit * 255.0), 0.0, 255.0));
            }
        }
    }
    return encoded;
}

int deepZoomLevelCount(const cv::Size& size) {
    const int maximumDimension = std::max(size.width, size.height);
    return std::max(1, static_cast<int>(std::ceil(std::log2(std::max(1, maximumDimension)))) + 1);
}

cv::Size deepZoomLevelSize(const cv::Size& original, int level, int levelCount) {
    const double divisor = std::pow(2.0, static_cast<double>(levelCount - 1 - level));
    return cv::Size(
        std::max(1, static_cast<int>(std::ceil(original.width / divisor))),
        std::max(1, static_cast<int>(std::ceil(original.height / divisor))));
}

void writeDeepZoomPyramid(const cv::Mat& image, const fs::path& basePath, int quality) {
    const int tileSize = 256;
    const int overlap = 0;
    const int levelCount = deepZoomLevelCount(image.size());
    const fs::path filesDir = basePath.string() + "_files";
    fs::remove_all(filesDir);
    fs::create_directories(filesDir);

    {
        CheckedOutputFile checked(basePath.string() + ".dzi");
        std::ostream& dzi = checked.stream();
        dzi << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
        dzi << "<Image TileSize=\"" << tileSize << "\" Overlap=\"" << overlap
            << "\" Format=\"jpg\" xmlns=\"http://schemas.microsoft.com/deepzoom/2008\">\n";
        dzi << "  <Size Width=\"" << image.cols << "\" Height=\"" << image.rows << "\"/>\n";
        dzi << "</Image>\n";
        checked.commit();
    }

    const std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, quality};
    for (int level = 0; level < levelCount; ++level) {
        const cv::Size levelSize = deepZoomLevelSize(image.size(), level, levelCount);
        cv::Mat levelImage;
        if (levelSize == image.size()) {
            levelImage = image;
        } else {
            cv::resize(image, levelImage, levelSize, 0.0, 0.0, cv::INTER_AREA);
        }
        const fs::path levelDir = filesDir / std::to_string(level);
        fs::create_directories(levelDir);
        const int tilesX = (levelImage.cols + tileSize - 1) / tileSize;
        const int tilesY = (levelImage.rows + tileSize - 1) / tileSize;
        for (int ty = 0; ty < tilesY; ++ty) {
            for (int tx = 0; tx < tilesX; ++tx) {
                const cv::Rect roi(
                    tx * tileSize,
                    ty * tileSize,
                    std::min(tileSize, levelImage.cols - tx * tileSize),
                    std::min(tileSize, levelImage.rows - ty * tileSize));
                writeImageChecked(
                    levelDir / (std::to_string(tx) + "_" + std::to_string(ty) + ".jpg"),
                    levelImage(roi),
                    params);
            }
        }
    }
}

void writeInfoJson(
    const fs::path& outputDir,
    const cv::Size& size,
    const std::vector<cv::Vec3f>& lights,
    const std::vector<double>& scales,
    const std::vector<double>& biases,
    RtiColorMode colorMode,
    int quality) {
    CheckedOutputFile checked(outputDir / "info.json");
    std::ostream& out = checked.stream();
    out << std::setprecision(8);
    out << "{\n";
    out << "\"width\": " << size.width << ", \"height\": " << size.height << ",\n";
    out << "\"format\": \"jpg\",\n";
    out << "\"type\":\"ptm\",\n";
    out << "\"colorspace\":\"" << (colorMode == RtiColorMode::Lrgb ? "lrgb" : "rgb") << "\",\n";
    out << "\"lights\": [";
    for (size_t i = 0; i < lights.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << lights[i][0] << ", " << lights[i][1] << ", " << lights[i][2];
    }
    out << "],\n";
    out << "\"nplanes\": " << scales.size() << ",\n";
    out << "\"quality\": " << quality << ",\n";
    out << "\"materials\": [\n";
    out << "{\n \"scale\": [";
    for (size_t i = 0; i < scales.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        out << scales[i];
    }
    out << "],\n \"bias\": [";
    for (size_t i = 0; i < biases.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        out << biases[i];
    }
    out << "] }\n";
    out << "]\n";
    out << "}\n";
    checked.commit();
}

void removePlaneFiles(const fs::path& outputDir, int planeIndex) {
    fs::remove(outputDir / ("plane_" + std::to_string(planeIndex) + ".jpg"));
    fs::remove(outputDir / ("plane_" + std::to_string(planeIndex) + ".dzi"));
    fs::remove_all(outputDir / ("plane_" + std::to_string(planeIndex) + "_files"));
}

void cleanupStalePlaneFiles(const fs::path& outputDir, int firstStalePlane) {
    for (int p = firstStalePlane; p < 12; ++p) {
        removePlaneFiles(outputDir, p);
    }
}

int webRtiInternalCoeffIndex(int webIndex) {
    constexpr std::array<int, 6> webToInternal = {3, 5, 4, 1, 2, 0};
    return webToInternal[static_cast<size_t>(webIndex)];
}

bool isWebRtiTileName(const std::string& name) {
    const size_t underscore = name.find('_');
    const size_t dot = name.rfind('.');
    if (underscore == std::string::npos || dot == std::string::npos || underscore == 0 || dot <= underscore + 1) {
        return false;
    }
    const std::string ext = name.substr(dot);
    if (ext != ".jpg" && ext != ".jpeg" && ext != ".png") {
        return false;
    }
    for (size_t i = 0; i < dot; ++i) {
        if (i == underscore) {
            continue;
        }
        if (!std::isdigit(static_cast<unsigned char>(name[i]))) {
            return false;
        }
    }
    return true;
}

void cleanupWebRtiFiles(const fs::path& outputDir) {
    fs::remove(outputDir / "info.xml");
    if (!fs::exists(outputDir)) {
        return;
    }
    for (const fs::directory_entry& entry : fs::directory_iterator(outputDir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        if (isWebRtiTileName(entry.path().filename().string())) {
            fs::remove(entry.path());
        }
    }
}

std::pair<double, double> robustRangeAllChannels(const cv::Mat& coeff) {
    std::vector<float> values;
    values.reserve(static_cast<size_t>(coeff.rows) * static_cast<size_t>(coeff.cols) * static_cast<size_t>(coeff.channels()));
    if (coeff.channels() == 1) {
        for (int y = 0; y < coeff.rows; ++y) {
            const float* row = coeff.ptr<float>(y);
            for (int x = 0; x < coeff.cols; ++x) {
                const float value = row[x];
                if (std::isfinite(value)) {
                    values.push_back(value);
                }
            }
        }
    } else {
        for (int y = 0; y < coeff.rows; ++y) {
            const cv::Vec3f* row = coeff.ptr<cv::Vec3f>(y);
            for (int x = 0; x < coeff.cols; ++x) {
                for (int bgr = 0; bgr < 3; ++bgr) {
                    const float value = row[x][bgr];
                    if (std::isfinite(value)) {
                        values.push_back(value);
                    }
                }
            }
        }
    }
    if (values.empty()) {
        return {0.0, 1.0};
    }

    const size_t lowIndex = std::min(values.size() - 1, static_cast<size_t>(std::floor(values.size() * 0.005)));
    const size_t highIndex = std::min(values.size() - 1, static_cast<size_t>(std::floor(values.size() * 0.995)));
    std::nth_element(values.begin(), values.begin() + lowIndex, values.end());
    const double low = values[lowIndex];
    std::nth_element(values.begin(), values.begin() + highIndex, values.end());
    const double high = values[highIndex];
    if (!std::isfinite(low) || !std::isfinite(high) || high <= low) {
        return {0.0, 1.0};
    }
    return {low, high};
}

cv::Mat encodeWebRtiRgbCoefficient(const cv::Mat& coeff, double& scale, double& bias) {
    if (coeff.empty()) {
        scale = 1.0;
        bias = 0.0;
        return {};
    }
    const auto [minValue, maxValue] = robustRangeAllChannels(coeff);
    scale = maxValue - minValue;
    bias = -minValue / scale * 255.0;
    cv::Mat encoded(coeff.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    for (int y = 0; y < coeff.rows; ++y) {
        const cv::Vec3f* src = coeff.ptr<cv::Vec3f>(y);
        cv::Vec3b* dst = encoded.ptr<cv::Vec3b>(y);
        for (int x = 0; x < coeff.cols; ++x) {
            for (int bgr = 0; bgr < 3; ++bgr) {
                const double unit = (static_cast<double>(src[x][bgr]) - minValue) / scale;
                dst[x][bgr] = static_cast<uchar>(std::clamp(std::round(unit * 255.0), 0.0, 255.0));
            }
        }
    }
    return encoded;
}

cv::Mat encodeWebRtiLrgbGroup(
    const std::vector<cv::Mat>& coeffImages,
    int groupIndex,
    int basisCount,
    std::vector<double>& scales,
    std::vector<double>& biases,
    const cv::Size& size) {
    cv::Mat encoded(size, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int rgb = 0; rgb < 3; ++rgb) {
        const int webCoeff = groupIndex * 3 + rgb;
        const int internalCoeff = webRtiInternalCoeffIndex(webCoeff);
        const int bgr = 2 - rgb;
        if (internalCoeff >= basisCount) {
            scales[static_cast<size_t>(webCoeff)] = 1.0;
            biases[static_cast<size_t>(webCoeff)] = 0.0;
            continue;
        }

        const cv::Mat& coeff = coeffImages[static_cast<size_t>(internalCoeff)];
        const auto [minValue, maxValue] = robustRangeFloat(coeff);
        const double scale = maxValue - minValue;
        const double bias = -minValue / scale * 255.0;
        scales[static_cast<size_t>(webCoeff)] = scale;
        biases[static_cast<size_t>(webCoeff)] = bias;
        for (int y = 0; y < encoded.rows; ++y) {
            const float* src = coeff.ptr<float>(y);
            cv::Vec3b* dst = encoded.ptr<cv::Vec3b>(y);
            for (int x = 0; x < encoded.cols; ++x) {
                const double unit = (static_cast<double>(src[x]) - minValue) / scale;
                dst[x][bgr] = static_cast<uchar>(std::clamp(std::round(unit * 255.0), 0.0, 255.0));
            }
        }
    }
    return encoded;
}

int nextPowerOfTwo(int value) {
    int result = 1;
    while (result < value) {
        result *= 2;
    }
    return result;
}

struct WebRtiNode {
    int index = 0;
    int parent = -1;
    int level = 0;
    cv::Rect rect;
    std::array<int, 4> children = {-1, -1, -1, -1};
    std::array<double, 4> box = {0.0, 0.0, 1.0, 1.0};
    bool valid = false;
};

bool intersectsStrict(const cv::Rect& a, const cv::Rect& b) {
    return (a & b).area() > 0;
}

std::vector<WebRtiNode> buildWebRtiNodes(const cv::Size& imageSize, int tileSize, int maxSize) {
    int levelCount = 1;
    for (int side = maxSize; side > tileSize; side /= 2) {
        ++levelCount;
    }
    int nodeCount = 0;
    int levelNodes = 1;
    for (int level = 0; level < levelCount; ++level) {
        nodeCount += levelNodes;
        levelNodes *= 4;
    }

    const cv::Rect imageRect(
        (maxSize - imageSize.width) / 2,
        (maxSize - imageSize.height) / 2,
        imageSize.width,
        imageSize.height);
    std::vector<WebRtiNode> nodes(static_cast<size_t>(nodeCount));
    nodes[0].index = 0;
    nodes[0].parent = -1;
    nodes[0].level = 0;
    nodes[0].rect = cv::Rect(0, 0, maxSize, maxSize);
    nodes[0].box = {0.0, 0.0, 1.0, 1.0};
    nodes[0].valid = true;

    for (int index = 1; index < nodeCount; ++index) {
        const int parentIndex = (index - 1) / 4;
        WebRtiNode& parent = nodes[static_cast<size_t>(parentIndex)];
        const int halfW = parent.rect.width / 2;
        const int halfH = parent.rect.height / 2;
        cv::Rect rect = parent.rect;
        std::array<double, 4> box = parent.box;
        const double halfBoxW = (parent.box[2] - parent.box[0]) * 0.5;
        const double halfBoxH = (parent.box[3] - parent.box[1]) * 0.5;
        box[2] = box[0] + halfBoxW;
        box[3] = box[1] + halfBoxH;
        const int childSlot = index - (4 * parentIndex + 1);
        const int t = index % 4;
        if (t == 1) {
            rect = cv::Rect(parent.rect.x, parent.rect.y, halfW, halfH);
            box[1] = parent.box[1] + halfBoxH;
            box[3] = parent.box[3];
        } else if (t == 2) {
            rect = cv::Rect(parent.rect.x + halfW, parent.rect.y, halfW, halfH);
            box[0] = parent.box[0] + halfBoxW;
            box[1] = parent.box[1] + halfBoxH;
            box[2] = parent.box[2];
            box[3] = parent.box[3];
        } else if (t == 3) {
            rect = cv::Rect(parent.rect.x, parent.rect.y + halfH, halfW, halfH);
        } else {
            rect = cv::Rect(parent.rect.x + halfW, parent.rect.y + halfH, halfW, halfH);
            box[0] = parent.box[0] + halfBoxW;
            box[2] = parent.box[2];
        }
        WebRtiNode& node = nodes[static_cast<size_t>(index)];
        node.index = index;
        node.parent = parentIndex;
        node.level = parent.level + 1;
        node.rect = rect;
        node.box = box;
        node.valid = intersectsStrict(rect, imageRect);
        parent.children[static_cast<size_t>(childSlot)] = index;
    }
    int maxLevel = 0;
    for (const WebRtiNode& node : nodes) {
        maxLevel = std::max(maxLevel, node.level);
    }
    for (WebRtiNode& node : nodes) {
        const int pad = 1 << (maxLevel - node.level);
        node.rect = cv::Rect(node.rect.x - pad, node.rect.y - pad, node.rect.width + 2 * pad, node.rect.height + 2 * pad);
    }
    return nodes;
}

cv::Mat cropWithBlackBorder(const cv::Mat& image, const cv::Rect& rect) {
    cv::Mat tile(rect.height, rect.width, image.type(), cv::Scalar(0, 0, 0));
    const cv::Rect imageRect(0, 0, image.cols, image.rows);
    const cv::Rect clipped = rect & imageRect;
    if (clipped.area() <= 0) {
        return tile;
    }
    const cv::Rect dstRect(clipped.x - rect.x, clipped.y - rect.y, clipped.width, clipped.height);
    image(clipped).copyTo(tile(dstRect));
    return tile;
}

void writeWebRtiLayerTiles(
    const cv::Mat& layer,
    const fs::path& outputDir,
    int layerIndex,
    const std::vector<WebRtiNode>& nodes,
    const cv::Size& imageSize,
    int tileSize,
    int maxSize,
    int quality) {
    const int offsetX = (maxSize - imageSize.width) / 2;
    const int offsetY = (maxSize - imageSize.height) / 2;
    cv::Mat square(maxSize, maxSize, CV_8UC3, cv::Scalar(0, 0, 0));
    layer.copyTo(square(cv::Rect(offsetX, offsetY, imageSize.width, imageSize.height)));

    const std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, quality};
    for (const WebRtiNode& node : nodes) {
        if (!node.valid) {
            continue;
        }
        cv::Mat cropped = cropWithBlackBorder(square, node.rect);
        cv::Mat tile;
        cv::resize(cropped, tile, cv::Size(tileSize + 2, tileSize + 2), 0.0, 0.0, cv::INTER_AREA);
        const std::string name = std::to_string(node.index + 1) + "_" + std::to_string(layerIndex) + ".jpg";
        writeImageChecked(outputDir / name, tile, params);
    }
}

void writeWebRtiInfoXml(
    const fs::path& outputDir,
    const cv::Size& size,
    int maxSize,
    int tileSize,
    const std::vector<WebRtiNode>& nodes,
    RtiColorMode colorMode,
    const std::vector<double>& scales,
    const std::vector<double>& biases) {
    CheckedOutputFile checked(outputDir / "info.xml");
    std::ostream& out = checked.stream();
    out << std::setprecision(8);
    out << "<!DOCTYPE Doc>\n";
    out << "<MultiRes format=\"0\">\n";
    out << "<Content type=\"" << (colorMode == RtiColorMode::Lrgb ? "LRGB_PTM" : "RGB_PTM") << "\">\n";
    out << "<Size width=\"" << size.width << "\" height=\"" << size.height << "\" coefficients=\"6\"/>\n";
    out << "<Scale>";
    for (double scale : scales) {
        out << scale << " ";
    }
    out << "</Scale>\n";
    out << "<Bias>";
    for (double bias : biases) {
        out << bias << " ";
    }
    out << "</Bias>\n";
    out << "</Content>\n";
    out << "<Tree>" << nodes.size() << " 0\n";
    out << tileSize << "\n";
    out << maxSize << " " << maxSize << " 255\n";
    out << "0 0 0\n";
    for (const WebRtiNode& node : nodes) {
        out << (node.index + 1) << " " << node.parent << " ";
        for (int j = 0; j < 4; ++j) {
            out << node.children[static_cast<size_t>((j + 2) % 4)] << " ";
        }
        out << tileSize << " ";
        out << (node.valid ? 1 : 0) << " ";
        out << node.box[0] << " " << node.box[1] << " 0 ";
        out << node.box[2] << " " << node.box[3] << " 1\n";
    }
    out << "</Tree>\n";
    out << "</MultiRes>\n";
    checked.commit();
}

void writeWebRtiViewerPackage(
    const fs::path& outputDir,
    const cv::Size& size,
    RtiColorMode colorMode,
    const std::vector<cv::Mat>& layers,
    const std::vector<double>& scales,
    const std::vector<double>& biases,
    const std::function<void(const std::string&)>& progress) {
    constexpr int tileSize = 256;
    constexpr int quality = 95;
    const int maxSize = nextPowerOfTwo(std::max(size.width, size.height));
    const std::vector<WebRtiNode> nodes = buildWebRtiNodes(size, tileSize, maxSize);

    cleanupWebRtiFiles(outputDir);
    writeWebRtiInfoXml(outputDir, size, maxSize, tileSize, nodes, colorMode, scales, biases);
    for (size_t i = 0; i < layers.size(); ++i) {
        reportProgress(progress, "RTI: writing webRTIViewer tile layer " + std::to_string(i + 1) + " of " + std::to_string(layers.size()) + "...");
        writeWebRtiLayerTiles(layers[i], outputDir, static_cast<int>(i + 1), nodes, size, tileSize, maxSize, quality);
    }
}

const char* rtiLayoutName(RtiLayoutMode mode) {
    switch (mode) {
    case RtiLayoutMode::DeepZoom:
        return "deepzoom";
    case RtiLayoutMode::WebRtiViewer:
        return "webrti";
    case RtiLayoutMode::Image:
    default:
        return "image";
    }
}

void writeRtiPackageManifest(
    const fs::path& outputDir,
    const Options& opt,
    const cv::Size& size,
    int basisCount) {
    std::uintmax_t fileCount = 0;
    std::uintmax_t byteCount = 0;
    for (const fs::directory_entry& entry : fs::recursive_directory_iterator(outputDir)) {
        if (entry.is_regular_file()) {
            ++fileCount;
            byteCount += entry.file_size();
        }
    }

    CheckedOutputFile checked(outputDir / "rti_manifest.json");
    std::ostream& out = checked.stream();
    out << "{\n";
    out << "  \"schema_version\": 1,\n";
    out << "  \"status\": \"complete\",\n";
    out << "  \"layout\": \"" << rtiLayoutName(opt.rtiLayoutMode) << "\",\n";
    out << "  \"color_mode\": \"" << (opt.rtiColorMode == RtiColorMode::Lrgb ? "lrgb" : "rgb") << "\",\n";
    out << "  \"width\": " << size.width << ",\n";
    out << "  \"height\": " << size.height << ",\n";
    out << "  \"source_images\": " << opt.imagePaths.size() << ",\n";
    out << "  \"ptm_basis_terms\": " << basisCount << ",\n";
    out << "  \"payload_files\": " << fileCount << ",\n";
    out << "  \"payload_bytes\": " << byteCount << "\n";
    out << "}\n";
    checked.commit();
}

} // namespace

void exportRtiPackage(
    const Options& opt,
    const std::vector<cv::Vec3f>& lights,
    const cv::Size& expectedSize,
    const std::function<void(const std::string&)>& progress) {
    if (!opt.exportRti) {
        return;
    }
    if (lights.size() != opt.imagePaths.size()) {
        die("RTI export requires one calibrated or loaded light direction per input image.");
    }
    if (lights.size() < 3) {
        die("RTI export requires at least 3 calibrated or loaded light directions.");
    }

    const fs::path finalOutputDir = opt.rtiPath.empty()
        ? (fs::path(opt.outputDir) / "rti")
        : fs::path(opt.rtiPath);
    validateRtiDestination(finalOutputDir, opt);
    RtiDirectoryTransaction transaction(finalOutputDir);
    const fs::path& outputDir = transaction.stagingPath();
    reportProgress(progress, "RTI: loading color images...");

    std::vector<cv::Mat> images;
    images.reserve(opt.imagePaths.size());
    for (const std::string& path : opt.imagePaths) {
        images.push_back(loadRtiColorImage(path, expectedSize, opt.srgb));
    }
    normalizeRelativeIntensityStack(images, false);

    std::vector<std::array<double, 6>> bases;
    bases.reserve(lights.size());
    for (const cv::Vec3f& light : lights) {
        bases.push_back(ptmBasis(light));
    }
    int basisCount = lights.size() < 9 ? 3 : 6;
    auto normal = buildNormalMatrix(lights, basisCount);
    if (basisCount == 3) {
        reportProgress(
            progress,
            "RTI: using stable 3-term PTM for small calibrated image stack.");
    } else if (!normalMatrixIsStable(normal, basisCount)) {
        basisCount = 3;
        normal = buildNormalMatrix(lights, basisCount);
        reportProgress(
            progress,
            "RTI: light geometry is ill-conditioned for quadratic PTM; using stable 3-term PTM.");
    }

    const auto finishPackage = [&]() {
        writeRtiPackageManifest(outputDir, opt, expectedSize, basisCount);
        transaction.commit();
        reportProgress(progress, "RTI: wrote " + finalOutputDir.string());
    };

    const int quality = 95;
    const std::vector<int> jpegParams = {cv::IMWRITE_JPEG_QUALITY, quality};

    if (opt.rtiColorMode == RtiColorMode::Lrgb) {
        reportProgress(progress, "RTI: fitting LRGB base image and luminance PTM...");
        const cv::Mat baseImage = buildLrgbBaseImage(images, lights);
        std::vector<cv::Mat> coeffImages(static_cast<size_t>(basisCount));
        for (cv::Mat& plane : coeffImages) {
            plane = cv::Mat(expectedSize, CV_32F, cv::Scalar(0));
        }
        for (int y = 0; y < expectedSize.height; ++y) {
            for (int x = 0; x < expectedSize.width; ++x) {
                const std::array<double, 6> coeffs =
                    solvePixelLuminanceCoefficients(images, baseImage, bases, normal, basisCount, x, y);
                for (int p = 0; p < basisCount; ++p) {
                    coeffImages[static_cast<size_t>(p)].ptr<float>(y)[x] = static_cast<float>(coeffs[p]);
                }
            }
        }

        if (opt.rtiLayoutMode == RtiLayoutMode::WebRtiViewer) {
            reportProgress(progress, "RTI: writing webRTIViewer LRGB component tiles...");
            std::vector<double> webScales(6, 1.0);
            std::vector<double> webBiases(6, 0.0);
            std::vector<cv::Mat> webLayers;
            webLayers.push_back(encodeWebRtiLrgbGroup(coeffImages, 0, basisCount, webScales, webBiases, expectedSize));
            webLayers.push_back(encodeWebRtiLrgbGroup(coeffImages, 1, basisCount, webScales, webBiases, expectedSize));
            webLayers.push_back(encodeLrgbBaseJpeg(baseImage));
            writeWebRtiViewerPackage(
                outputDir,
                expectedSize,
                opt.rtiColorMode,
                webLayers,
                webScales,
                webBiases,
                progress);
            finishPackage();
            return;
        }

        std::vector<double> scales(static_cast<size_t>(3 + basisCount), 1.0);
        std::vector<double> biases(static_cast<size_t>(3 + basisCount), 0.0);
        const int coefficientJpegs = (basisCount + 2) / 3;
        reportProgress(progress, opt.rtiLayoutMode == RtiLayoutMode::DeepZoom
            ? "RTI: writing DeepZoom LRGB pyramids..."
            : "RTI: writing LRGB images...");
        for (int p = 0; p <= coefficientJpegs; ++p) {
            const cv::Mat encoded = p == 0
                ? encodeLrgbBaseJpeg(baseImage)
                : encodeLrgbCoefficientJpeg(coeffImages, p - 1, scales, biases);
            const fs::path planePath = outputDir / ("plane_" + std::to_string(p) + ".jpg");
            if (opt.rtiLayoutMode == RtiLayoutMode::DeepZoom) {
                fs::remove(planePath);
                writeDeepZoomPyramid(encoded, outputDir / ("plane_" + std::to_string(p)), quality);
            } else {
                fs::remove(outputDir / ("plane_" + std::to_string(p) + ".dzi"));
                fs::remove_all(outputDir / ("plane_" + std::to_string(p) + "_files"));
                writeImageChecked(planePath, encoded, jpegParams);
            }
        }
        cleanupStalePlaneFiles(outputDir, 1 + coefficientJpegs);
        writeInfoJson(outputDir, expectedSize, lights, scales, biases, opt.rtiColorMode, quality);
        finishPackage();
        return;
    }

    std::vector<cv::Mat> coeffImages(static_cast<size_t>(basisCount));
    for (cv::Mat& plane : coeffImages) {
        plane = cv::Mat(expectedSize, CV_32FC3, cv::Scalar(0, 0, 0));
    }

    reportProgress(progress, "RTI: fitting PTM coefficient planes...");
    for (int y = 0; y < expectedSize.height; ++y) {
        for (int x = 0; x < expectedSize.width; ++x) {
            std::array<std::array<double, 6>, 3> channelCoeffs{};
            for (int c = 0; c < 3; ++c) {
                channelCoeffs[c] = solvePixelCoefficients(images, bases, normal, basisCount, x, y, c);
            }
            for (int p = 0; p < basisCount; ++p) {
                cv::Vec3f& dst = coeffImages[p].ptr<cv::Vec3f>(y)[x];
                dst[0] = static_cast<float>(channelCoeffs[2][p]);
                dst[1] = static_cast<float>(channelCoeffs[1][p]);
                dst[2] = static_cast<float>(channelCoeffs[0][p]);
            }
        }
    }

    if (opt.rtiLayoutMode == RtiLayoutMode::WebRtiViewer) {
        reportProgress(progress, "RTI: writing webRTIViewer RGB component tiles...");
        std::vector<double> webScales(6, 1.0);
        std::vector<double> webBiases(6, 0.0);
        std::vector<cv::Mat> webLayers;
        webLayers.reserve(6);
        for (int webCoeff = 0; webCoeff < 6; ++webCoeff) {
            const int internalCoeff = webRtiInternalCoeffIndex(webCoeff);
            if (internalCoeff >= basisCount) {
                webLayers.push_back(cv::Mat(expectedSize, CV_8UC3, cv::Scalar(0, 0, 0)));
                continue;
            }
            double scale = 1.0;
            double bias = 0.0;
            webLayers.push_back(encodeWebRtiRgbCoefficient(coeffImages[static_cast<size_t>(internalCoeff)], scale, bias));
            webScales[static_cast<size_t>(webCoeff)] = scale;
            webBiases[static_cast<size_t>(webCoeff)] = bias;
        }
        writeWebRtiViewerPackage(
            outputDir,
            expectedSize,
            opt.rtiColorMode,
            webLayers,
            webScales,
            webBiases,
            progress);
        finishPackage();
        return;
    }

    std::vector<double> scales(static_cast<size_t>(basisCount * 3), 1.0);
    std::vector<double> biases(static_cast<size_t>(basisCount * 3), 0.0);
    reportProgress(progress, opt.rtiLayoutMode == RtiLayoutMode::DeepZoom
        ? "RTI: writing DeepZoom coefficient pyramids..."
        : "RTI: writing coefficient images...");
    for (int p = 0; p < basisCount; ++p) {
        cv::Mat encoded = encodeCoefficientJpeg(coeffImages[p], p, scales, biases);
        const fs::path planePath = outputDir / ("plane_" + std::to_string(p) + ".jpg");
        if (opt.rtiLayoutMode == RtiLayoutMode::DeepZoom) {
            fs::remove(planePath);
            writeDeepZoomPyramid(encoded, outputDir / ("plane_" + std::to_string(p)), quality);
        } else {
            fs::remove(outputDir / ("plane_" + std::to_string(p) + ".dzi"));
            fs::remove_all(outputDir / ("plane_" + std::to_string(p) + "_files"));
            writeImageChecked(planePath, encoded, jpegParams);
        }
    }
    cleanupStalePlaneFiles(outputDir, basisCount);

    writeInfoJson(outputDir, expectedSize, lights, scales, biases, opt.rtiColorMode, quality);
    finishPackage();
}
