#include "calibration.hpp"

#include "checked_io.hpp"
#include "input_response.hpp"
#include "photometric.hpp"
#include "radiometry.hpp"

#include <nlohmann/json.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;
using Json = nlohmann::json;

namespace {

constexpr double kMaximumCalibrationRmsPixels = 0.75;
constexpr double kMaximumCalibrationResidualPixels = 2.5;

[[noreturn]] void die(const std::string& message) {
    throw std::runtime_error(message);
}

double median(std::vector<double> values) {
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    const size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(middle), values.end());
    double value = values[middle];
    if (values.size() % 2 == 0) {
        const auto lower = std::max_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(middle));
        value = 0.5 * (value + *lower);
    }
    return value;
}

std::array<double, 10> terms(double x, double y, int width, int height) {
    const double scale = static_cast<double>(std::max(width, height));
    const double u = (x - 0.5 * static_cast<double>(width - 1)) / scale;
    const double v = (y - 0.5 * static_cast<double>(height - 1)) / scale;
    return {1.0, u, v, u * u, u * v, v * v,
        u * u * u, u * u * v, u * v * v, v * v * v};
}

double evaluate(const std::array<double, 10>& coefficients, double x, double y, int width, int height) {
    const auto basis = terms(x, y, width, height);
    return std::inner_product(coefficients.begin(), coefficients.end(), basis.begin(), 0.0);
}

std::array<double, 10> fitPolynomial(
    const std::vector<cv::Point2f>& ideal,
    const std::vector<cv::Point2f>& observed,
    int width,
    int height,
    bool xCoordinate) {
    cv::Mat a(static_cast<int>(ideal.size()), 10, CV_64F);
    cv::Mat b(static_cast<int>(ideal.size()), 1, CV_64F);
    for (size_t i = 0; i < ideal.size(); ++i) {
        const auto basis = terms(ideal[i].x, ideal[i].y, width, height);
        for (int j = 0; j < 10; ++j) a.at<double>(static_cast<int>(i), j) = basis[static_cast<size_t>(j)];
        b.at<double>(static_cast<int>(i)) = xCoordinate ? observed[i].x : observed[i].y;
    }
    cv::Mat coefficients;
    if (!cv::solve(a, b, coefficients, cv::DECOMP_SVD) || coefficients.rows != 10) {
        die("Could not fit the microscope working-plane correction.");
    }
    std::array<double, 10> result{};
    for (int i = 0; i < 10; ++i) result[static_cast<size_t>(i)] = coefficients.at<double>(i);
    return result;
}

cv::Mat calibrationComposite(const Options& opt, std::vector<cv::Mat>& linearImages) {
    linearImages.clear();
    linearImages.reserve(opt.imagePaths.size());
    cv::Mat composite;
    cv::Size expected;
    for (size_t i = 0; i < opt.imagePaths.size(); ++i) {
        cv::Mat raw = cv::imread(opt.imagePaths[i], cv::IMREAD_UNCHANGED);
        if (raw.empty()) die("Could not read calibration image: " + opt.imagePaths[i]);
        cv::Mat linear = convertToLinearLuminance(raw, inputResponseForImage(opt, i).srgb);
        if (expected.empty()) expected = linear.size();
        else if (linear.size() != expected) die("Microscope calibration images must have identical dimensions.");
        linearImages.push_back(linear);

        double low = 0.0;
        double high = 0.0;
        cv::minMaxLoc(linear, &low, &high);
        if (!(high > low)) continue;
        cv::Mat normalized;
        linear.convertTo(normalized, CV_32F, 1.0 / (high - low), -low / (high - low));
        if (composite.empty()) composite = cv::Mat::zeros(linear.size(), CV_32F);
        composite += normalized;
    }
    if (composite.empty()) die("Calibration stack contains no usable contrast.");
    composite /= static_cast<float>(opt.imagePaths.size());
    cv::Mat display;
    composite.convertTo(display, CV_8U, 255.0);
    return display;
}

std::vector<cv::Point2f> detectCorners(const cv::Mat& composite, int innerColumns, int innerRows) {
    const int maximumSide = std::max(composite.cols, composite.rows);
    const double scale = maximumSide > 2500 ? 2500.0 / maximumSide : 1.0;
    cv::Mat detection = composite;
    if (scale < 1.0) cv::resize(composite, detection, cv::Size(), scale, scale, cv::INTER_AREA);
    std::vector<cv::Point2f> corners;
    const bool found = cv::findChessboardCornersSB(
        detection,
        cv::Size(innerColumns, innerRows),
        corners,
        cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_EXHAUSTIVE | cv::CALIB_CB_ACCURACY);
    if (!found || corners.size() != static_cast<size_t>(innerColumns * innerRows)) {
        die("Could not find the complete checkerboard. Use the generated target, keep it flat and in focus, and fill most of the microscope field without clipping its outer border.");
    }
    if (scale < 1.0) {
        for (cv::Point2f& point : corners) point *= static_cast<float>(1.0 / scale);
    }
    return corners;
}

std::vector<double> cellLevels(
    const cv::Mat& image,
    const std::vector<cv::Point2f>& corners,
    int innerColumns,
    int innerRows,
    int parity) {
    std::vector<double> levels;
    const std::array<cv::Point2f, 4> destination{
        cv::Point2f(0, 0), cv::Point2f(15, 0), cv::Point2f(15, 15), cv::Point2f(0, 15)};
    for (int row = 0; row + 1 < innerRows; ++row) {
        for (int column = 0; column + 1 < innerColumns; ++column) {
            if ((row + column) % 2 != parity) continue;
            const auto at = [&](int y, int x) -> const cv::Point2f& {
                return corners[static_cast<size_t>(y * innerColumns + x)];
            };
            const std::array<cv::Point2f, 4> source{
                at(row, column), at(row, column + 1), at(row + 1, column + 1), at(row + 1, column)};
            const cv::Mat transform = cv::getPerspectiveTransform(source.data(), destination.data());
            cv::Mat patch;
            cv::warpPerspective(image, patch, transform, cv::Size(16, 16), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
            std::vector<double> samples;
            samples.reserve(64);
            for (int y = 4; y < 12; ++y) {
                const float* values = patch.ptr<float>(y);
                for (int x = 4; x < 12; ++x) {
                    const double value = values[x];
                    if (std::isfinite(value) && value > 0.002 && value < 0.98) samples.push_back(value);
                }
            }
            const double level = median(std::move(samples));
            if (std::isfinite(level)) levels.push_back(level);
        }
    }
    return levels;
}

void normalizeGains(std::vector<double>& gains) {
    double logSum = 0.0;
    for (const double gain : gains) {
        if (!std::isfinite(gain) || gain <= 0.0) die("Light gains must be finite and positive.");
        logSum += std::log(gain);
    }
    const double scale = std::exp(logSum / static_cast<double>(gains.size()));
    for (double& gain : gains) gain /= scale;
}

bool solveScaledNormal(
    const std::vector<cv::Mat>& images,
    const std::vector<cv::Mat>& saturationMasks,
    const std::vector<cv::Vec3d>& pixelLights,
    const std::vector<double>& gains,
    int x,
    int y,
    float low,
    float high,
    int excludedLight,
    cv::Vec3d& solution) {
    cv::Matx33d a = cv::Matx33d::zeros();
    cv::Vec3d b(0, 0, 0);
    int observations = 0;
    for (size_t i = 0; i < images.size(); ++i) {
        if (static_cast<int>(i) == excludedLight) continue;
        if (!saturationMasks.empty() && saturationMasks[i].ptr<uchar>(y)[x] != 0) continue;
        const double intensity = images[i].ptr<float>(y)[x];
        if (!std::isfinite(intensity) || intensity <= low || intensity >= high) continue;
        const cv::Vec3d l = gains[i] * pixelLights[i];
        for (int r = 0; r < 3; ++r) {
            b[r] += l[r] * intensity;
            for (int c = 0; c < 3; ++c) a(r, c) += l[r] * l[c];
        }
        ++observations;
    }
    if (observations < 3) return false;
    // A is only 3x3. LU avoids the much heavier per-pixel SVD; singular lighting
    // subsets produce a zero/invalid q and are discarded by the checks below.
    const cv::Matx33d inverse = a.inv(cv::DECOMP_LU);
    solution = inverse * b;
    return std::isfinite(solution[0]) && std::isfinite(solution[1]) && std::isfinite(solution[2]) &&
        solution[2] > 0.0 && cv::norm(solution) > 1.0e-8;
}

struct PartitionFit {
    std::vector<double> gains;
    int pixels = 0;
    double normalDiversity = 0.0;
};

PartitionFit fitGainPartition(
    const std::vector<cv::Mat>& images,
    const std::vector<cv::Mat>& saturationMasks,
    const std::vector<cv::Vec3f>& lights,
    const cv::Mat& mask,
    float low,
    float high,
    int partition,
    const std::vector<double>& priorGains,
    const LightGainGeometry& geometry) {
    const double requested = partition < 0 ? 120000.0 : 70000.0;
    const int step = std::max(1, static_cast<int>(std::ceil(std::sqrt(
        static_cast<double>(images[0].total()) / requested))));
    std::vector<double> gains = priorGains.empty()
        ? std::vector<double>(images.size(), 1.0)
        : priorGains;
    if (gains.size() != images.size()) die("Light-gain prior count must match image count.");
    normalizeGains(gains);
    const std::vector<double> prior = gains;
    int usedPixels = 0;
    cv::Vec3d normalSum(0, 0, 0);
    std::vector<cv::Vec3d> pixelLights(lights.size());
    for (int iteration = 0; iteration < 6; ++iteration) {
        std::vector<std::vector<double>> ratios(images.size());
        usedPixels = 0;
        normalSum = cv::Vec3d(0, 0, 0);
        for (int y = step / 2; y < images[0].rows; y += step) {
            const uchar* maskRow = mask.ptr<uchar>(y);
            for (int x = step / 2; x < images[0].cols; x += step) {
                if (maskRow[x] == 0) continue;
                if (partition >= 0 && ((x / (4 * step) + y / (4 * step)) & 1) != partition) continue;
                for (size_t i = 0; i < lights.size(); ++i) {
                    pixelLights[i] = cv::Vec3d(photometricLightVectorAtPixel(
                        lights[i], static_cast<int>(i), static_cast<int>(lights.size()), x, y,
                        geometry.model, geometry.ringLightRadiusMm, geometry.ringLightHeightMm,
                        geometry.pixelScaleMm, geometry.lightingCenter));
                }
                cv::Vec3d q;
                if (!solveScaledNormal(
                    images, saturationMasks, pixelLights, gains, x, y, low, high, -1, q)) continue;
                ++usedPixels;
                normalSum += q * (1.0 / cv::norm(q));
                for (size_t i = 0; i < images.size(); ++i) {
                    const double intensity = images[i].ptr<float>(y)[x];
                    if (!saturationMasks.empty() && saturationMasks[i].ptr<uchar>(y)[x] != 0) continue;
                    const double predictedWithoutGain = pixelLights[i].dot(q);
                    if (intensity <= low || intensity >= high || predictedWithoutGain <= std::max(0.03f, 1.5f * low)) continue;
                    const double ratio = intensity / predictedWithoutGain;
                    if (std::isfinite(ratio) && ratio > 0.2 && ratio < 5.0) ratios[i].push_back(ratio);
                }
            }
        }
        std::vector<double> next = gains;
        for (size_t i = 0; i < gains.size(); ++i) {
            if (ratios[i].size() < 64) continue;
            const double estimate = median(std::move(ratios[i]));
            // A modest log-domain prior prevents the specimen from manufacturing extreme gains.
            next[i] = std::exp(0.85 * std::log(estimate) + 0.15 * std::log(prior[i]));
        }
        normalizeGains(next);
        double change = 0.0;
        for (size_t i = 0; i < gains.size(); ++i) change = std::max(change, std::abs(std::log(next[i] / gains[i])));
        gains = std::move(next);
        if (change < 0.001) break;
    }
    double normalDiversity = 0.0;
    if (usedPixels > 0) {
        const cv::Vec3d meanNormal = normalSum * (1.0 / static_cast<double>(usedPixels));
        normalDiversity = std::sqrt(std::max(0.0, 1.0 - meanNormal.dot(meanNormal)));
    }
    return {std::move(gains), usedPixels, normalDiversity};
}

double leaveOneLightOutError(
    const std::vector<cv::Mat>& images,
    const std::vector<cv::Mat>& saturationMasks,
    const std::vector<cv::Vec3f>& lights,
    const cv::Mat& mask,
    const std::vector<double>& gains,
    float low,
    float high,
    const LightGainGeometry& geometry) {
    const int step = std::max(1, static_cast<int>(std::ceil(std::sqrt(
        static_cast<double>(images[0].total()) / 16000.0))));
    double absoluteError = 0.0;
    double reference = 0.0;
    size_t count = 0;
    std::vector<cv::Vec3d> pixelLights(lights.size());
    for (int y = step / 2; y < images[0].rows; y += step) {
        const uchar* maskRow = mask.ptr<uchar>(y);
        for (int x = step / 2; x < images[0].cols; x += step) {
            if (maskRow[x] == 0 || ((x / (4 * step) + y / (4 * step)) & 1) == 0) continue;
            for (size_t i = 0; i < lights.size(); ++i) {
                pixelLights[i] = cv::Vec3d(photometricLightVectorAtPixel(
                    lights[i], static_cast<int>(i), static_cast<int>(lights.size()), x, y,
                    geometry.model, geometry.ringLightRadiusMm, geometry.ringLightHeightMm,
                    geometry.pixelScaleMm, geometry.lightingCenter));
            }
            for (size_t heldOut = 0; heldOut < images.size(); ++heldOut) {
                if (!saturationMasks.empty() && saturationMasks[heldOut].ptr<uchar>(y)[x] != 0) continue;
                const double observed = images[heldOut].ptr<float>(y)[x];
                if (observed <= low || observed >= high) continue;
                cv::Vec3d q;
                if (!solveScaledNormal(images, saturationMasks, pixelLights, gains, x, y, low, high,
                    static_cast<int>(heldOut), q)) continue;
                const double predicted = gains[heldOut] * std::max(0.0, pixelLights[heldOut].dot(q));
                if (predicted <= low) continue;
                absoluteError += std::min(std::abs(predicted - observed), 0.5 * observed + 0.05);
                reference += observed;
                ++count;
            }
        }
    }
    if (count < 256 || reference <= 0.0) return std::numeric_limits<double>::infinity();
    return absoluteError / reference;
}

} // namespace

void writePrintableCheckerboardSvg(
    const std::string& path,
    int gridColumns,
    int gridRows,
    double squareSizeMm,
    double pageWidthMm,
    double pageHeightMm) {
    if (gridColumns < 5 || gridRows < 5 || gridColumns > 80 || gridRows > 80) {
        die("Checkerboard must contain 5 to 80 squares in each direction.");
    }
    if (!std::isfinite(squareSizeMm) || squareSizeMm <= 0.0 ||
        !std::isfinite(pageWidthMm) || !std::isfinite(pageHeightMm) ||
        pageWidthMm <= 0.0 || pageHeightMm <= 0.0) {
        die("Checkerboard and page dimensions must be positive finite millimeter values.");
    }
    const double boardWidth = gridColumns * squareSizeMm;
    const double boardHeight = gridRows * squareSizeMm;
    if (boardWidth + 10.0 > pageWidthMm || boardHeight + 16.0 > pageHeightMm) {
        die("The checkerboard does not fit on the requested page with measurement margins.");
    }
    const double left = 0.5 * (pageWidthMm - boardWidth);
    const double top = 0.5 * (pageHeightMm - boardHeight);
    CheckedOutputFile file(path);
    std::ostream& out = file.stream();
    out << std::fixed << std::setprecision(4)
        << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << pageWidthMm
        << "mm\" height=\"" << pageHeightMm << "mm\" viewBox=\"0 0 "
        << pageWidthMm << ' ' << pageHeightMm << "\">\n"
        << "  <rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
    for (int row = 0; row < gridRows; ++row) {
        for (int column = 0; column < gridColumns; ++column) {
            if ((row + column) % 2 != 0) continue;
            out << "  <rect x=\"" << left + column * squareSizeMm << "\" y=\""
                << top + row * squareSizeMm << "\" width=\"" << squareSizeMm
                << "\" height=\"" << squareSizeMm << "\" fill=\"black\"/>\n";
        }
    }
    out << "  <rect x=\"" << left << "\" y=\"" << top << "\" width=\"" << boardWidth
        << "\" height=\"" << boardHeight << "\" fill=\"none\" stroke=\"black\" stroke-width=\"0.15\"/>\n"
        << "  <text x=\"" << pageWidthMm / 2.0 << "\" y=\"" << top + boardHeight + 5.0
        << "\" font-family=\"sans-serif\" font-size=\"3\" text-anchor=\"middle\">"
        << gridColumns << " x " << gridRows << " squares; " << squareSizeMm
        << " mm each. Print at 100% / Actual size.</text>\n"
        << "  <line x1=\"" << left << "\" y1=\"" << top - 3.0 << "\" x2=\""
        << left + 10.0 << "\" y2=\"" << top - 3.0 << "\" stroke=\"black\" stroke-width=\"0.3\"/>\n"
        << "  <text x=\"" << left + 5.0 << "\" y=\"" << top - 4.0
        << "\" font-family=\"sans-serif\" font-size=\"2.5\" text-anchor=\"middle\">10 mm check</text>\n"
        << "</svg>\n";
    file.commit();
}

MicroscopeCalibration createMicroscopeCalibration(
    const Options& calibrationImages,
    int gridColumns,
    int gridRows,
    double squareSizeMm) {
    if (calibrationImages.imagePaths.empty()) die("Select at least one checkerboard calibration image.");
    if (gridColumns < 5 || gridRows < 5 || !std::isfinite(squareSizeMm) || squareSizeMm <= 0.0) {
        die("Calibration grid dimensions must match a valid generated checkerboard.");
    }
    std::vector<cv::Mat> images;
    const cv::Mat composite = calibrationComposite(calibrationImages, images);
    const int innerColumns = gridColumns - 1;
    const int innerRows = gridRows - 1;
    const std::vector<cv::Point2f> corners = detectCorners(composite, innerColumns, innerRows);

    std::vector<double> spacings;
    for (int row = 0; row < innerRows; ++row) {
        for (int column = 0; column < innerColumns; ++column) {
            const size_t index = static_cast<size_t>(row * innerColumns + column);
            if (column + 1 < innerColumns) spacings.push_back(cv::norm(corners[index + 1] - corners[index]));
            if (row + 1 < innerRows) spacings.push_back(cv::norm(corners[index + innerColumns] - corners[index]));
        }
    }
    const double spacing = median(std::move(spacings));
    if (!std::isfinite(spacing) || spacing < 3.0) die("Detected checkerboard cells are too small for reliable calibration.");
    cv::Point2d center(0, 0);
    for (const cv::Point2f& point : corners) center += cv::Point2d(point);
    center *= 1.0 / static_cast<double>(corners.size());
    std::vector<cv::Point2f> ideal;
    ideal.reserve(corners.size());
    for (int row = 0; row < innerRows; ++row) {
        for (int column = 0; column < innerColumns; ++column) {
            ideal.emplace_back(
                static_cast<float>(center.x + (column - 0.5 * (innerColumns - 1)) * spacing),
                static_cast<float>(center.y + (row - 0.5 * (innerRows - 1)) * spacing));
        }
    }

    MicroscopeCalibration result;
    result.imageWidth = composite.cols;
    result.imageHeight = composite.rows;
    result.gridColumns = gridColumns;
    result.gridRows = gridRows;
    result.squareSizeMm = squareSizeMm;
    result.pixelScaleMm = squareSizeMm / spacing;
    result.sourceX = fitPolynomial(ideal, corners, composite.cols, composite.rows, true);
    result.sourceY = fitPolynomial(ideal, corners, composite.cols, composite.rows, false);
    double squaredError = 0.0;
    double maximumError = 0.0;
    for (size_t i = 0; i < ideal.size(); ++i) {
        const double dx = evaluate(result.sourceX, ideal[i].x, ideal[i].y, composite.cols, composite.rows) - corners[i].x;
        const double dy = evaluate(result.sourceY, ideal[i].x, ideal[i].y, composite.cols, composite.rows) - corners[i].y;
        const double error = std::sqrt(dx * dx + dy * dy);
        squaredError += error * error;
        maximumError = std::max(maximumError, error);
    }
    result.reprojectionRmsPixels = std::sqrt(squaredError / static_cast<double>(ideal.size()));
    result.reprojectionMaxPixels = maximumError;
    const cv::Rect boardBounds = cv::boundingRect(corners);
    result.boardCoverageFraction =
        (static_cast<double>(boardBounds.width) * static_cast<double>(boardBounds.height)) /
        static_cast<double>(composite.total());
    if (result.boardCoverageFraction < 0.20) {
        die("The checkerboard covers less than 20% of the image. Move closer or use a larger grid so the calibration constrains the working field.");
    }
    if (result.reprojectionRmsPixels > kMaximumCalibrationRmsPixels ||
        result.reprojectionMaxPixels > kMaximumCalibrationResidualPixels) {
        die("The checkerboard working-plane fit is not accurate enough (RMS " +
            std::to_string(result.reprojectionRmsPixels) + " px, maximum " +
            std::to_string(result.reprojectionMaxPixels) +
            " px). Keep the target flat and fronto-parallel, improve focus, or use a denser/larger target.");
    }

    const std::vector<double> firstParityZero = cellLevels(images.front(), corners, innerColumns, innerRows, 0);
    const std::vector<double> firstParityOne = cellLevels(images.front(), corners, innerColumns, innerRows, 1);
    const double parityZero = median(firstParityZero);
    const double parityOne = median(firstParityOne);
    if (!std::isfinite(parityZero) || !std::isfinite(parityOne)) {
        die("The checkerboard does not contain usable unclipped cell interiors in the first calibration image.");
    }
    const int whiteParity = parityOne > parityZero ? 1 : 0;
    const size_t minimumCells = std::max<size_t>(3,
        static_cast<size_t>((innerColumns - 1) * (innerRows - 1)) / 8);
    std::vector<double> levels;
    for (size_t i = 0; i < images.size(); ++i) {
        const std::vector<double> whiteCells = cellLevels(
            images[i], corners, innerColumns, innerRows, whiteParity);
        const std::vector<double> blackCells = cellLevels(
            images[i], corners, innerColumns, innerRows, 1 - whiteParity);
        if (whiteCells.size() < minimumCells || blackCells.size() < minimumCells) {
            die("Too few unclipped checker-cell samples in calibration image: " + calibrationImages.imagePaths[i]);
        }
        const double whiteLevel = median(whiteCells);
        const double blackLevel = median(blackCells);
        const double level = whiteLevel - blackLevel;
        if (!std::isfinite(level) || level <= 0.02 || level < 0.10 * whiteLevel) {
            die("Checkerboard contrast is too low in calibration image: " + calibrationImages.imagePaths[i]);
        }
        levels.push_back(level);
        result.calibrationImageNames.push_back(fs::path(calibrationImages.imagePaths[i]).filename().string());
    }
    result.lightGains = levels;
    normalizeGains(result.lightGains);
    for (const double gain : result.lightGains) {
        if (gain < 0.25 || gain > 4.0) die("Calibration light balance exceeds the supported 4:1 range; check exposure and target clipping.");
    }
    return result;
}

void saveMicroscopeCalibration(const std::string& path, const MicroscopeCalibration& calibration) {
    Json document{
        {"schema", "what-a-relief-microscope-calibration"},
        {"schema_version", 1},
        {"model", "working_plane_cubic_warp"},
        {"image_size", {calibration.imageWidth, calibration.imageHeight}},
        {"checkerboard", {{"squares_x", calibration.gridColumns}, {"squares_y", calibration.gridRows},
            {"square_size_mm", calibration.squareSizeMm}, {"coverage_fraction", calibration.boardCoverageFraction}}},
        {"pixel_scale_mm_per_pixel", calibration.pixelScaleMm},
        {"fit", {{"rms_pixels", calibration.reprojectionRmsPixels}, {"maximum_pixels", calibration.reprojectionMaxPixels}}},
        {"output_to_source_polynomial", {{"terms", {"1", "u", "v", "u2", "uv", "v2", "u3", "u2v", "uv2", "v3"}},
            {"x", calibration.sourceX}, {"y", calibration.sourceY}}},
        {"light_gains", calibration.lightGains},
        {"calibration_image_names", calibration.calibrationImageNames},
        {"gain_convention", "observed relative response; application divides each image by its gain; geometric mean is 1"},
        {"scope", "fixed microscope working plane; not a general pinhole intrinsic calibration"}
    };
    CheckedOutputFile file(path);
    file.stream() << std::setw(2) << document << '\n';
    file.commit();
}

MicroscopeCalibration loadMicroscopeCalibration(const std::string& path) {
    std::ifstream input(path);
    if (!input) die("Could not open microscope calibration: " + path);
    Json document;
    try { input >> document; }
    catch (const std::exception&) { die("Microscope calibration is not valid JSON: " + path); }
    if (document.value("schema", "") != "what-a-relief-microscope-calibration" || document.value("schema_version", 0) != 1 ||
        document.value("model", "") != "working_plane_cubic_warp") {
        die("Unsupported microscope calibration schema or model: " + path);
    }
    MicroscopeCalibration result;
    try {
        result.imageWidth = document.at("image_size").at(0).get<int>();
        result.imageHeight = document.at("image_size").at(1).get<int>();
        result.gridColumns = document.at("checkerboard").at("squares_x").get<int>();
        result.gridRows = document.at("checkerboard").at("squares_y").get<int>();
        result.squareSizeMm = document.at("checkerboard").at("square_size_mm").get<double>();
        result.boardCoverageFraction = document.at("checkerboard").at("coverage_fraction").get<double>();
        result.pixelScaleMm = document.at("pixel_scale_mm_per_pixel").get<double>();
        result.reprojectionRmsPixels = document.at("fit").at("rms_pixels").get<double>();
        result.reprojectionMaxPixels = document.at("fit").at("maximum_pixels").get<double>();
        result.sourceX = document.at("output_to_source_polynomial").at("x").get<std::array<double, 10>>();
        result.sourceY = document.at("output_to_source_polynomial").at("y").get<std::array<double, 10>>();
        result.lightGains = document.at("light_gains").get<std::vector<double>>();
        result.calibrationImageNames = document.value("calibration_image_names", std::vector<std::string>{});
    } catch (const std::exception&) {
        die("Microscope calibration is incomplete or contains invalid value types: " + path);
    }
    if (result.imageWidth <= 0 || result.imageHeight <= 0 ||
        result.gridColumns < 5 || result.gridColumns > 80 ||
        result.gridRows < 5 || result.gridRows > 80 ||
        !std::isfinite(result.squareSizeMm) || result.squareSizeMm <= 0.0 ||
        !std::isfinite(result.pixelScaleMm) || result.pixelScaleMm <= 0.0 ||
        !std::isfinite(result.boardCoverageFraction) || result.boardCoverageFraction < 0.20 ||
        result.boardCoverageFraction > 1.0 ||
        !std::isfinite(result.reprojectionRmsPixels) || result.reprojectionRmsPixels < 0.0 ||
        result.reprojectionRmsPixels > kMaximumCalibrationRmsPixels ||
        !std::isfinite(result.reprojectionMaxPixels) || result.reprojectionMaxPixels < 0.0 ||
        result.reprojectionMaxPixels > kMaximumCalibrationResidualPixels ||
        result.lightGains.empty()) {
        die("Microscope calibration contains invalid dimensions, scale, or gains: " + path);
    }
    for (const double coefficient : result.sourceX) {
        if (!std::isfinite(coefficient)) die("Microscope calibration contains a non-finite X warp coefficient: " + path);
    }
    for (const double coefficient : result.sourceY) {
        if (!std::isfinite(coefficient)) die("Microscope calibration contains a non-finite Y warp coefficient: " + path);
    }
    normalizeGains(result.lightGains);
    for (const double gain : result.lightGains) {
        if (gain < 0.25 || gain > 4.0) die("Microscope calibration reference gains exceed the supported range: " + path);
    }
    return result;
}

void validateMicroscopeCalibration(
    const MicroscopeCalibration& calibration,
    const cv::Size& imageSize,
    size_t lightCount) {
    if (imageSize.width != calibration.imageWidth || imageSize.height != calibration.imageHeight) {
        die("Microscope calibration image size is " + std::to_string(calibration.imageWidth) + "x" +
            std::to_string(calibration.imageHeight) + ", but the specimen images are " +
            std::to_string(imageSize.width) + "x" + std::to_string(imageSize.height) + ". Recalibrate at the same camera resolution.");
    }
    if (calibration.lightGains.size() != lightCount) {
        die("Microscope calibration contains " + std::to_string(calibration.lightGains.size()) +
            " lighting positions, but the specimen stack contains " + std::to_string(lightCount) +
            ". Select the same positions in the same order.");
    }
}

MicroscopeRectificationMaps buildMicroscopeRectificationMaps(
    const MicroscopeCalibration& calibration) {
    MicroscopeRectificationMaps maps;
    maps.sourceCoordinates.create(calibration.imageHeight, calibration.imageWidth, CV_32FC2);
    cv::parallel_for_(cv::Range(0, calibration.imageHeight), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            cv::Vec2f* row = maps.sourceCoordinates.ptr<cv::Vec2f>(y);
            for (int x = 0; x < calibration.imageWidth; ++x) {
                row[x][0] = static_cast<float>(evaluate(
                    calibration.sourceX, x, y, calibration.imageWidth, calibration.imageHeight));
                row[x][1] = static_cast<float>(evaluate(
                    calibration.sourceY, x, y, calibration.imageWidth, calibration.imageHeight));
            }
        }
    });
    return maps;
}

cv::Mat rectifyMicroscopeImage(
    const cv::Mat& image,
    const MicroscopeRectificationMaps& maps,
    int interpolation) {
    if (maps.sourceCoordinates.empty() || maps.sourceCoordinates.type() != CV_32FC2 ||
        maps.sourceCoordinates.size() != image.size()) {
        die("Microscope rectification map does not match the image dimensions.");
    }
    cv::Mat result;
    cv::remap(image, result, maps.sourceCoordinates, cv::Mat(), interpolation, cv::BORDER_CONSTANT, cv::Scalar(0));
    return result;
}

cv::Mat rectifyMicroscopeImage(
    const cv::Mat& image,
    const MicroscopeCalibration& calibration,
    int interpolation) {
    validateMicroscopeCalibration(calibration, image.size(), calibration.lightGains.size());
    return rectifyMicroscopeImage(image, buildMicroscopeRectificationMaps(calibration), interpolation);
}

void applyLightGainCorrection(std::vector<cv::Mat>& images, const std::vector<double>& gains) {
    if (images.size() != gains.size()) die("Light-gain count must match image count.");
    for (size_t i = 0; i < images.size(); ++i) {
        if (!std::isfinite(gains[i]) || gains[i] <= 0.0) die("Light gains must be finite and positive.");
        images[i] *= static_cast<float>(1.0 / gains[i]);
    }
}

LightGainEstimate estimateRelativeLightGains(
    const std::vector<cv::Mat>& images,
    const std::vector<cv::Vec3f>& lights,
    const cv::Mat& mask,
    float shadowThreshold,
    float highOutlierThreshold,
    const std::vector<double>& priorGains,
    LightGainGeometry geometry,
    const std::vector<cv::Mat>& saturationMasks) {
    LightGainEstimate result;
    result.attempted = true;
    if (images.size() < 5 || images.size() != lights.size()) {
        result.decision = "rejected_requires_five_calibrated_lights";
        return result;
    }
    if (images.empty() || mask.size() != images.front().size() || mask.type() != CV_8U) {
        die("Relative light-gain estimation requires matching images, lights, and mask.");
    }
    if (!saturationMasks.empty()) {
        if (saturationMasks.size() != images.size()) {
            die("Light-gain saturation-mask count must match image count.");
        }
        for (const cv::Mat& saturation : saturationMasks) {
            if (saturation.size() != images.front().size() || saturation.type() != CV_8U) {
                die("Light-gain saturation masks must be matching 8-bit images.");
            }
        }
    }
    if (geometry.model == LightingModel::NearFieldRing &&
        (!std::isfinite(geometry.ringLightRadiusMm) || geometry.ringLightRadiusMm <= 0.0 ||
         !std::isfinite(geometry.ringLightHeightMm) || geometry.ringLightHeightMm <= 0.0 ||
         !std::isfinite(geometry.pixelScaleMm) || geometry.pixelScaleMm <= 0.0 ||
         !std::isfinite(geometry.lightingCenter.x) || !std::isfinite(geometry.lightingCenter.y))) {
        die("Near-field light-gain estimation requires finite positive ring geometry and pixel scale.");
    }
    const PartitionFit first = fitGainPartition(
        images, saturationMasks, lights, mask, shadowThreshold, highOutlierThreshold, 0, priorGains, geometry);
    const PartitionFit second = fitGainPartition(
        images, saturationMasks, lights, mask, shadowThreshold, highOutlierThreshold, 1, priorGains, geometry);
    result.sampledPixels = first.pixels + second.pixels;
    if (first.pixels < 128 || second.pixels < 128) {
        result.decision = "rejected_insufficient_diffuse_samples";
        return result;
    }
    result.normalDiversity = std::min(first.normalDiversity, second.normalDiversity);
    if (result.normalDiversity < 0.06) {
        result.decision = "rejected_insufficient_normal_diversity";
        return result;
    }
    double squaredStability = 0.0;
    double maximumMagnitude = 0.0;
    for (size_t i = 0; i < first.gains.size(); ++i) {
        const double difference = std::log(first.gains[i] / second.gains[i]);
        squaredStability += difference * difference;
        maximumMagnitude = std::max(maximumMagnitude, std::abs(std::log(first.gains[i])));
    }
    result.stability = std::sqrt(squaredStability / static_cast<double>(first.gains.size()));
    result.validationErrorBefore = leaveOneLightOutError(
        images, saturationMasks, lights, mask, std::vector<double>(lights.size(), 1.0),
        shadowThreshold, highOutlierThreshold, geometry);
    result.validationErrorAfter = leaveOneLightOutError(
        images, saturationMasks, lights, mask, first.gains, shadowThreshold, highOutlierThreshold, geometry);
    if (!std::isfinite(result.validationErrorBefore) || !std::isfinite(result.validationErrorAfter)) {
        result.decision = "rejected_insufficient_validation_samples";
        return result;
    }
    if (maximumMagnitude > std::log(2.0)) {
        result.decision = "rejected_gain_range_exceeded";
        return result;
    }
    if (result.stability > 0.10) {
        result.decision = "rejected_spatially_unstable";
        return result;
    }
    if (result.validationErrorAfter >= 0.98 * result.validationErrorBefore) {
        result.decision = "rejected_withheld_prediction_did_not_improve";
        return result;
    }
    const PartitionFit full = fitGainPartition(
        images, saturationMasks, lights, mask, shadowThreshold, highOutlierThreshold, -1, priorGains, geometry);
    result.gains = full.gains;
    result.accepted = true;
    result.decision = "accepted_withheld_prediction_improved";
    return result;
}
