#include "photometric.hpp"
#include "radiometry.hpp"
#include "shadow_refinement.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

constexpr double kPi = 3.14159265358979323846;
constexpr double kRadiansToDegrees = 180.0 / kPi;

struct TestContext {
    int failures = 0;

    void check(bool condition, const std::string& message) {
        if (!condition) {
            ++failures;
            std::cerr << "FAIL: " << message << '\n';
        }
    }
};

void expectThrows(
    TestContext& context,
    const std::function<void()>& action,
    const std::string& message) {
    try {
        action();
        context.check(false, message);
    } catch (const std::exception&) {
    }
}

struct SolveResult {
    cv::Mat normals;
    cv::Mat albedo;
    cv::Mat residual;
    cv::Mat validMask;
    PhotometricDiagnostics diagnostics;
};

cv::Vec3f normalized(const cv::Vec3f& value) {
    return value / std::sqrt(value.dot(value));
}

std::vector<cv::Vec3f> makeRingLights(int count, float z = 0.75f) {
    std::vector<cv::Vec3f> lights;
    lights.reserve(static_cast<size_t>(count));
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

std::vector<cv::Mat> renderLambertianPlane(
    int rows,
    int cols,
    const std::vector<cv::Vec3f>& lights,
    const cv::Vec3f& normal,
    float albedo) {
    std::vector<cv::Mat> images;
    images.reserve(lights.size());
    for (const cv::Vec3f& light : lights) {
        const float intensity = albedo * std::max(0.0f, normal.dot(light));
        images.emplace_back(rows, cols, CV_32F, cv::Scalar(intensity));
    }
    return images;
}

std::vector<cv::Mat> encodeLinearStack(
    const std::vector<cv::Mat>& linearImages,
    int outputDepth,
    double maximumCode,
    bool encodeSrgb) {
    std::vector<cv::Mat> converted;
    converted.reserve(linearImages.size());
    for (const cv::Mat& linear : linearImages) {
        cv::Mat encodedFloat = linear.clone();
        if (encodeSrgb) {
            for (int y = 0; y < encodedFloat.rows; ++y) {
                float* row = encodedFloat.ptr<float>(y);
                for (int x = 0; x < encodedFloat.cols; ++x) {
                    const float value = std::clamp(row[x], 0.0f, 1.0f);
                    row[x] = value <= 0.0031308f
                        ? 12.92f * value
                        : 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
                }
            }
        }
        cv::Mat encoded;
        encodedFloat.convertTo(encoded, outputDepth, maximumCode);
        converted.push_back(convertToLinearLuminance(encoded, encodeSrgb));
    }
    normalizeRelativeIntensityStack(converted);
    return converted;
}

SolveResult solve(
    const std::vector<cv::Mat>& images,
    const std::vector<cv::Vec3f>& lights,
    NormalSolverMode solverMode,
    LightingModel lightingModel = LightingModel::Directional,
    double ringRadiusMm = 10.0,
    double ringHeightMm = 10.0,
    double pixelScaleMm = 0.01,
    const std::vector<cv::Mat>& saturationMasks = {},
    bool collectObservationMasks = false,
    const cv::Mat& inputMask = cv::Mat()) {
    SolveResult result;
    result.diagnostics.collectObservationMasks = collectObservationMasks;
    const cv::Mat mask = inputMask.empty()
        ? cv::Mat(images.front().size(), CV_8U, cv::Scalar(255))
        : inputMask;
    solvePhotometricStereo(
        images,
        lights,
        mask,
        0.02f,
        solverMode,
        0.98f,
        lightingModel,
        ringRadiusMm,
        ringHeightMm,
        pixelScaleMm,
        cv::Point2d(
            static_cast<double>(images.front().cols - 1) * 0.5,
            static_cast<double>(images.front().rows - 1) * 0.5),
        cv::Vec3f(0.0f, 0.0f, 1.0f),
        result.normals,
        result.albedo,
        result.residual,
        result.validMask,
        result.diagnostics,
        saturationMasks);
    return result;
}

std::vector<cv::Mat> renderNearFieldPlane(
    int rows,
    int cols,
    const std::vector<cv::Vec3f>& referenceLights,
    const cv::Vec3f& normal,
    float albedo,
    double ringRadiusMm,
    double ringHeightMm,
    double pixelScaleMm) {
    const cv::Point2d center(
        static_cast<double>(cols - 1) * 0.5,
        static_cast<double>(rows - 1) * 0.5);
    const double referenceDistanceSquared =
        ringRadiusMm * ringRadiusMm + ringHeightMm * ringHeightMm;
    std::vector<cv::Mat> images;
    images.reserve(referenceLights.size());
    for (size_t i = 0; i < referenceLights.size(); ++i) {
        cv::Mat image(rows, cols, CV_32F, cv::Scalar(0));
        double ax = referenceLights[i][0];
        double ay = referenceLights[i][1];
        const double radialLength = std::sqrt(ax * ax + ay * ay);
        if (radialLength > 1.0e-8) {
            ax /= radialLength;
            ay /= radialLength;
        } else {
            const double angle = 2.0 * kPi * static_cast<double>(i) /
                static_cast<double>(referenceLights.size());
            ax = std::cos(angle);
            ay = std::sin(angle);
        }
        const cv::Vec3d source(ringRadiusMm * ax, ringRadiusMm * ay, ringHeightMm);
        for (int y = 0; y < rows; ++y) {
            float* row = image.ptr<float>(y);
            for (int x = 0; x < cols; ++x) {
                const cv::Vec3d point(
                    (static_cast<double>(x) - center.x) * pixelScaleMm,
                    (center.y - static_cast<double>(y)) * pixelScaleMm,
                    0.0);
                const cv::Vec3d displacement = source - point;
                const double distanceSquared = displacement.dot(displacement);
                const cv::Vec3d direction = displacement / std::sqrt(distanceSquared);
                const double irradiance = referenceDistanceSquared / distanceSquared;
                row[x] = static_cast<float>(
                    albedo * irradiance * std::max(
                        0.0,
                        direction.dot(cv::Vec3d(normal[0], normal[1], normal[2]))));
            }
        }
        images.push_back(std::move(image));
    }
    return images;
}

struct MitsubaFixture {
    std::string name;
    std::vector<cv::Mat> images;
    std::vector<cv::Mat> saturationMasks;
    std::vector<cv::Mat> shadowTruth;
    std::vector<cv::Mat> attachedShadowTruth;
    std::vector<cv::Mat> castShadowTruth;
    std::vector<cv::Mat> highlightTruth;
    std::vector<cv::Vec3f> lights;
    cv::Mat expectedNormals;
    cv::Mat solveMask;
    cv::Mat shapeIndex;
    cv::Mat positionZ;
    LightingModel lightingModel = LightingModel::Directional;
    double ringRadiusMm = 10.0;
    double ringHeightMm = 10.0;
    double pixelScaleMm = 0.01;
};

cv::Mat readRequiredImage(const fs::path& path, int flags) {
    const cv::Mat image = cv::imread(path.string(), flags);
    if (image.empty()) {
        throw std::runtime_error("Could not read test fixture image: " + path.string());
    }
    return image;
}

MitsubaFixture loadMitsubaFixture(
    const std::string& name,
    LightingModel lightingModel = LightingModel::Directional,
    double ringRadiusMm = 10.0,
    double ringHeightMm = 10.0,
    double pixelScaleMm = 0.01) {
#ifndef WHAT_A_RELIEF_SOURCE_DIR
#error WHAT_A_RELIEF_SOURCE_DIR must identify the source tree for committed test fixtures.
#endif
    const fs::path root = fs::path(WHAT_A_RELIEF_SOURCE_DIR) /
        "tests" / "fixtures" / "mitsuba" / name;
    MitsubaFixture fixture;
    fixture.name = name;
    fixture.lightingModel = lightingModel;
    fixture.ringRadiusMm = ringRadiusMm;
    fixture.ringHeightMm = ringHeightMm;
    fixture.pixelScaleMm = pixelScaleMm;
    fixture.solveMask = readRequiredImage(root / "solve_mask.png", cv::IMREAD_GRAYSCALE);
    fixture.shapeIndex = readRequiredImage(root / "shape_index.png", cv::IMREAD_GRAYSCALE);
    const cv::Mat encodedPositionZ = readRequiredImage(
        root / "position_z.png",
        cv::IMREAD_UNCHANGED);
    if (encodedPositionZ.type() != CV_16U || encodedPositionZ.size() != fixture.solveMask.size()) {
        throw std::runtime_error("Mitsuba position-z fixture must be a matching 16-bit image.");
    }
    encodedPositionZ.convertTo(fixture.positionZ, CV_32F, 1.25 / 65535.0, -0.25);

    std::vector<cv::Mat> components;
    for (const char axis : {'x', 'y', 'z'}) {
        const cv::Mat component = readRequiredImage(
            root / (std::string("normal_") + axis + ".png"),
            cv::IMREAD_UNCHANGED);
        if (component.type() != CV_16U || component.size() != fixture.solveMask.size()) {
            throw std::runtime_error("Mitsuba normal fixture must contain matching 16-bit components.");
        }
        components.push_back(component);
    }
    fixture.expectedNormals = cv::Mat(fixture.solveMask.size(), CV_32FC3);
    for (int y = 0; y < fixture.expectedNormals.rows; ++y) {
        cv::Vec3f* outputRow = fixture.expectedNormals.ptr<cv::Vec3f>(y);
        const unsigned short* xRow = components[0].ptr<unsigned short>(y);
        const unsigned short* yRow = components[1].ptr<unsigned short>(y);
        const unsigned short* zRow = components[2].ptr<unsigned short>(y);
        for (int x = 0; x < fixture.expectedNormals.cols; ++x) {
            outputRow[x] = normalized(cv::Vec3f(
                static_cast<float>(2.0 * xRow[x] / 65535.0 - 1.0),
                static_cast<float>(2.0 * yRow[x] / 65535.0 - 1.0),
                static_cast<float>(2.0 * zRow[x] / 65535.0 - 1.0)));
        }
    }

    std::vector<std::string> imagePaths;
    std::vector<fs::path> fixtureImages;
    for (const fs::directory_entry& entry : fs::directory_iterator(root / "images")) {
        if (entry.is_regular_file() && entry.path().extension() == ".png") {
            fixtureImages.push_back(entry.path());
        }
    }
    std::sort(fixtureImages.begin(), fixtureImages.end());
    if (fixtureImages.size() < 3) {
        throw std::runtime_error("Mitsuba fixture must contain at least three input images: " + name);
    }
    for (const fs::path& imagePath : fixtureImages) {
        const fs::path filename = imagePath.filename();
        imagePaths.push_back(imagePath.string());
        fixture.images.push_back(convertToLinearLuminance(
            readRequiredImage(imagePath, cv::IMREAD_UNCHANGED),
            false));
        fixture.shadowTruth.push_back(readRequiredImage(
            root / "shadow_truth" / filename,
            cv::IMREAD_GRAYSCALE));
        fixture.attachedShadowTruth.push_back(readRequiredImage(
            root / "attached_shadow_truth" / filename,
            cv::IMREAD_GRAYSCALE));
        fixture.castShadowTruth.push_back(readRequiredImage(
            root / "cast_shadow_truth" / filename,
            cv::IMREAD_GRAYSCALE));
        fixture.highlightTruth.push_back(readRequiredImage(
            root / "highlight_truth" / filename,
            cv::IMREAD_GRAYSCALE));
        fixture.saturationMasks.push_back(readRequiredImage(
            root / "saturation_truth" / filename,
            cv::IMREAD_GRAYSCALE));
    }
    fixture.lights = loadLightsFile((root / "lights.csv").string(), imagePaths);
    return fixture;
}

struct ClassificationMetrics {
    double precision = 1.0;
    double recall = 1.0;
    int truthCount = 0;
    int predictedCount = 0;
};

ClassificationMetrics classificationMetrics(
    const std::vector<cv::Mat>& predicted,
    const std::vector<cv::Mat>& truth,
    const cv::Mat& evaluationMask = cv::Mat()) {
    int truePositive = 0;
    int falsePositive = 0;
    int falseNegative = 0;
    for (size_t i = 0; i < truth.size(); ++i) {
        for (int y = 0; y < truth[i].rows; ++y) {
            const uchar* predictedRow = predicted[i].ptr<uchar>(y);
            const uchar* truthRow = truth[i].ptr<uchar>(y);
            const uchar* evaluationRow = evaluationMask.empty()
                ? nullptr
                : evaluationMask.ptr<uchar>(y);
            for (int x = 0; x < truth[i].cols; ++x) {
                if (evaluationRow != nullptr && evaluationRow[x] == 0) {
                    continue;
                }
                const bool predictedValue = predictedRow[x] != 0;
                const bool truthValue = truthRow[x] != 0;
                truePositive += predictedValue && truthValue ? 1 : 0;
                falsePositive += predictedValue && !truthValue ? 1 : 0;
                falseNegative += !predictedValue && truthValue ? 1 : 0;
            }
        }
    }
    ClassificationMetrics metrics;
    metrics.truthCount = truePositive + falseNegative;
    metrics.predictedCount = truePositive + falsePositive;
    metrics.precision = metrics.predictedCount > 0
        ? static_cast<double>(truePositive) / metrics.predictedCount
        : 1.0;
    metrics.recall = metrics.truthCount > 0
        ? static_cast<double>(truePositive) / metrics.truthCount
        : 1.0;
    return metrics;
}

std::vector<cv::Mat> unionMasks(
    const std::vector<cv::Mat>& first,
    const std::vector<cv::Mat>& second) {
    std::vector<cv::Mat> result;
    result.reserve(first.size());
    for (size_t i = 0; i < first.size(); ++i) {
        cv::Mat combined;
        cv::bitwise_or(first[i], second[i], combined);
        result.push_back(std::move(combined));
    }
    return result;
}

cv::Mat unionAcrossLights(const std::vector<cv::Mat>& masks) {
    cv::Mat result(masks.front().size(), CV_8U, cv::Scalar(0));
    for (const cv::Mat& mask : masks) {
        cv::bitwise_or(result, mask, result);
    }
    return result;
}

double meanAngularErrorDegrees(
    const cv::Mat& normals,
    const cv::Mat& validMask,
    const cv::Vec3f& expected) {
    double sum = 0.0;
    int count = 0;
    for (int y = 0; y < normals.rows; ++y) {
        const cv::Vec3f* normalRow = normals.ptr<cv::Vec3f>(y);
        const uchar* maskRow = validMask.ptr<uchar>(y);
        for (int x = 0; x < normals.cols; ++x) {
            if (maskRow[x] == 0) {
                continue;
            }
            const double cosine = std::clamp(
                static_cast<double>(normalRow[x].dot(expected)),
                -1.0,
                1.0);
            sum += std::acos(cosine) * kRadiansToDegrees;
            ++count;
        }
    }
    return count > 0 ? sum / static_cast<double>(count) : 180.0;
}

double meanAngularErrorDegrees(
    const cv::Mat& normals,
    const cv::Mat& validMask,
    const cv::Mat& expectedNormals) {
    double sum = 0.0;
    int count = 0;
    for (int y = 0; y < normals.rows; ++y) {
        const cv::Vec3f* normalRow = normals.ptr<cv::Vec3f>(y);
        const cv::Vec3f* expectedRow = expectedNormals.ptr<cv::Vec3f>(y);
        const uchar* maskRow = validMask.ptr<uchar>(y);
        for (int x = 0; x < normals.cols; ++x) {
            if (maskRow[x] == 0) {
                continue;
            }
            const double cosine = std::clamp(
                static_cast<double>(normalRow[x].dot(expectedRow[x])),
                -1.0,
                1.0);
            sum += std::acos(cosine) * kRadiansToDegrees;
            ++count;
        }
    }
    return count > 0 ? sum / static_cast<double>(count) : 180.0;
}

double maskedMean(const cv::Mat& values, const cv::Mat& mask) {
    double sum = 0.0;
    int count = 0;
    for (int y = 0; y < values.rows; ++y) {
        const float* valueRow = values.ptr<float>(y);
        const uchar* maskRow = mask.ptr<uchar>(y);
        for (int x = 0; x < values.cols; ++x) {
            if (maskRow[x] != 0) {
                sum += valueRow[x];
                ++count;
            }
        }
    }
    return count > 0 ? sum / static_cast<double>(count) : 0.0;
}

double normalizedHeightRmse(
    const cv::Mat& actual,
    const cv::Mat& expected,
    const cv::Mat& mask) {
    const double actualMean = maskedMean(actual, mask);
    const double expectedMean = maskedMean(expected, mask);
    double squaredError = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
    int count = 0;
    bool first = true;
    for (int y = 0; y < actual.rows; ++y) {
        const float* actualRow = actual.ptr<float>(y);
        const float* expectedRow = expected.ptr<float>(y);
        const uchar* maskRow = mask.ptr<uchar>(y);
        for (int x = 0; x < actual.cols; ++x) {
            if (maskRow[x] == 0) {
                continue;
            }
            const double expectedCentered = expectedRow[x] - expectedMean;
            const double difference = (actualRow[x] - actualMean) - expectedCentered;
            squaredError += difference * difference;
            if (first) {
                minimum = expectedCentered;
                maximum = expectedCentered;
                first = false;
            } else {
                minimum = std::min(minimum, expectedCentered);
                maximum = std::max(maximum, expectedCentered);
            }
            ++count;
        }
    }
    if (count == 0 || maximum <= minimum) {
        return 1.0;
    }
    return std::sqrt(squaredError / static_cast<double>(count)) / (maximum - minimum);
}

void makeHeightFixture(cv::Mat& height, cv::Mat& normals) {
    constexpr int rows = 65;
    constexpr int cols = 81;
    height = cv::Mat(rows, cols, CV_32F);
    for (int y = 0; y < rows; ++y) {
        float* row = height.ptr<float>(y);
        const double fy = static_cast<double>(y) / static_cast<double>(rows - 1);
        for (int x = 0; x < cols; ++x) {
            const double fx = static_cast<double>(x) / static_cast<double>(cols - 1);
            row[x] = static_cast<float>(
                1.7 * std::cos(kPi * fx) +
                0.9 * std::cos(2.0 * kPi * fy) +
                0.35 * std::cos(2.0 * kPi * fx) * std::cos(kPi * fy));
        }
    }

    normals = cv::Mat(rows, cols, CV_32FC3);
    for (int y = 0; y < rows; ++y) {
        cv::Vec3f* normalRow = normals.ptr<cv::Vec3f>(y);
        const float* row = height.ptr<float>(y);
        const float* nextRow = y + 1 < rows ? height.ptr<float>(y + 1) : nullptr;
        for (int x = 0; x < cols; ++x) {
            const float p = x + 1 < cols ? row[x + 1] - row[x] : 0.0f;
            const float q = y + 1 < rows ? nextRow[x] - row[x] : 0.0f;
            normalRow[x] = normalized(cv::Vec3f(-p, q, 1.0f));
        }
    }
}

void testCleanLambertianCounts(TestContext& context) {
    const cv::Vec3f expected = normalized(cv::Vec3f(0.16f, -0.11f, 1.0f));
    for (const int count : {3, 4, 8, 25, 64}) {
        const std::vector<cv::Vec3f> lights = makeRingLights(count);
        const SolveResult result = solve(
            renderLambertianPlane(19, 23, lights, expected, 0.72f),
            lights,
            NormalSolverMode::Robust);
        const double error = meanAngularErrorDegrees(result.normals, result.validMask, expected);
        std::cout << "clean_lambertian_" << count << "_mean_degrees=" << error << '\n';
        context.check(
            cv::countNonZero(result.validMask) == result.validMask.rows * result.validMask.cols,
            "clean " + std::to_string(count) + "-light solve must retain every pixel");
        context.check(
            error < 0.05,
            "clean " + std::to_string(count) + "-light mean angular error was " + std::to_string(error));
        context.check(
            std::isfinite(result.diagnostics.lightingConditionNumber) &&
                result.diagnostics.lightingConditionNumber < 10.0,
            "clean " + std::to_string(count) + "-light geometry must be well-conditioned");
        context.check(
            std::abs(result.diagnostics.solvedFraction - 1.0) < 1.0e-12,
            "clean " + std::to_string(count) + "-light solve must report complete coverage");
        const int fallbackPixels = cv::countNonZero(result.diagnostics.robustFallbackMask);
        context.check(
            fallbackPixels == (count == 3 ? result.validMask.rows * result.validMask.cols : 0),
            "robust fallback mask mismatch for " + std::to_string(count) + " lights");
    }
}

void testCalibrationIdentityAndValidation(TestContext& context) {
    const fs::path namedPath = "calibration_identity_test.csv";
    {
        std::ofstream out(namedPath);
        out << "lighting_model,directional\n";
        out << "image,highlight_x,highlight_y,light_x,light_y,light_z,threshold,peak,selected_pixels\n";
        out << "\"D:/old/location/sample_02.tif\",0,0,0,1,1,0,0,0\n";
        out << "\"D:/old/location/sample_01.tif\",0,0,1,0,1,0,0,0\n";
    }
    const std::vector<std::string> selected{
        "C:/moved/project/sample_01.tif",
        "C:/moved/project/sample_02.tif"};
    const std::vector<cv::Vec3f> reordered = loadLightsFile(namedPath.string(), selected);
    context.check(
        reordered.size() == 2 && reordered[0].dot(normalized(cv::Vec3f(1.0f, 0.0f, 1.0f))) > 0.99999f &&
            reordered[1].dot(normalized(cv::Vec3f(0.0f, 1.0f, 1.0f))) > 0.99999f,
        "named calibration rows must follow selected image identity rather than CSV row order");
    const std::vector<cv::Vec3f> orderOverride = loadLightsFile(
        namedPath.string(),
        {"C:/new/project/renamed_a.tif", "C:/new/project/renamed_b.tif"},
        nullptr,
        true);
    context.check(
        orderOverride.size() == 2 &&
            orderOverride[0].dot(normalized(cv::Vec3f(0.0f, 1.0f, 1.0f))) > 0.99999f &&
            orderOverride[1].dot(normalized(cv::Vec3f(1.0f, 0.0f, 1.0f))) > 0.99999f,
        "explicit calibration order override must ignore stored image names and preserve CSV row order");
    expectThrows(
        context,
        [&]() {
            (void)loadLightsFile(
                namedPath.string(),
                {"C:/new/project/only_one_image.tif"},
                nullptr,
                true);
        },
        "calibration order override must still require one light row per selected image");
    expectThrows(
        context,
        [&]() {
            (void)loadLightsFile(
                namedPath.string(),
                {"C:/moved/project/sample_01.tif", "C:/moved/project/missing.tif"});
        },
        "named calibration must reject a missing selected image");
    fs::remove(namedPath);

    const fs::path ambiguousPath = "calibration_ambiguous_test.csv";
    {
        std::ofstream out(ambiguousPath);
        out << "image,highlight_x,highlight_y,light_x,light_y,light_z\n";
        out << "\"D:/first/repeated.tif\",0,0,1,0,1\n";
        out << "\"D:/second/repeated.tif\",0,0,0,1,1\n";
    }
    expectThrows(
        context,
        [&]() {
            (void)loadLightsFile(
                ambiguousPath.string(),
                {"C:/moved/repeated.tif", "C:/moved/another.tif"});
        },
        "named calibration must reject ambiguous filename matching");
    fs::remove(ambiguousPath);

    const fs::path vectorPath = "calibration_vectors_test.csv";
    {
        std::ofstream out(vectorPath);
        out << "x,y,z\n";
        out << "0,1,1\n";
        out << "1,0,1\n";
    }
    const std::vector<cv::Vec3f> positional = loadLightsFile(vectorPath.string(), selected);
    context.check(
        positional.size() == 2 && positional[0].dot(normalized(cv::Vec3f(0.0f, 1.0f, 1.0f))) > 0.99999f,
        "vector-only calibration must retain documented positional row order");
    fs::remove(vectorPath);

    const std::vector<cv::Vec3f> degenerateLights(8, normalized(cv::Vec3f(0.2f, 0.1f, 1.0f)));
    const std::vector<cv::Mat> degenerateImages = renderLambertianPlane(
        7, 9, degenerateLights, normalized(cv::Vec3f(0.1f, -0.1f, 1.0f)), 0.7f);
    expectThrows(
        context,
        [&]() { (void)solve(degenerateImages, degenerateLights, NormalSolverMode::Robust); },
        "rank-deficient lighting must be rejected before pixel solving");

    const std::vector<cv::Vec3f> narrowConeLights = makeRingLights(8, 0.001f);
    const std::vector<cv::Mat> narrowConeImages = renderLambertianPlane(
        7, 9, narrowConeLights, normalized(cv::Vec3f(0.1f, -0.1f, 1.0f)), 0.7f);
    expectThrows(
        context,
        [&]() { (void)solve(narrowConeImages, narrowConeLights, NormalSolverMode::Robust); },
        "full-rank lighting with condition number above the guard must be rejected");

    const fs::path nonfinitePath = "calibration_nonfinite_test.csv";
    {
        std::ofstream out(nonfinitePath);
        out << "x,y,z\n";
        out << "0,1,1\n";
        out << "nan,0,1\n";
    }
    expectThrows(
        context,
        [&]() { (void)loadLightsFile(nonfinitePath.string(), selected); },
        "non-finite calibration vectors must be rejected");
    fs::remove(nonfinitePath);

    const std::vector<cv::Vec3f> ringLights = makeRingLights(8);
    std::vector<cv::Mat> darkImages;
    for (size_t i = 0; i < ringLights.size(); ++i) {
        darkImages.emplace_back(8, 8, CV_32F, cv::Scalar(0));
    }
    expectThrows(
        context,
        [&]() { (void)solve(darkImages, ringLights, NormalSolverMode::Robust); },
        "a stack with no spatial solve coverage must be rejected");
}

void testNearFieldRingModel(TestContext& context) {
    constexpr int rows = 61;
    constexpr int cols = 81;
    constexpr double radiusMm = 14.0;
    constexpr double heightMm = 9.0;
    constexpr double pixelScaleMm = 0.22;
    const float centerZ = static_cast<float>(
        heightMm / std::sqrt(radiusMm * radiusMm + heightMm * heightMm));
    const std::vector<cv::Vec3f> lights = makeRingLights(8, centerZ);
    const cv::Vec3f expected = normalized(cv::Vec3f(0.13f, -0.09f, 1.0f));
    const std::vector<cv::Mat> images = renderNearFieldPlane(
        rows, cols, lights, expected, 0.64f, radiusMm, heightMm, pixelScaleMm);

    const SolveResult directional = solve(images, lights, NormalSolverMode::Standard);
    const SolveResult nearField = solve(
        images,
        lights,
        NormalSolverMode::Standard,
        LightingModel::NearFieldRing,
        radiusMm,
        heightMm,
        pixelScaleMm);
    const double directionalError = meanAngularErrorDegrees(
        directional.normals, directional.validMask, expected);
    const double nearFieldError = meanAngularErrorDegrees(
        nearField.normals, nearField.validMask, expected);
    const double meanAlbedo = cv::mean(nearField.albedo, nearField.validMask)[0];
    std::cout << "near_field_directional_mean_degrees=" << directionalError << '\n';
    std::cout << "near_field_corrected_mean_degrees=" << nearFieldError << '\n';
    std::cout << "near_field_corrected_mean_albedo=" << meanAlbedo << '\n';
    context.check(
        directionalError > 2.0,
        "near-field fixture must expose the distant-light model error");
    context.check(
        nearFieldError < 0.05,
        "inverse-square near-field mean angular error was " + std::to_string(nearFieldError));
    context.check(
        nearFieldError < directionalError * 0.02,
        "near-field model must reduce angular error by at least 98 percent");
    context.check(
        std::abs(meanAlbedo - 0.64) < 1.0e-4,
        "near-field attenuation correction must recover spatially consistent albedo");
}

void testRadiometricConversion(TestContext& context) {
    cv::Mat encoded8 = (cv::Mat_<uchar>(1, 4) << 0, 64, 128, 255);
    cv::Mat encoded16 = (cv::Mat_<unsigned short>(1, 4) << 0, 16384, 32768, 65535);
    cv::Mat encoded12 = (cv::Mat_<unsigned short>(1, 4) << 0, 1024, 2048, 4095);
    cv::Mat encodedFloat = (cv::Mat_<float>(1, 4) << 0.0f, 0.25f, 0.5f, 1.0f);

    std::vector<cv::Mat> stack8{convertToLinearLuminance(encoded8, false)};
    std::vector<cv::Mat> stack16{convertToLinearLuminance(encoded16, false)};
    std::vector<cv::Mat> stack12{convertToLinearLuminance(encoded12, false)};
    std::vector<cv::Mat> stackFloat{convertToLinearLuminance(encodedFloat, false)};
    normalizeRelativeIntensityStack(stack8);
    normalizeRelativeIntensityStack(stack16);
    normalizeRelativeIntensityStack(stack12);
    normalizeRelativeIntensityStack(stackFloat);

    const std::vector<float> expected{0.0f, 0.25f, 0.5f, 1.0f};
    for (int x = 0; x < 4; ++x) {
        context.check(std::abs(stack8[0].at<float>(0, x) - expected[x]) < 0.003f, "8-bit intensity conversion mismatch");
        context.check(std::abs(stack16[0].at<float>(0, x) - expected[x]) < 0.0001f, "16-bit intensity conversion mismatch");
        context.check(std::abs(stack12[0].at<float>(0, x) - expected[x]) < 0.0002f, "12-bit-in-16-bit intensity conversion mismatch");
        context.check(std::abs(stackFloat[0].at<float>(0, x) - expected[x]) < 0.0001f, "float intensity conversion mismatch");
    }

    cv::Mat exposureA(1, 1, CV_16U, cv::Scalar(1000));
    cv::Mat exposureB(1, 1, CV_16U, cv::Scalar(2000));
    std::vector<cv::Mat> exposureStack{
        convertToLinearLuminance(exposureA, false),
        convertToLinearLuminance(exposureB, false)};
    normalizeRelativeIntensityStack(exposureStack);
    context.check(
        std::abs(exposureStack[0].at<float>(0, 0) - 0.5f) < 0.0001f &&
            std::abs(exposureStack[1].at<float>(0, 0) - 1.0f) < 0.0001f,
        "stack normalization must preserve inter-image exposure ratios");

    cv::Mat color(1, 1, CV_8UC3);
    color.at<cv::Vec3b>(0, 0) = cv::Vec3b(32, 128, 240);
    const cv::Mat linearLuminance = convertToLinearLuminance(color, true);
    const auto decode = [](double value) {
        value /= 255.0;
        return value <= 0.04045 ? value / 12.92 : std::pow((value + 0.055) / 1.055, 2.4);
    };
    const double expectedLuminance = 0.0722 * decode(32.0) + 0.7152 * decode(128.0) + 0.2126 * decode(240.0);
    context.check(
        std::abs(static_cast<double>(linearLuminance.at<float>(0, 0)) - expectedLuminance) < 1.0e-6,
        "sRGB channels must be linearized before luminance conversion");

    const cv::Vec3f linearColor = convertToLinearColor(color, true).at<cv::Vec3f>(0, 0);
    context.check(
        std::abs(static_cast<double>(linearColor[0]) - decode(32.0)) < 1.0e-6 &&
            std::abs(static_cast<double>(linearColor[1]) - decode(128.0)) < 1.0e-6 &&
            std::abs(static_cast<double>(linearColor[2]) - decode(240.0)) < 1.0e-6,
        "RTI color conversion must use the same exact piecewise sRGB decoding as luminance conversion");

    std::vector<cv::Mat> floatingColorStack{
        cv::Mat(1, 1, CV_32FC3, cv::Scalar(2.0f, 4.0f, 6.0f)),
        cv::Mat(1, 1, CV_32FC3, cv::Scalar(4.0f, 8.0f, 12.0f))};
    normalizeRelativeIntensityStack(floatingColorStack);
    const cv::Vec3f firstColor = floatingColorStack[0].at<cv::Vec3f>(0, 0);
    const cv::Vec3f secondColor = floatingColorStack[1].at<cv::Vec3f>(0, 0);
    context.check(
        std::abs(firstColor[2] - 0.5f) < 1.0e-6f &&
            std::abs(secondColor[2] - 1.0f) < 1.0e-6f &&
            std::abs(firstColor[1] * 2.0f - secondColor[1]) < 1.0e-6f,
        "color-stack normalization must use one common scale and preserve light-to-light ratios");
}

void testObservationValidityMask(TestContext& context) {
    std::vector<cv::Mat> images;
    for (int i = 0; i < 4; ++i) {
        images.emplace_back(1, 5, CV_32F, cv::Scalar(0.5f));
    }
    images[0].at<float>(0, 1) = 0.0f;
    images[1].at<float>(0, 1) = 0.0f;
    images[0].at<float>(0, 2) = std::numeric_limits<float>::quiet_NaN();
    images[1].at<float>(0, 3) = 0.02f;
    images[2].at<float>(0, 3) = 0.02f;
    cv::Mat solveMask(1, 5, CV_8U, cv::Scalar(255));
    solveMask.at<uchar>(0, 4) = 0;

    const cv::Mat validity = buildObservationValidityMask(images, solveMask, 0.02f, 3);
    context.check(validity.at<uchar>(0, 0) == 255, "four illuminated observations must be valid");
    context.check(validity.at<uchar>(0, 1) == 0, "two illuminated observations must be invalid");
    context.check(validity.at<uchar>(0, 2) == 255, "one non-finite sample must leave three valid observations");
    context.check(validity.at<uchar>(0, 3) == 0, "samples equal to the shadow threshold must be omitted");
    context.check(validity.at<uchar>(0, 4) == 0, "the solve mask must bound neural validity");
}

void testEncodedNormalRecovery(TestContext& context) {
    const cv::Vec3f expected = normalized(cv::Vec3f(0.21f, -0.16f, 1.0f));
    const std::vector<cv::Vec3f> lights = makeRingLights(8);
    const std::vector<cv::Mat> linear = renderLambertianPlane(13, 15, lights, expected, 0.73f);

    struct EncodingCase {
        const char* name;
        int depth;
        double maximumCode;
        bool srgb;
        double maximumErrorDegrees;
    };
    const std::vector<EncodingCase> cases{
        {"linear_8_bit", CV_8U, 255.0, false, 0.35},
        {"linear_12_in_16_bit", CV_16U, 4095.0, false, 0.05},
        {"linear_16_bit", CV_16U, 65535.0, false, 0.05},
        {"linear_float_range_37", CV_32F, 37.0, false, 0.05},
        {"srgb_8_bit", CV_8U, 255.0, true, 0.50}};

    for (const EncodingCase& encoding : cases) {
        const SolveResult result = solve(
            encodeLinearStack(linear, encoding.depth, encoding.maximumCode, encoding.srgb),
            lights,
            NormalSolverMode::Robust);
        const double error = meanAngularErrorDegrees(result.normals, result.validMask, expected);
        std::cout << encoding.name << "_mean_degrees=" << error << '\n';
        context.check(
            error < encoding.maximumErrorDegrees,
            std::string(encoding.name) + " mean angular error was " + std::to_string(error));
    }

    std::vector<cv::Mat> highlighted = linear;
    highlighted[0] = cv::Mat(linear[0].size(), CV_32F, cv::Scalar(1.0));
    const SolveResult robust12 = solve(
        encodeLinearStack(highlighted, CV_16U, 4095.0, false),
        lights,
        NormalSolverMode::Robust);
    const double robust12Error = meanAngularErrorDegrees(robust12.normals, robust12.validMask, expected);
    std::cout << "linear_12_in_16_bit_highlight_robust_mean_degrees=" << robust12Error << '\n';
    context.check(
        robust12Error < 0.10,
        "12-bit-in-16-bit robust highlight error was " + std::to_string(robust12Error));
}

void testShadowOmission(TestContext& context) {
    const cv::Vec3f expected = normalized(cv::Vec3f(0.19f, -0.13f, 1.0f));
    const std::vector<cv::Vec3f> lights = makeRingLights(8);
    std::vector<cv::Mat> images = renderLambertianPlane(17, 21, lights, expected, 0.68f);
    images[2].setTo(cv::Scalar(0));
    const SolveResult result = solve(images, lights, NormalSolverMode::Robust);
    const double error = meanAngularErrorDegrees(result.normals, result.validMask, expected);
    const double meanShadowCount = cv::mean(result.diagnostics.shadowCount)[0];
    std::cout << "hard_shadow_mean_degrees=" << error << '\n';
    context.check(error < 0.05, "hard-shadow omission mean angular error was " + std::to_string(error));
    context.check(
        std::abs(meanShadowCount - 1.0) < 1.0e-6,
        "hard-shadow fixture must report one omitted observation per pixel");
}

void testPenumbraRobustness(TestContext& context) {
    const cv::Vec3f expected = normalized(cv::Vec3f(0.19f, -0.13f, 1.0f));
    const std::vector<cv::Vec3f> lights = makeRingLights(8);
    std::vector<cv::Mat> images = renderLambertianPlane(17, 21, lights, expected, 0.68f);
    images[2] *= 0.5f;

    const SolveResult standard = solve(images, lights, NormalSolverMode::Standard);
    const SolveResult robust = solve(images, lights, NormalSolverMode::Robust);
    const double standardError = meanAngularErrorDegrees(standard.normals, standard.validMask, expected);
    const double robustError = meanAngularErrorDegrees(robust.normals, robust.validMask, expected);
    const double shadowCount = cv::mean(robust.diagnostics.shadowCount)[0];
    const double specularCueRate = static_cast<double>(cv::countNonZero(robust.diagnostics.specularCueMask)) /
        static_cast<double>(robust.validMask.rows * robust.validMask.cols);
    std::cout << "penumbra_standard_mean_degrees=" << standardError << '\n';
    std::cout << "penumbra_robust_mean_degrees=" << robustError << '\n';
    std::cout << "penumbra_shadow_outlier_count=" << shadowCount << '\n';
    context.check(standardError > 5.0, "penumbra fixture must exercise least-squares failure");
    context.check(robustError < 1.0, "penumbra robust mean angular error was " + std::to_string(robustError));
    context.check(
        robustError < standardError * 0.20,
        "penumbra robust solve must improve mean angular error by at least 80 percent");
    context.check(shadowCount >= 0.99, "penumbra must be reported as a shadow outlier");
    context.check(specularCueRate < 0.01, "penumbra must not be mislabeled as a specular cue");
}

void testNarrowHighlightRobustness(TestContext& context) {
    const cv::Vec3f expected = normalized(cv::Vec3f(0.22f, -0.18f, 1.0f));
    const std::vector<cv::Vec3f> lights = makeRingLights(8);
    std::vector<cv::Mat> images = renderLambertianPlane(17, 21, lights, expected, 0.66f);
    images[0].setTo(cv::Scalar(1.0));

    const SolveResult standard = solve(images, lights, NormalSolverMode::Standard);
    const SolveResult robust = solve(images, lights, NormalSolverMode::Robust);
    const double standardError = meanAngularErrorDegrees(standard.normals, standard.validMask, expected);
    const double robustError = meanAngularErrorDegrees(robust.normals, robust.validMask, expected);
    const double meanOutlierCount = cv::mean(robust.diagnostics.highlightOutlierCount)[0];
    std::cout << "narrow_highlight_standard_mean_degrees=" << standardError << '\n';
    std::cout << "narrow_highlight_robust_mean_degrees=" << robustError << '\n';

    context.check(standardError > 2.0, "highlight fixture must exercise the non-robust failure mode");
    context.check(robustError < 0.10, "robust highlight mean angular error was " + std::to_string(robustError));
    context.check(
        robustError < standardError * 0.10,
        "robust highlight solve must improve angular error by at least 90 percent");
    context.check(
        meanOutlierCount >= 0.99,
        "robust highlight fixture must report the injected bright outlier");
}

void testFourLightSaturatedHighlight(TestContext& context) {
    const cv::Vec3f expected = normalized(cv::Vec3f(0.18f, -0.14f, 1.0f));
    const std::vector<cv::Vec3f> lights = makeRingLights(4);
    std::vector<cv::Mat> images = renderLambertianPlane(11, 13, lights, expected, 0.68f);
    images[0].setTo(cv::Scalar(1.0));

    const SolveResult standard = solve(images, lights, NormalSolverMode::Standard);
    const SolveResult robust = solve(images, lights, NormalSolverMode::Robust);
    const double standardError = meanAngularErrorDegrees(standard.normals, standard.validMask, expected);
    const double robustError = meanAngularErrorDegrees(robust.normals, robust.validMask, expected);
    std::cout << "four_light_saturated_standard_mean_degrees=" << standardError << '\n';
    std::cout << "four_light_saturated_robust_mean_degrees=" << robustError << '\n';
    context.check(standardError > 2.0, "four-light highlight fixture must exercise least-squares failure");
    context.check(
        robustError < 0.10,
        "four-light robust saturated-highlight error was " + std::to_string(robustError));
}

void testRobustHighlightCounts(TestContext& context) {
    const cv::Vec3f expected = normalized(cv::Vec3f(0.17f, -0.12f, 1.0f));
    for (const int count : {5, 8, 25, 64}) {
        const std::vector<cv::Vec3f> lights = makeRingLights(count);
        std::vector<cv::Mat> images = renderLambertianPlane(9, 11, lights, expected, 0.67f);
        const int injectedOutliers = count == 5 ? 1 : count == 8 ? 2 : count == 25 ? 4 : 8;
        for (int i = 0; i < injectedOutliers; ++i) {
            images[static_cast<size_t>(i)].setTo(cv::Scalar(1.0));
        }
        const SolveResult robust = solve(images, lights, NormalSolverMode::Robust);
        const double error = meanAngularErrorDegrees(robust.normals, robust.validMask, expected);
        const double outlierCount =
            cv::mean(robust.diagnostics.highlightOutlierCount)[0] +
            cv::mean(robust.diagnostics.modelMismatchCount)[0];
        std::cout << "robust_" << count << "_light_highlight_mean_degrees=" << error << '\n';
        context.check(
            error < 0.10,
            std::to_string(count) + "-light robust highlight error was " + std::to_string(error));
        context.check(
            outlierCount >= static_cast<double>(injectedOutliers) - 0.01,
            std::to_string(count) + "-light solve did not report every injected bright outlier");
    }
}

void testBroadGlossRobustness(TestContext& context) {
    constexpr int rows = 31;
    constexpr int cols = 37;
    const std::vector<cv::Vec3f> lights = makeRingLights(8);
    cv::Mat expectedNormals(rows, cols, CV_32FC3);
    std::vector<cv::Mat> images;
    images.reserve(lights.size());
    for (size_t i = 0; i < lights.size(); ++i) {
        images.emplace_back(rows, cols, CV_32F, cv::Scalar(0));
    }

    const cv::Vec3f view(0.0f, 0.0f, 1.0f);
    for (int y = 0; y < rows; ++y) {
        cv::Vec3f* expectedRow = expectedNormals.ptr<cv::Vec3f>(y);
        const double v = 2.0 * static_cast<double>(y) / static_cast<double>(rows - 1) - 1.0;
        for (int x = 0; x < cols; ++x) {
            const double u = 2.0 * static_cast<double>(x) / static_cast<double>(cols - 1) - 1.0;
            const float p = static_cast<float>(0.34 * std::sin(kPi * u) * std::cos(0.5 * kPi * v));
            const float q = static_cast<float>(0.28 * std::cos(0.5 * kPi * u) * std::sin(kPi * v));
            const cv::Vec3f normal = normalized(cv::Vec3f(-p, q, 1.0f));
            expectedRow[x] = normal;
            for (size_t i = 0; i < lights.size(); ++i) {
                const cv::Vec3f halfVector = normalized(lights[i] + view);
                const float diffuse = 0.12f * std::max(0.0f, normal.dot(lights[i]));
                const float specular = 0.70f * std::pow(std::max(0.0f, normal.dot(halfVector)), 12.0f);
                images[i].at<float>(y, x) = std::min(1.0f, diffuse + specular);
            }
        }
    }

    const SolveResult standard = solve(images, lights, NormalSolverMode::Standard);
    const SolveResult robust = solve(images, lights, NormalSolverMode::Robust);
    const double standardError = meanAngularErrorDegrees(standard.normals, standard.validMask, expectedNormals);
    const double robustError = meanAngularErrorDegrees(robust.normals, robust.validMask, expectedNormals);
    const double cueRate = static_cast<double>(cv::countNonZero(robust.diagnostics.specularCueMask)) /
        static_cast<double>(rows * cols);
    std::cout << "broad_gloss_standard_mean_degrees=" << standardError << '\n';
    std::cout << "broad_gloss_robust_mean_degrees=" << robustError << '\n';
    std::cout << "broad_gloss_specular_cue_rate=" << cueRate << '\n';
    context.check(standardError > 10.0, "broad-gloss fixture must be meaningfully non-Lambertian");
    context.check(
        robustError < standardError,
        "broad-gloss robust solve must improve on ordinary least squares");
    context.check(
        cueRate > 0.35,
        "broad-gloss half-vector diagnostic cue rate was " + std::to_string(cueRate));
}

void testBrightDiffuseObservationIsRetained(TestContext& context) {
    const std::vector<cv::Vec3f> lights = makeRingLights(8);
    const cv::Vec3f expected = lights.front();
    const std::vector<cv::Mat> images = renderLambertianPlane(13, 17, lights, expected, 1.0f);
    const SolveResult robust = solve(images, lights, NormalSolverMode::Robust);
    const double error = meanAngularErrorDegrees(robust.normals, robust.validMask, expected);
    const double highCount = cv::mean(robust.diagnostics.highlightOutlierCount)[0];
    std::cout << "bright_diffuse_mean_degrees=" << error << '\n';
    std::cout << "bright_diffuse_high_outlier_count=" << highCount << '\n';
    context.check(error < 0.05, "bright diffuse observation mean angular error was " + std::to_string(error));
    context.check(
        highCount < 0.01,
        "a model-consistent diffuse value at normalized intensity 1 must not be labeled a highlight");
}

void testUnsupportedPixelsAreNotForceFit(TestContext& context) {
    constexpr int rows = 20;
    constexpr int cols = 24;
    const std::vector<cv::Vec3f> lights = makeRingLights(8);
    const cv::Vec3f expected = normalized(cv::Vec3f(0.12f, -0.09f, 1.0f));
    std::vector<cv::Mat> images = renderLambertianPlane(rows, cols, lights, expected, 0.62f);
    for (size_t i = 2; i < images.size(); ++i) {
        images[i](cv::Rect(0, 0, cols / 2, rows)).setTo(cv::Scalar(0));
    }

    const SolveResult robust = solve(images, lights, NormalSolverMode::Robust);
    const cv::Rect unsupportedHalf(0, 0, cols / 2, rows);
    const cv::Rect supportedHalf(cols / 2, 0, cols - cols / 2, rows);
    context.check(
        cv::countNonZero(robust.validMask(unsupportedHalf)) == 0,
        "pixels with only two informative observations must not be force-fit");
    context.check(
        cv::countNonZero(robust.diagnostics.unsupportedMask(unsupportedHalf)) == unsupportedHalf.area(),
        "insufficient-observation pixels must be marked unsupported");
    context.check(
        cv::countNonZero(robust.validMask(supportedHalf)) == supportedHalf.area(),
        "well-observed pixels beside an unsupported region must remain solved");
}

void testMitsubaMixedMaterialRecovery(TestContext& context) {
    const MitsubaFixture fixture = loadMitsubaFixture("robust_v1");
    const SolveResult standard = solve(
        fixture.images,
        fixture.lights,
        NormalSolverMode::Standard,
        LightingModel::Directional,
        10.0,
        10.0,
        0.01,
        {},
        false,
        fixture.solveMask);
    const SolveResult robust = solve(
        fixture.images,
        fixture.lights,
        NormalSolverMode::Robust,
        LightingModel::Directional,
        10.0,
        10.0,
        0.01,
        fixture.saturationMasks,
        true,
        fixture.solveMask);

    const double standardError = meanAngularErrorDegrees(
        standard.normals,
        robust.validMask,
        fixture.expectedNormals);
    const double robustError = meanAngularErrorDegrees(
        robust.normals,
        robust.validMask,
        fixture.expectedNormals);
    cv::Mat affectedMask = unionAcrossLights(unionMasks(fixture.shadowTruth, fixture.highlightTruth));
    cv::bitwise_or(affectedMask, unionAcrossLights(fixture.saturationMasks), affectedMask);
    cv::bitwise_and(affectedMask, robust.validMask, affectedMask);
    const double affectedStandardError = meanAngularErrorDegrees(
        standard.normals,
        affectedMask,
        fixture.expectedNormals);
    const double affectedRobustError = meanAngularErrorDegrees(
        robust.normals,
        affectedMask,
        fixture.expectedNormals);
    const ClassificationMetrics shadowMetrics = classificationMetrics(
        robust.diagnostics.shadowObservationMasks,
        fixture.shadowTruth);
    const ClassificationMetrics highlightMetrics = classificationMetrics(
        unionMasks(
            robust.diagnostics.highlightObservationMasks,
            robust.diagnostics.saturationObservationMasks),
        unionMasks(fixture.highlightTruth, fixture.saturationMasks));
    const ClassificationMetrics saturationMetrics = classificationMetrics(
        robust.diagnostics.saturationObservationMasks,
        fixture.saturationMasks);
    int belowThresholdCount = 0;
    for (const cv::Mat& image : fixture.images) {
        for (int y = 0; y < image.rows; ++y) {
            const float* row = image.ptr<float>(y);
            for (int x = 0; x < image.cols; ++x) {
                belowThresholdCount += row[x] <= 0.02f ? 1 : 0;
            }
        }
    }

    const double shadowF1 = shadowMetrics.precision + shadowMetrics.recall > 0.0
        ? 2.0 * shadowMetrics.precision * shadowMetrics.recall /
            (shadowMetrics.precision + shadowMetrics.recall)
        : 0.0;
    const double highlightF1 = highlightMetrics.precision + highlightMetrics.recall > 0.0
        ? 2.0 * highlightMetrics.precision * highlightMetrics.recall /
            (highlightMetrics.precision + highlightMetrics.recall)
        : 0.0;
    cv::Mat objectRegion;
    cv::compare(fixture.shapeIndex, 1, objectRegion, cv::CMP_GT);
    cv::bitwise_and(objectRegion, fixture.solveMask, objectRegion);
    const ClassificationMetrics objectShadowMetrics = classificationMetrics(
        robust.diagnostics.shadowObservationMasks,
        fixture.shadowTruth,
        objectRegion);
    const double objectShadowF1 = objectShadowMetrics.precision + objectShadowMetrics.recall > 0.0
        ? 2.0 * objectShadowMetrics.precision * objectShadowMetrics.recall /
            (objectShadowMetrics.precision + objectShadowMetrics.recall)
        : 0.0;

    std::cout << "mitsuba_standard_mean_degrees=" << standardError << '\n';
    std::cout << "mitsuba_robust_mean_degrees=" << robustError << '\n';
    std::cout << "mitsuba_affected_standard_mean_degrees=" << affectedStandardError << '\n';
    std::cout << "mitsuba_affected_robust_mean_degrees=" << affectedRobustError << '\n';
    std::cout << "mitsuba_solved_fraction=" << robust.diagnostics.solvedFraction << '\n';
    std::cout << "mitsuba_shadow_precision=" << shadowMetrics.precision << '\n';
    std::cout << "mitsuba_shadow_recall=" << shadowMetrics.recall << '\n';
    std::cout << "mitsuba_shadow_f1=" << shadowF1 << '\n';
    std::cout << "mitsuba_shadow_truth_count=" << shadowMetrics.truthCount << '\n';
    std::cout << "mitsuba_shadow_predicted_count=" << shadowMetrics.predictedCount << '\n';
    std::cout << "mitsuba_object_shadow_precision=" << objectShadowMetrics.precision << '\n';
    std::cout << "mitsuba_object_shadow_recall=" << objectShadowMetrics.recall << '\n';
    std::cout << "mitsuba_object_shadow_f1=" << objectShadowF1 << '\n';
    std::cout << "mitsuba_highlight_precision=" << highlightMetrics.precision << '\n';
    std::cout << "mitsuba_highlight_recall=" << highlightMetrics.recall << '\n';
    std::cout << "mitsuba_highlight_f1=" << highlightF1 << '\n';
    std::cout << "mitsuba_highlight_truth_count=" << highlightMetrics.truthCount << '\n';
    std::cout << "mitsuba_highlight_predicted_count=" << highlightMetrics.predictedCount << '\n';
    std::cout << "mitsuba_saturation_recall=" << saturationMetrics.recall << '\n';
    std::cout << "mitsuba_saturation_truth_count=" << saturationMetrics.truthCount << '\n';
    std::cout << "mitsuba_below_threshold_count=" << belowThresholdCount << '\n';
    const std::vector<cv::Mat> predictedHighlights = unionMasks(
        robust.diagnostics.highlightObservationMasks,
        robust.diagnostics.saturationObservationMasks);
    const std::vector<cv::Mat> truthHighlights = unionMasks(
        fixture.highlightTruth,
        fixture.saturationMasks);
    std::vector<double> regionStandardErrors(5, 180.0);
    std::vector<double> regionRobustErrors(5, 180.0);
    for (int shape = 1; shape <= 4; ++shape) {
        cv::Mat region;
        cv::compare(fixture.shapeIndex, shape, region, cv::CMP_EQ);
        cv::bitwise_and(region, fixture.solveMask, region);
        const ClassificationMetrics regionShadow = classificationMetrics(
            robust.diagnostics.shadowObservationMasks,
            fixture.shadowTruth,
            region);
        const ClassificationMetrics regionHighlight = classificationMetrics(
            predictedHighlights,
            truthHighlights,
            region);
        cv::Mat solvedRegion;
        cv::bitwise_and(region, robust.validMask, solvedRegion);
        const double regionStandardError = meanAngularErrorDegrees(
            standard.normals,
            solvedRegion,
            fixture.expectedNormals);
        const double regionRobustError = meanAngularErrorDegrees(
            robust.normals,
            solvedRegion,
            fixture.expectedNormals);
        regionStandardErrors[static_cast<size_t>(shape)] = regionStandardError;
        regionRobustErrors[static_cast<size_t>(shape)] = regionRobustError;
        std::cout << "mitsuba_shape_" << shape << "_pixels=" << cv::countNonZero(region) << '\n';
        std::cout << "mitsuba_shape_" << shape << "_standard_mean_degrees=" << regionStandardError << '\n';
        std::cout << "mitsuba_shape_" << shape << "_robust_mean_degrees=" << regionRobustError << '\n';
        std::cout << "mitsuba_shape_" << shape << "_shadow_precision=" << regionShadow.precision << '\n';
        std::cout << "mitsuba_shape_" << shape << "_shadow_recall=" << regionShadow.recall << '\n';
        std::cout << "mitsuba_shape_" << shape << "_shadow_predicted=" << regionShadow.predictedCount << '\n';
        std::cout << "mitsuba_shape_" << shape << "_highlight_precision=" << regionHighlight.precision << '\n';
        std::cout << "mitsuba_shape_" << shape << "_highlight_recall=" << regionHighlight.recall << '\n';
        std::cout << "mitsuba_shape_" << shape << "_highlight_predicted=" << regionHighlight.predictedCount << '\n';
    }

    context.check(
        robust.diagnostics.solvedFraction > 0.98,
        "Mitsuba mixed-material solve coverage was " + std::to_string(robust.diagnostics.solvedFraction));
    context.check(
        robustError < 2.0,
        "Mitsuba mixed-material robust mean angular error was " + std::to_string(robustError));
    context.check(
        robustError < standardError * 0.50 && affectedRobustError < affectedStandardError * 0.50,
        "Mitsuba robust solve must improve both overall and corrupted-region angular error");
    context.check(
        regionRobustErrors[3] < 20.0 && regionRobustErrors[3] < regionStandardErrors[3] * 0.50,
        "Mitsuba black glossy-sphere normal error was " + std::to_string(regionRobustErrors[3]));
    context.check(
        regionRobustErrors[4] < 6.0 && regionRobustErrors[4] < regionStandardErrors[4] * 0.65,
        "Mitsuba rough glossy-sphere normal error was " + std::to_string(regionRobustErrors[4]));
    context.check(
        shadowMetrics.precision > 0.90 && shadowMetrics.recall > 0.90 && shadowF1 > 0.92,
        "Mitsuba shadow classification precision/recall/F1 was " +
            std::to_string(shadowMetrics.precision) + "/" + std::to_string(shadowMetrics.recall) +
            "/" + std::to_string(shadowF1));
    context.check(
        objectShadowMetrics.precision > 0.70 && objectShadowMetrics.recall > 0.90 &&
            objectShadowF1 > 0.80,
        "Mitsuba object-only shadow classification precision/recall/F1 was " +
            std::to_string(objectShadowMetrics.precision) + "/" +
            std::to_string(objectShadowMetrics.recall) + "/" +
            std::to_string(objectShadowF1));
    context.check(
        highlightMetrics.precision > 0.90 && highlightMetrics.recall > 0.80 && highlightF1 > 0.85,
        "Mitsuba highlight classification precision/recall/F1 was " +
            std::to_string(highlightMetrics.precision) + "/" + std::to_string(highlightMetrics.recall) +
            "/" + std::to_string(highlightF1));
    context.check(
        saturationMetrics.recall > 0.999,
        "definite sensor clipping must be preserved in the robust diagnostics");
}

struct MitsubaAcceptance {
    double minimumCoverage = 0.0;
    double maximumRobustError = 180.0;
    double maximumOverallErrorRatio = 1.0;
    double maximumAffectedErrorRatio = 1.0;
    double minimumShadowF1 = 0.0;
    double minimumObjectShadowF1 = 0.0;
    double minimumHighlightF1 = 0.0;
};

double f1Score(const ClassificationMetrics& metrics) {
    return metrics.precision + metrics.recall > 0.0
        ? 2.0 * metrics.precision * metrics.recall /
            (metrics.precision + metrics.recall)
        : 0.0;
}

void testAdditionalMitsubaFixture(
    TestContext& context,
    const MitsubaFixture& fixture,
    const MitsubaAcceptance& acceptance) {
    const SolveResult standard = solve(
        fixture.images,
        fixture.lights,
        NormalSolverMode::Standard,
        fixture.lightingModel,
        fixture.ringRadiusMm,
        fixture.ringHeightMm,
        fixture.pixelScaleMm,
        {},
        false,
        fixture.solveMask);
    const SolveResult robust = solve(
        fixture.images,
        fixture.lights,
        NormalSolverMode::Robust,
        fixture.lightingModel,
        fixture.ringRadiusMm,
        fixture.ringHeightMm,
        fixture.pixelScaleMm,
        fixture.saturationMasks,
        true,
        fixture.solveMask);

    const double standardError = meanAngularErrorDegrees(
        standard.normals,
        robust.validMask,
        fixture.expectedNormals);
    const double robustError = meanAngularErrorDegrees(
        robust.normals,
        robust.validMask,
        fixture.expectedNormals);
    cv::Mat affectedMask = unionAcrossLights(unionMasks(fixture.shadowTruth, fixture.highlightTruth));
    cv::bitwise_or(affectedMask, unionAcrossLights(fixture.saturationMasks), affectedMask);
    cv::bitwise_and(affectedMask, robust.validMask, affectedMask);
    const double affectedStandardError = meanAngularErrorDegrees(
        standard.normals,
        affectedMask,
        fixture.expectedNormals);
    const double affectedRobustError = meanAngularErrorDegrees(
        robust.normals,
        affectedMask,
        fixture.expectedNormals);
    const ClassificationMetrics shadowMetrics = classificationMetrics(
        robust.diagnostics.shadowObservationMasks,
        fixture.shadowTruth);
    const ClassificationMetrics highlightMetrics = classificationMetrics(
        unionMasks(
            robust.diagnostics.highlightObservationMasks,
            robust.diagnostics.saturationObservationMasks),
        unionMasks(fixture.highlightTruth, fixture.saturationMasks));
    const ClassificationMetrics saturationMetrics = classificationMetrics(
        robust.diagnostics.saturationObservationMasks,
        fixture.saturationMasks);
    cv::Mat objectRegion;
    cv::compare(fixture.shapeIndex, 1, objectRegion, cv::CMP_GT);
    cv::bitwise_and(objectRegion, fixture.solveMask, objectRegion);
    const ClassificationMetrics objectShadowMetrics = classificationMetrics(
        robust.diagnostics.shadowObservationMasks,
        fixture.shadowTruth,
        objectRegion);

    const double shadowF1 = f1Score(shadowMetrics);
    const double objectShadowF1 = f1Score(objectShadowMetrics);
    const double highlightF1 = f1Score(highlightMetrics);
    const std::string prefix = "mitsuba_" + fixture.name;
    std::cout << prefix << "_standard_mean_degrees=" << standardError << '\n';
    std::cout << prefix << "_robust_mean_degrees=" << robustError << '\n';
    std::cout << prefix << "_affected_standard_mean_degrees=" << affectedStandardError << '\n';
    std::cout << prefix << "_affected_robust_mean_degrees=" << affectedRobustError << '\n';
    std::cout << prefix << "_solved_fraction=" << robust.diagnostics.solvedFraction << '\n';
    std::cout << prefix << "_shadow_precision=" << shadowMetrics.precision << '\n';
    std::cout << prefix << "_shadow_recall=" << shadowMetrics.recall << '\n';
    std::cout << prefix << "_shadow_f1=" << shadowF1 << '\n';
    std::cout << prefix << "_object_shadow_precision=" << objectShadowMetrics.precision << '\n';
    std::cout << prefix << "_object_shadow_recall=" << objectShadowMetrics.recall << '\n';
    std::cout << prefix << "_object_shadow_f1=" << objectShadowF1 << '\n';
    std::cout << prefix << "_highlight_precision=" << highlightMetrics.precision << '\n';
    std::cout << prefix << "_highlight_recall=" << highlightMetrics.recall << '\n';
    std::cout << prefix << "_highlight_f1=" << highlightF1 << '\n';
    std::cout << prefix << "_saturation_recall=" << saturationMetrics.recall << '\n';

    double maximumShapeValue = 0.0;
    cv::minMaxLoc(fixture.shapeIndex, nullptr, &maximumShapeValue);
    for (int shape = 1; shape <= static_cast<int>(maximumShapeValue); ++shape) {
        cv::Mat region;
        cv::compare(fixture.shapeIndex, shape, region, cv::CMP_EQ);
        cv::bitwise_and(region, fixture.solveMask, region);
        if (cv::countNonZero(region) == 0) {
            continue;
        }
        cv::Mat solvedRegion;
        cv::bitwise_and(region, robust.validMask, solvedRegion);
        const ClassificationMetrics regionShadow = classificationMetrics(
            robust.diagnostics.shadowObservationMasks,
            fixture.shadowTruth,
            region);
        const ClassificationMetrics regionHighlight = classificationMetrics(
            unionMasks(
                robust.diagnostics.highlightObservationMasks,
                robust.diagnostics.saturationObservationMasks),
            unionMasks(fixture.highlightTruth, fixture.saturationMasks),
            region);
        std::cout << prefix << "_shape_" << shape << "_pixels=" << cv::countNonZero(region) << '\n';
        std::cout << prefix << "_shape_" << shape << "_robust_mean_degrees="
                  << meanAngularErrorDegrees(robust.normals, solvedRegion, fixture.expectedNormals) << '\n';
        std::cout << prefix << "_shape_" << shape << "_shadow_f1=" << f1Score(regionShadow) << '\n';
        std::cout << prefix << "_shape_" << shape << "_highlight_f1=" << f1Score(regionHighlight) << '\n';
    }

    context.check(
        robust.diagnostics.solvedFraction > acceptance.minimumCoverage,
        fixture.name + " solve coverage was " + std::to_string(robust.diagnostics.solvedFraction));
    context.check(
        robustError < acceptance.maximumRobustError,
        fixture.name + " robust mean angular error was " + std::to_string(robustError));
    context.check(
        robustError < standardError * acceptance.maximumOverallErrorRatio,
        fixture.name + " robust solve did not sufficiently improve overall angular error");
    context.check(
        affectedRobustError < affectedStandardError * acceptance.maximumAffectedErrorRatio,
        fixture.name + " robust solve did not sufficiently improve corrupted-region angular error");
    context.check(
        shadowF1 > acceptance.minimumShadowF1,
        fixture.name + " shadow F1 was " + std::to_string(shadowF1));
    context.check(
        objectShadowF1 > acceptance.minimumObjectShadowF1,
        fixture.name + " object-region shadow F1 was " + std::to_string(objectShadowF1));
    context.check(
        highlightF1 > acceptance.minimumHighlightF1,
        fixture.name + " highlight-or-clipping F1 was " + std::to_string(highlightF1));
    context.check(
        saturationMetrics.recall > 0.999,
        fixture.name + " definite-clipping recall was " + std::to_string(saturationMetrics.recall));
}

void testAdditionalMitsubaFixtures(TestContext& context) {
    testAdditionalMitsubaFixture(
        context,
        loadMitsubaFixture("textured_primitives_v1"),
        MitsubaAcceptance{0.94, 6.0, 0.85, 0.90, 0.78, 0.65, 0.68});
    testAdditionalMitsubaFixture(
        context,
        loadMitsubaFixture(
            "holdout_relief_v1",
            LightingModel::NearFieldRing,
            3.6,
            1.25,
            2.0 / 72.0),
        MitsubaAcceptance{0.90, 8.0, 1.0, 1.0, 0.75, 0.65, 0.70});
}

cv::Mat normalsFromHeight(const cv::Mat& height) {
    cv::Mat normals(height.size(), CV_32FC3);
    for (int y = 0; y < height.rows; ++y) {
        cv::Vec3f* normalRow = normals.ptr<cv::Vec3f>(y);
        for (int x = 0; x < height.cols; ++x) {
            const int left = std::max(0, x - 1);
            const int right = std::min(height.cols - 1, x + 1);
            const int up = std::max(0, y - 1);
            const int down = std::min(height.rows - 1, y + 1);
            const float p = (height.at<float>(y, right) - height.at<float>(y, left)) /
                static_cast<float>(right - left);
            const float q = (height.at<float>(down, x) - height.at<float>(up, x)) /
                static_cast<float>(down - up);
            normalRow[x] = normalized(cv::Vec3f(-p, q, 1.0f));
        }
    }
    return normals;
}

float bilinearHeight(const cv::Mat& height, double x, double y) {
    const int x0 = std::clamp(static_cast<int>(std::floor(x)), 0, height.cols - 1);
    const int y0 = std::clamp(static_cast<int>(std::floor(y)), 0, height.rows - 1);
    const int x1 = std::min(x0 + 1, height.cols - 1);
    const int y1 = std::min(y0 + 1, height.rows - 1);
    const double fx = std::clamp(x - x0, 0.0, 1.0);
    const double fy = std::clamp(y - y0, 0.0, 1.0);
    return static_cast<float>(
        (1.0 - fy) * ((1.0 - fx) * height.at<float>(y0, x0) + fx * height.at<float>(y0, x1)) +
        fy * ((1.0 - fx) * height.at<float>(y1, x0) + fx * height.at<float>(y1, x1)));
}

struct NearFieldShadowFixture {
    cv::Mat trueHeight;
    cv::Mat initialHeight;
    cv::Mat normals;
    cv::Mat mask;
    std::vector<cv::Vec3f> lights;
    std::vector<cv::Mat> images;
    std::vector<cv::Mat> rawShadowMasks;
    std::vector<cv::Mat> castShadowFractions;
};

cv::Vec3d normalizedDouble(const cv::Vec3d& value) {
    return value / std::sqrt(value.dot(value));
}

std::vector<cv::Vec3d> finiteDiskSamplesForFixture(
    const cv::Vec3d& center,
    double referenceSurfaceZMm,
    double diameterMm) {
    std::vector<cv::Vec3d> samples{center};
    if (diameterMm <= 0.0) {
        return samples;
    }
    const cv::Vec3d opticalAxis = normalizedDouble(
        cv::Vec3d(0.0, 0.0, referenceSurfaceZMm) - center);
    cv::Vec3d axisU = opticalAxis.cross(cv::Vec3d(0.0, 0.0, 1.0));
    if (axisU.dot(axisU) <= 1.0e-12) {
        axisU = opticalAxis.cross(cv::Vec3d(1.0, 0.0, 0.0));
    }
    axisU = normalizedDouble(axisU);
    const cv::Vec3d axisV = normalizedDouble(opticalAxis.cross(axisU));
    const double radius = 0.5 * diameterMm;
    for (int ring = 0; ring < 2; ++ring) {
        const int count = ring == 0 ? 6 : 12;
        const double sampleRadius = radius * (ring == 0 ? 0.42 : 0.82);
        for (int i = 0; i < count; ++i) {
            const double angle = 2.0 * CV_PI * (static_cast<double>(i) + 0.25 * ring) /
                static_cast<double>(count);
            samples.push_back(center + sampleRadius * (
                std::cos(angle) * axisU + std::sin(angle) * axisV));
        }
    }
    return samples;
}

NearFieldShadowFixture makeNearFieldShadowFixture(
    double ringRadiusMm,
    double ringHeightMm,
    double pixelScaleMm,
    double referenceSurfaceZMm = 0.0,
    double ledDiameterMm = 0.0) {
    constexpr int rows = 72;
    constexpr int cols = 96;
    NearFieldShadowFixture fixture;
    fixture.trueHeight = cv::Mat(rows, cols, CV_32F);
    fixture.initialHeight = cv::Mat(rows, cols, CV_32F);
    fixture.mask = cv::Mat(rows, cols, CV_8U, cv::Scalar(255));
    for (int y = 0; y < rows; ++y) {
        float* trueRow = fixture.trueHeight.ptr<float>(y);
        float* initialRow = fixture.initialHeight.ptr<float>(y);
        const double yn = (static_cast<double>(y) - 0.5 * (rows - 1)) / (0.5 * rows);
        for (int x = 0; x < cols; ++x) {
            const double xn = (static_cast<double>(x) - 0.5 * (cols - 1)) / (0.5 * cols);
            const double mound = 8.5 * std::exp(-2.8 * (xn * xn + 1.3 * yn * yn));
            const double ridge = 5.2 * std::exp(-180.0 * (xn + 0.18) * (xn + 0.18)) *
                std::exp(-1.8 * (yn - 0.05) * (yn - 0.05));
            const double shoulder = 3.0 * std::exp(
                -18.0 * ((xn - 0.38) * (xn - 0.38) + (yn + 0.23) * (yn + 0.23)));
            const double fine = 0.32 * std::sin(13.0 * xn + 2.0 * yn) *
                std::cos(11.0 * yn - xn);
            const double truth = mound + ridge + shoulder + fine;
            const double broadDrift = 4.2 * (
                0.52 * xn * xn - 0.34 * yn * yn + 0.30 * xn * yn +
                0.22 * xn - 0.15 * yn +
                0.07 * std::sin(0.8 * CV_PI * xn) * std::cos(0.55 * CV_PI * yn));
            trueRow[x] = static_cast<float>(truth);
            initialRow[x] = static_cast<float>(truth + broadDrift);
        }
    }
    fixture.normals = normalsFromHeight(fixture.trueHeight);
    fixture.lights = makeRingLights(8, static_cast<float>(
        ringHeightMm / std::sqrt(ringRadiusMm * ringRadiusMm + ringHeightMm * ringHeightMm)));
    const cv::Point2d center(0.5 * (cols - 1), 0.5 * (rows - 1));
    const double referenceDistanceSquared = ringRadiusMm * ringRadiusMm + ringHeightMm * ringHeightMm;

    for (size_t lightIndex = 0; lightIndex < fixture.lights.size(); ++lightIndex) {
        const cv::Vec3f reference = fixture.lights[lightIndex];
        const double radial = std::hypot(reference[0], reference[1]);
        const cv::Vec3d source(
            ringRadiusMm * reference[0] / radial,
            ringRadiusMm * reference[1] / radial,
            ringHeightMm);
        const std::vector<cv::Vec3d> sourceSamples = finiteDiskSamplesForFixture(
            source, referenceSurfaceZMm, ledDiameterMm);
        cv::Mat image(rows, cols, CV_32F, cv::Scalar(0));
        cv::Mat shadow(rows, cols, CV_8U, cv::Scalar(0));
        cv::Mat shadowFraction(rows, cols, CV_32F, cv::Scalar(0));
        for (int y = 0; y < rows; ++y) {
            float* imageRow = image.ptr<float>(y);
            uchar* shadowRow = shadow.ptr<uchar>(y);
            float* shadowFractionRow = shadowFraction.ptr<float>(y);
            const cv::Vec3f* normalRow = fixture.normals.ptr<cv::Vec3f>(y);
            for (int x = 0; x < cols; ++x) {
                const double receiverHeightMm = referenceSurfaceZMm +
                    fixture.trueHeight.at<float>(y, x) * pixelScaleMm;
                const cv::Vec3d receiver(
                    (static_cast<double>(x) - center.x) * pixelScaleMm,
                    (center.y - static_cast<double>(y)) * pixelScaleMm,
                    receiverHeightMm);
                const cv::Vec3d displacement = source - receiver;
                const double distanceSquared = displacement.dot(displacement);
                const cv::Vec3d local = displacement / std::sqrt(distanceSquared);
                const double nDotL = normalRow[x].dot(cv::Vec3f(
                    static_cast<float>(local[0]),
                    static_cast<float>(local[1]),
                    static_cast<float>(local[2])));
                const bool attached = nDotL <= 0.04;
                int blockedSamples = 0;
                for (const cv::Vec3d& sampleSource : sourceSamples) {
                    const double sourceImageX = center.x + sampleSource[0] / pixelScaleMm;
                    const double sourceImageY = center.y - sampleSource[1] / pixelScaleMm;
                    const double dx = sourceImageX - x;
                    const double dy = sourceImageY - y;
                    const int steps = static_cast<int>(std::ceil(
                        std::max(std::abs(dx), std::abs(dy))));
                    bool sampleBlocked = false;
                    for (int step = 2; !attached && step < steps; ++step) {
                        const double t = static_cast<double>(step) / static_cast<double>(steps);
                        const double sx = x + t * dx;
                        const double sy = y + t * dy;
                        if (sx < 0.0 || sx > cols - 1.0 || sy < 0.0 || sy > rows - 1.0) {
                            break;
                        }
                        const double terrainMm = referenceSurfaceZMm +
                            bilinearHeight(fixture.trueHeight, sx, sy) * pixelScaleMm;
                        const double rayMm = receiverHeightMm +
                            t * (sampleSource[2] - receiverHeightMm);
                        if (terrainMm > rayMm + 0.006) {
                            sampleBlocked = true;
                            break;
                        }
                    }
                    blockedSamples += sampleBlocked ? 1 : 0;
                }
                const double blockedFraction = static_cast<double>(blockedSamples) /
                    static_cast<double>(sourceSamples.size());
                const bool cast = blockedFraction >= 0.5;
                shadowFractionRow[x] = static_cast<float>(blockedFraction);

                const double albedo = std::clamp(
                    0.44 + 0.12 * std::sin(0.17 * x + 0.11 * y) +
                        0.06 * std::cos(0.07 * x - 0.19 * y),
                    0.18,
                    0.68);
                const double irradiance = referenceDistanceSquared / distanceSquared;
                double value = albedo * irradiance * std::max(0.0, nDotL);
                if (attached) {
                    value *= 0.02;
                } else if (blockedFraction > 0.0) {
                    value *= std::max(0.025, 1.0 - 0.96 * blockedFraction);
                }
                value += 0.0015 * std::sin(0.31 * x + 0.23 * y + 0.7 * lightIndex);
                imageRow[x] = static_cast<float>(std::clamp(value, 0.0, 1.0));

                bool reportedShadow = attached || cast;
                if (reportedShadow && ((x + 3 * y + 7 * static_cast<int>(lightIndex)) % 37 == 0)) {
                    reportedShadow = false;
                }
                if (!reportedShadow && !attached &&
                    ((11 * x + 17 * y + 19 * static_cast<int>(lightIndex)) % 521 == 0)) {
                    reportedShadow = true;
                }
                shadowRow[x] = reportedShadow ? 255 : 0;
            }
        }
        fixture.images.push_back(std::move(image));
        fixture.rawShadowMasks.push_back(std::move(shadow));
        fixture.castShadowFractions.push_back(std::move(shadowFraction));
    }
    return fixture;
}

NearFieldShadowFixture makeDirectionalShadowFixture() {
    NearFieldShadowFixture fixture = makeNearFieldShadowFixture(4.2, 1.15, 0.035);
    fixture.lights = makeRingLights(8, 0.36f);
    fixture.images.clear();
    fixture.rawShadowMasks.clear();
    fixture.castShadowFractions.clear();

    for (size_t lightIndex = 0; lightIndex < fixture.lights.size(); ++lightIndex) {
        const cv::Vec3f light = normalized(fixture.lights[lightIndex]);
        const double radial = std::hypot(light[0], light[1]);
        const double rayX = light[0] / radial;
        const double rayY = -light[1] / radial;
        const double risePerPixel = light[2] / radial;
        cv::Mat image(fixture.mask.size(), CV_32F, cv::Scalar(0));
        cv::Mat shadow(fixture.mask.size(), CV_8U, cv::Scalar(0));
        cv::Mat shadowFraction(fixture.mask.size(), CV_32F, cv::Scalar(0));

        for (int y = 0; y < fixture.mask.rows; ++y) {
            float* imageRow = image.ptr<float>(y);
            uchar* shadowRow = shadow.ptr<uchar>(y);
            float* fractionRow = shadowFraction.ptr<float>(y);
            const cv::Vec3f* normalRow = fixture.normals.ptr<cv::Vec3f>(y);
            for (int x = 0; x < fixture.mask.cols; ++x) {
                const double nDotL = normalRow[x].dot(light);
                const bool attached = nDotL <= 0.04;
                bool cast = false;
                if (!attached) {
                    const double receiver = fixture.trueHeight.at<float>(y, x);
                    for (double distance = 1.5;; distance += 0.5) {
                        const double sampleX = x + distance * rayX;
                        const double sampleY = y + distance * rayY;
                        if (sampleX < 0.0 || sampleX > fixture.mask.cols - 1.0 ||
                            sampleY < 0.0 || sampleY > fixture.mask.rows - 1.0) {
                            break;
                        }
                        const double rayHeight = receiver + distance * risePerPixel;
                        if (bilinearHeight(fixture.trueHeight, sampleX, sampleY) >
                            rayHeight + 0.06) {
                            cast = true;
                            break;
                        }
                    }
                }
                fractionRow[x] = cast ? 1.0f : 0.0f;

                const double albedo = std::clamp(
                    0.44 + 0.12 * std::sin(0.17 * x + 0.11 * y) +
                        0.06 * std::cos(0.07 * x - 0.19 * y),
                    0.18,
                    0.68);
                double value = albedo * std::max(0.0, nDotL);
                if (attached) {
                    value *= 0.02;
                } else if (cast) {
                    value *= 0.025;
                }
                value += 0.0015 * std::sin(0.31 * x + 0.23 * y + 0.7 * lightIndex);
                imageRow[x] = static_cast<float>(std::clamp(value, 0.0, 1.0));

                bool reportedShadow = attached || cast;
                if (reportedShadow && ((x + 3 * y + 7 * static_cast<int>(lightIndex)) % 37 == 0)) {
                    reportedShadow = false;
                }
                if (!reportedShadow && !attached &&
                    ((11 * x + 17 * y + 19 * static_cast<int>(lightIndex)) % 521 == 0)) {
                    reportedShadow = true;
                }
                shadowRow[x] = reportedShadow ? 255 : 0;
            }
        }
        fixture.images.push_back(std::move(image));
        fixture.rawShadowMasks.push_back(std::move(shadow));
        fixture.castShadowFractions.push_back(std::move(shadowFraction));
    }
    return fixture;
}

double highFrequencyDifferenceRms(
    const cv::Mat& first,
    const cv::Mat& second,
    const cv::Mat& mask) {
    cv::Mat firstLow;
    cv::Mat secondLow;
    cv::GaussianBlur(first, firstLow, cv::Size(), 2.2);
    cv::GaussianBlur(second, secondLow, cv::Size(), 2.2);
    const cv::Mat difference = (first - firstLow) - (second - secondLow);
    return std::sqrt(
        cv::sum(difference.mul(difference))[0] /
        static_cast<double>(std::max(1, cv::countNonZero(mask))));
}

void testNearFieldShadowHeightRefinement(TestContext& context) {
    constexpr double ringRadiusMm = 4.2;
    constexpr double ringHeightMm = 1.15;
    constexpr double pixelScaleMm = 0.035;
    const NearFieldShadowFixture fixture = makeNearFieldShadowFixture(
        ringRadiusMm,
        ringHeightMm,
        pixelScaleMm);
    PhotometricDiagnostics diagnostics;
    diagnostics.shadowObservationMasks = fixture.rawShadowMasks;
    diagnostics.highlightObservationMasks.assign(
        fixture.lights.size(),
        cv::Mat(fixture.mask.size(), CV_8U, cv::Scalar(0)));
    diagnostics.saturationObservationMasks.assign(
        fixture.lights.size(),
        cv::Mat(fixture.mask.size(), CV_8U, cv::Scalar(0)));
    ShadowRefinementSettings settings;
    settings.lightingModel = LightingModel::NearFieldRing;
    settings.ringLightRadiusMm = ringRadiusMm;
    settings.ringLightHeightMm = ringHeightMm;
    settings.pixelScaleMm = pixelScaleMm;
    settings.lightingCenter = cv::Point2d(
        0.5 * (fixture.mask.cols - 1),
        0.5 * (fixture.mask.rows - 1));
    settings.maximumCoarseSide = 96;
    settings.iterations = 5;

    cv::Mat refined = fixture.initialHeight.clone();
    refineHeightFromCastShadows(
        refined,
        fixture.mask,
        fixture.mask,
        fixture.normals,
        fixture.images,
        fixture.lights,
        settings,
        diagnostics,
        [](const std::string& message) {
            std::cout << "shadow_height_progress=" << message << '\n';
        });
    const double beforeRmse = normalizedHeightRmse(
        fixture.initialHeight, fixture.trueHeight, fixture.mask);
    const double afterRmse = normalizedHeightRmse(refined, fixture.trueHeight, fixture.mask);
    const double detailChange = highFrequencyDifferenceRms(
        refined, fixture.initialHeight, fixture.mask);
    std::cout << "shadow_height_applied=" << diagnostics.shadowHeightRefinementApplied << '\n';
    std::cout << "shadow_height_mismatch_before=" << diagnostics.shadowMismatchRateBefore << '\n';
    std::cout << "shadow_height_mismatch_after=" << diagnostics.shadowMismatchRateAfter << '\n';
    std::cout << "shadow_height_holdout_before=" << diagnostics.shadowHoldoutMismatchRateBefore << '\n';
    std::cout << "shadow_height_holdout_after=" << diagnostics.shadowHoldoutMismatchRateAfter << '\n';
    std::cout << "shadow_height_rmse_before=" << beforeRmse << '\n';
    std::cout << "shadow_height_rmse_after=" << afterRmse << '\n';
    std::cout << "shadow_height_detail_change_rms=" << detailChange << '\n';
    context.check(
        diagnostics.shadowHeightRefinementApplied,
        "near-field shadow height correction must pass its internal fit/holdout gate");
    context.check(
        diagnostics.shadowMismatchRateAfter < diagnostics.shadowMismatchRateBefore * 0.96,
        "shadow refinement must reduce balanced cast-shadow mismatch by at least 4 percent");
    context.check(
        diagnostics.shadowHoldoutMismatchRateAfter <= diagnostics.shadowHoldoutMismatchRateBefore,
        "shadow refinement must not worsen withheld light directions");
    context.check(
        afterRmse < beforeRmse * 0.97,
        "shadow refinement must improve offset-normalized broad height RMSE");
    context.check(
        detailChange < 0.055,
        "shadow refinement changed too much high-frequency photometric relief");

    PhotometricDiagnostics scaledDiagnostics;
    scaledDiagnostics.shadowObservationMasks = fixture.rawShadowMasks;
    scaledDiagnostics.highlightObservationMasks = diagnostics.highlightObservationMasks;
    scaledDiagnostics.saturationObservationMasks = diagnostics.saturationObservationMasks;
    ShadowRefinementSettings scaledSettings = settings;
    scaledSettings.ringLightRadiusMm *= 3.0;
    scaledSettings.ringLightHeightMm *= 3.0;
    scaledSettings.pixelScaleMm *= 3.0;
    cv::Mat scaledResult = fixture.initialHeight.clone();
    refineHeightFromCastShadows(
        scaledResult,
        fixture.mask,
        fixture.mask,
        fixture.normals,
        fixture.images,
        fixture.lights,
        scaledSettings,
        scaledDiagnostics);
    context.check(
        cv::norm(refined, scaledResult, cv::NORM_INF) < 1.0e-5,
        "near-field shadow refinement must be invariant to a common physical-unit scale factor");

    PhotometricDiagnostics translatedDiagnostics;
    translatedDiagnostics.shadowObservationMasks = fixture.rawShadowMasks;
    translatedDiagnostics.highlightObservationMasks = diagnostics.highlightObservationMasks;
    translatedDiagnostics.saturationObservationMasks = diagnostics.saturationObservationMasks;
    cv::Mat translatedResult = fixture.initialHeight + 17.0f;
    refineHeightFromCastShadows(
        translatedResult,
        fixture.mask,
        fixture.mask,
        fixture.normals,
        fixture.images,
        fixture.lights,
        settings,
        translatedDiagnostics);
    translatedResult -= 17.0f;
    context.check(
        cv::norm(refined, translatedResult, cv::NORM_INF) < 1.0e-4,
        "shadow refinement must be invariant to the arbitrary integrated-height offset");
}

void testDirectionalShadowHeightRefinement(TestContext& context) {
    const NearFieldShadowFixture fixture = makeDirectionalShadowFixture();
    auto makeDiagnostics = [&]() {
        PhotometricDiagnostics diagnostics;
        diagnostics.shadowObservationMasks = fixture.rawShadowMasks;
        diagnostics.highlightObservationMasks.assign(
            fixture.lights.size(),
            cv::Mat(fixture.mask.size(), CV_8U, cv::Scalar(0)));
        diagnostics.saturationObservationMasks.assign(
            fixture.lights.size(),
            cv::Mat(fixture.mask.size(), CV_8U, cv::Scalar(0)));
        return diagnostics;
    };

    ShadowRefinementSettings settings;
    settings.lightingModel = LightingModel::Directional;
    settings.maximumCoarseSide = 96;
    settings.iterations = 5;
    settings.pixelScaleMm = 0.0123;
    settings.ringLightRadiusMm = 123.0;
    settings.ringLightHeightMm = 456.0;
    settings.referenceSurfaceZMm = 78.0;
    settings.ledDiameterMm = 9.0;

    PhotometricDiagnostics diagnostics = makeDiagnostics();
    cv::Mat refined = fixture.initialHeight.clone();
    refineHeightFromCastShadows(
        refined,
        fixture.mask,
        fixture.mask,
        fixture.normals,
        fixture.images,
        fixture.lights,
        settings,
        diagnostics);
    const double beforeRmse = normalizedHeightRmse(
        fixture.initialHeight, fixture.trueHeight, fixture.mask);
    const double afterRmse = normalizedHeightRmse(refined, fixture.trueHeight, fixture.mask);
    std::cout << "directional_shadow_height_applied="
              << diagnostics.shadowHeightRefinementApplied << '\n';
    std::cout << "directional_shadow_height_mismatch_before="
              << diagnostics.shadowMismatchRateBefore << '\n';
    std::cout << "directional_shadow_height_mismatch_after="
              << diagnostics.shadowMismatchRateAfter << '\n';
    std::cout << "directional_shadow_height_rmse_before=" << beforeRmse << '\n';
    std::cout << "directional_shadow_height_rmse_after=" << afterRmse << '\n';
    context.check(
        diagnostics.shadowHeightRefinementApplied,
        "sphere-calibrated directional lights must support guarded shadow height refinement");
    context.check(
        diagnostics.shadowMismatchRateAfter < diagnostics.shadowMismatchRateBefore,
        "directional shadow refinement must reduce balanced cast-shadow mismatch");
    context.check(
        afterRmse < beforeRmse,
        "directional shadow refinement must improve offset-normalized broad height RMSE");

    ShadowRefinementSettings alternateRingOnlySettings = settings;
    alternateRingOnlySettings.pixelScaleMm = 3.7;
    alternateRingOnlySettings.ringLightRadiusMm = 2.0;
    alternateRingOnlySettings.ringLightHeightMm = 3.0;
    alternateRingOnlySettings.referenceSurfaceZMm = 1.5;
    alternateRingOnlySettings.ledDiameterMm = 0.0;
    PhotometricDiagnostics alternateDiagnostics = makeDiagnostics();
    cv::Mat alternate = fixture.initialHeight.clone();
    refineHeightFromCastShadows(
        alternate,
        fixture.mask,
        fixture.mask,
        fixture.normals,
        fixture.images,
        fixture.lights,
        alternateRingOnlySettings,
        alternateDiagnostics);
    context.check(
        cv::norm(refined, alternate, cv::NORM_INF) < 1.0e-6 &&
            diagnostics.shadowHeightRefinementDecision ==
                alternateDiagnostics.shadowHeightRefinementDecision,
        "directional shadow refinement must ignore ring-only geometry parameters");
}

void testFiniteLedAndReferenceSurfaceGeometry(TestContext& context) {
    constexpr double ringRadiusMm = 4.2;
    constexpr double ringHeightMm = 1.65;
    constexpr double pixelScaleMm = 0.035;
    constexpr double referenceSurfaceZMm = 0.48;
    constexpr double ledDiameterMm = 0.62;
    const NearFieldShadowFixture fixture = makeNearFieldShadowFixture(
        ringRadiusMm,
        ringHeightMm,
        pixelScaleMm,
        referenceSurfaceZMm,
        ledDiameterMm);

    auto makeDiagnostics = [&]() {
        PhotometricDiagnostics diagnostics;
        diagnostics.shadowObservationMasks = fixture.rawShadowMasks;
        diagnostics.highlightObservationMasks.assign(
            fixture.lights.size(),
            cv::Mat(fixture.mask.size(), CV_8U, cv::Scalar(0)));
        diagnostics.saturationObservationMasks.assign(
            fixture.lights.size(),
            cv::Mat(fixture.mask.size(), CV_8U, cv::Scalar(0)));
        return diagnostics;
    };
    ShadowRefinementSettings correctSettings;
    correctSettings.lightingModel = LightingModel::NearFieldRing;
    correctSettings.ringLightRadiusMm = ringRadiusMm;
    correctSettings.ringLightHeightMm = ringHeightMm;
    correctSettings.pixelScaleMm = pixelScaleMm;
    correctSettings.referenceSurfaceZMm = referenceSurfaceZMm;
    correctSettings.ledDiameterMm = ledDiameterMm;
    correctSettings.lightingCenter = cv::Point2d(
        0.5 * (fixture.mask.cols - 1),
        0.5 * (fixture.mask.rows - 1));
    correctSettings.maximumCoarseSide = 96;
    correctSettings.iterations = 1;

    PhotometricDiagnostics correctDiagnostics = makeDiagnostics();
    cv::Mat correctHeight = fixture.trueHeight.clone();
    refineHeightFromCastShadows(
        correctHeight,
        fixture.mask,
        fixture.mask,
        fixture.normals,
        fixture.images,
        fixture.lights,
        correctSettings,
        correctDiagnostics);

    PhotometricDiagnostics pointDiagnostics = makeDiagnostics();
    ShadowRefinementSettings pointSettings = correctSettings;
    pointSettings.ledDiameterMm = 0.0;
    cv::Mat pointHeight = fixture.trueHeight.clone();
    refineHeightFromCastShadows(
        pointHeight,
        fixture.mask,
        fixture.mask,
        fixture.normals,
        fixture.images,
        fixture.lights,
        pointSettings,
        pointDiagnostics);

    PhotometricDiagnostics wrongDatumDiagnostics = makeDiagnostics();
    ShadowRefinementSettings wrongDatumSettings = correctSettings;
    wrongDatumSettings.referenceSurfaceZMm = 0.0;
    cv::Mat wrongDatumHeight = fixture.trueHeight.clone();
    refineHeightFromCastShadows(
        wrongDatumHeight,
        fixture.mask,
        fixture.mask,
        fixture.normals,
        fixture.images,
        fixture.lights,
        wrongDatumSettings,
        wrongDatumDiagnostics);

    double observabilityMaximum = 0.0;
    cv::minMaxLoc(correctDiagnostics.shadowObservability, nullptr, &observabilityMaximum);
    double edgeSupportMaximum = 0.0;
    cv::minMaxLoc(correctDiagnostics.shadowEdgeSupport, nullptr, &edgeSupportMaximum);
    auto probabilityRmse = [&](const PhotometricDiagnostics& diagnostics) {
        double squaredError = 0.0;
        long long samples = 0;
        for (const int index : diagnostics.shadowRefinementLightIndices) {
            if (index < 0 ||
                static_cast<size_t>(index) >= diagnostics.shadowPredictedBeforeProbability.size() ||
                diagnostics.shadowPredictedBeforeProbability[static_cast<size_t>(index)].empty()) {
                continue;
            }
            const cv::Mat& predicted =
                diagnostics.shadowPredictedBeforeProbability[static_cast<size_t>(index)];
            const cv::Mat& expected = fixture.castShadowFractions[static_cast<size_t>(index)];
            for (int y = 0; y < fixture.mask.rows; ++y) {
                const float* predictedRow = predicted.ptr<float>(y);
                const float* expectedRow = expected.ptr<float>(y);
                const uchar* maskRow = fixture.mask.ptr<uchar>(y);
                for (int x = 0; x < fixture.mask.cols; ++x) {
                    if (maskRow[x] == 0) {
                        continue;
                    }
                    const double difference = predictedRow[x] - expectedRow[x];
                    squaredError += difference * difference;
                    ++samples;
                }
            }
        }
        return samples > 0
            ? std::sqrt(squaredError / static_cast<double>(samples))
            : std::numeric_limits<double>::infinity();
    };
    const double correctProbabilityRmse = probabilityRmse(correctDiagnostics);
    const double pointProbabilityRmse = probabilityRmse(pointDiagnostics);
    std::cout << "finite_led_correct_mismatch="
              << correctDiagnostics.shadowMismatchRateBefore << '\n';
    std::cout << "finite_led_point_mismatch="
              << pointDiagnostics.shadowMismatchRateBefore << '\n';
    std::cout << "finite_led_wrong_datum_mismatch="
              << wrongDatumDiagnostics.shadowMismatchRateBefore << '\n';
    std::cout << "finite_led_probability_rmse=" << correctProbabilityRmse << '\n';
    std::cout << "point_source_probability_rmse=" << pointProbabilityRmse << '\n';
    context.check(
        correctDiagnostics.shadowMismatchRateBefore >= 0.0,
        "finite-LED fixture must provide enough coherent shadow evidence");
    context.check(
        correctProbabilityRmse < pointProbabilityRmse * 0.85,
        "finite-emitter visibility probabilities must fit finite-area shadow truth better than a point source");
    context.check(
        correctDiagnostics.shadowMismatchRateBefore + 0.005 <
            wrongDatumDiagnostics.shadowMismatchRateBefore,
        "the measured reference-surface Z must fit elevated-surface shadow truth better than Z=0");
    context.check(
        observabilityMaximum > 0.05 && edgeSupportMaximum > 0.0,
        "shadow refinement must expose non-empty observability and edge-support diagnostics");
    context.check(
        cv::countNonZero(correctDiagnostics.shadowOccluderSupport) ==
            cv::countNonZero(fixture.mask),
        "shadow occluder audit must preserve the supplied supported occluder domain");
}

void testShadowHeightRefinementNoEvidence(TestContext& context) {
    constexpr double ringRadiusMm = 4.2;
    constexpr double ringHeightMm = 1.15;
    constexpr double pixelScaleMm = 0.035;
    const NearFieldShadowFixture fixture = makeNearFieldShadowFixture(
        ringRadiusMm,
        ringHeightMm,
        pixelScaleMm);
    PhotometricDiagnostics diagnostics;
    diagnostics.shadowObservationMasks.assign(
        fixture.lights.size(),
        cv::Mat(fixture.mask.size(), CV_8U, cv::Scalar(0)));
    diagnostics.highlightObservationMasks.assign(
        fixture.lights.size(),
        cv::Mat(fixture.mask.size(), CV_8U, cv::Scalar(0)));
    diagnostics.saturationObservationMasks.assign(
        fixture.lights.size(),
        cv::Mat(fixture.mask.size(), CV_8U, cv::Scalar(0)));
    ShadowRefinementSettings settings;
    settings.lightingModel = LightingModel::NearFieldRing;
    settings.ringLightRadiusMm = ringRadiusMm;
    settings.ringLightHeightMm = ringHeightMm;
    settings.pixelScaleMm = pixelScaleMm;
    settings.lightingCenter = cv::Point2d(
        0.5 * (fixture.mask.cols - 1),
        0.5 * (fixture.mask.rows - 1));

    cv::Mat unchanged = fixture.initialHeight.clone();
    refineHeightFromCastShadows(
        unchanged,
        fixture.mask,
        fixture.mask,
        fixture.normals,
        fixture.images,
        fixture.lights,
        settings,
        diagnostics);
    context.check(
        !diagnostics.shadowHeightRefinementApplied,
        "shadow refinement must reject a stack with no coherent cast-shadow evidence");
    context.check(
        diagnostics.shadowHeightRefinementDecision ==
            "rejected_insufficient_coherent_shadow_lights",
        "shadow refinement must report why evidence was insufficient");
    context.check(
        cv::norm(unchanged, fixture.initialHeight, cv::NORM_INF) == 0.0,
        "rejected shadow refinement must leave the height field byte-for-byte unchanged");
}

ClassificationMetrics selectedClassificationMetrics(
    const std::vector<cv::Mat>& predicted,
    const std::vector<cv::Mat>& truth,
    const std::vector<int>& indices,
    const cv::Mat& mask) {
    std::vector<cv::Mat> selectedPredicted;
    std::vector<cv::Mat> selectedTruth;
    for (const int index : indices) {
        if (index < 0 || static_cast<size_t>(index) >= predicted.size() ||
            predicted[static_cast<size_t>(index)].empty()) {
            continue;
        }
        selectedPredicted.push_back(predicted[static_cast<size_t>(index)]);
        selectedTruth.push_back(truth[static_cast<size_t>(index)]);
    }
    if (selectedPredicted.empty()) {
        return ClassificationMetrics{};
    }
    return classificationMetrics(selectedPredicted, selectedTruth, mask);
}

void testMitsubaShadowHeightHoldout(TestContext& context) {
    constexpr double ringRadiusMm = 3.6;
    constexpr double ringHeightMm = 1.25;
    constexpr double pixelScaleMm = 2.0 / 72.0;
    const MitsubaFixture fixture = loadMitsubaFixture(
        "holdout_relief_v1",
        LightingModel::NearFieldRing,
        ringRadiusMm,
        ringHeightMm,
        pixelScaleMm);
    const SolveResult robust = solve(
        fixture.images,
        fixture.lights,
        NormalSolverMode::Robust,
        fixture.lightingModel,
        fixture.ringRadiusMm,
        fixture.ringHeightMm,
        fixture.pixelScaleMm,
        fixture.saturationMasks,
        true,
        fixture.solveMask);

    cv::Mat trueHeight = fixture.positionZ / pixelScaleMm;
    cv::Mat initialHeight = trueHeight.clone();
    for (int y = 0; y < initialHeight.rows; ++y) {
        float* row = initialHeight.ptr<float>(y);
        const double yn = 2.0 * static_cast<double>(y) /
            static_cast<double>(initialHeight.rows - 1) - 1.0;
        for (int x = 0; x < initialHeight.cols; ++x) {
            const double xn = 2.0 * static_cast<double>(x) /
                static_cast<double>(initialHeight.cols - 1) - 1.0;
            row[x] += static_cast<float>(
                2.8 * (0.48 * xn * xn - 0.31 * yn * yn + 0.24 * xn * yn +
                    0.16 * xn - 0.11 * yn));
        }
    }

    PhotometricDiagnostics diagnostics = robust.diagnostics;
    ShadowRefinementSettings settings;
    settings.lightingModel = LightingModel::NearFieldRing;
    settings.ringLightRadiusMm = ringRadiusMm;
    settings.ringLightHeightMm = ringHeightMm;
    settings.pixelScaleMm = pixelScaleMm;
    settings.lightingCenter = cv::Point2d(
        0.5 * (initialHeight.cols - 1),
        0.5 * (initialHeight.rows - 1));
    settings.maximumCoarseSide = 96;
    settings.iterations = 5;

    cv::Mat refined = initialHeight.clone();
    refineHeightFromCastShadows(
        refined,
        fixture.solveMask,
        fixture.solveMask,
        robust.normals,
        fixture.images,
        fixture.lights,
        settings,
        diagnostics,
        [](const std::string& message) {
            std::cout << "mitsuba_shadow_height_progress=" << message << '\n';
        });

    const double beforeRmse = normalizedHeightRmse(initialHeight, trueHeight, fixture.solveMask);
    const double afterRmse = normalizedHeightRmse(refined, trueHeight, fixture.solveMask);
    const ClassificationMetrics observedMetrics = selectedClassificationMetrics(
        diagnostics.shadowObservedCastMasks,
        fixture.castShadowTruth,
        diagnostics.shadowRefinementLightIndices,
        fixture.solveMask);
    const ClassificationMetrics predictedBeforeMetrics = selectedClassificationMetrics(
        diagnostics.shadowPredictedBeforeMasks,
        fixture.castShadowTruth,
        diagnostics.shadowRefinementLightIndices,
        fixture.solveMask);
    const ClassificationMetrics predictedAfterMetrics = selectedClassificationMetrics(
        diagnostics.shadowPredictedAfterMasks,
        fixture.castShadowTruth,
        diagnostics.shadowRefinementLightIndices,
        fixture.solveMask);
    PhotometricDiagnostics truthDiagnostics = robust.diagnostics;
    ShadowRefinementSettings truthSettings = settings;
    truthSettings.iterations = 1;
    cv::Mat truthAuditHeight = trueHeight.clone();
    refineHeightFromCastShadows(
        truthAuditHeight,
        fixture.solveMask,
        fixture.solveMask,
        robust.normals,
        fixture.images,
        fixture.lights,
        truthSettings,
        truthDiagnostics,
        [](const std::string& message) {
            std::cout << "mitsuba_truth_shadow_progress=" << message << '\n';
        });
    const double truthAuditRmse = normalizedHeightRmse(
        truthAuditHeight, trueHeight, fixture.solveMask);
    const ClassificationMetrics predictedTruthMetrics = selectedClassificationMetrics(
        truthDiagnostics.shadowPredictedBeforeMasks,
        fixture.castShadowTruth,
        truthDiagnostics.shadowRefinementLightIndices,
        fixture.solveMask);
    const double observedF1 = f1Score(observedMetrics);
    const double predictedBeforeF1 = f1Score(predictedBeforeMetrics);
    const double predictedAfterF1 = f1Score(predictedAfterMetrics);
    const double predictedTruthF1 = f1Score(predictedTruthMetrics);
    std::cout << "mitsuba_shadow_height_applied=" << diagnostics.shadowHeightRefinementApplied << '\n';
    std::cout << "mitsuba_cast_observation_f1=" << observedF1 << '\n';
    std::cout << "mitsuba_cast_prediction_f1_before=" << predictedBeforeF1 << '\n';
    std::cout << "mitsuba_cast_prediction_f1_after=" << predictedAfterF1 << '\n';
    std::cout << "mitsuba_cast_prediction_f1_true_height=" << predictedTruthF1 << '\n';
    std::cout << "mitsuba_cast_prediction_true_precision=" << predictedTruthMetrics.precision << '\n';
    std::cout << "mitsuba_cast_prediction_true_recall=" << predictedTruthMetrics.recall << '\n';
    std::cout << "mitsuba_truth_start_applied="
              << truthDiagnostics.shadowHeightRefinementApplied << '\n';
    std::cout << "mitsuba_truth_start_height_rmse=" << truthAuditRmse << '\n';
    std::cout << "mitsuba_truth_start_mismatch_before="
              << truthDiagnostics.shadowMismatchRateBefore << '\n';
    std::cout << "mitsuba_truth_start_mismatch_after="
              << truthDiagnostics.shadowMismatchRateAfter << '\n';
    std::cout << "mitsuba_truth_start_slope_before="
              << truthDiagnostics.shadowNormalSlopeRmsBefore << '\n';
    std::cout << "mitsuba_truth_start_slope_after="
              << truthDiagnostics.shadowNormalSlopeRmsAfter << '\n';
    std::cout << "mitsuba_shadow_height_rmse_before=" << beforeRmse << '\n';
    std::cout << "mitsuba_shadow_height_rmse_after=" << afterRmse << '\n';
    std::cout << "mitsuba_shadow_correction_rms=" << diagnostics.shadowCorrectionRms << '\n';
    std::cout << "mitsuba_shadow_slope_rms_before=" << diagnostics.shadowNormalSlopeRmsBefore << '\n';
    std::cout << "mitsuba_shadow_slope_rms_after=" << diagnostics.shadowNormalSlopeRmsAfter << '\n';
    context.check(
        diagnostics.shadowRefinementLightIndices.size() >= 6,
        "Mitsuba holdout must provide coherent cast-shadow evidence in at least six lights");
    context.check(
        observedF1 > 0.72,
        "regularized robust cast-shadow evidence did not agree with Mitsuba truth");
    context.check(
        diagnostics.shadowHeightRefinementApplied,
        "Mitsuba broad-drift correction must pass the internal held-out-light gate");
    context.check(
        diagnostics.shadowHoldoutMismatchRateAfter <= diagnostics.shadowHoldoutMismatchRateBefore,
        "Mitsuba correction must not worsen internal held-out shadow agreement");
    context.check(
        predictedAfterF1 >= predictedBeforeF1,
        "Mitsuba cast-shadow prediction F1 must not decline after accepted correction");
    context.check(
        afterRmse < beforeRmse * 0.97,
        "Mitsuba shadow refinement must reduce the injected broad height drift");
    context.check(
        !truthDiagnostics.shadowHeightRefinementApplied && truthAuditRmse == 0.0,
        "Mitsuba renderer-truth height must pass through the refinement unchanged");
}

void testHeightIntegration(TestContext& context) {
    cv::Mat expectedHeight;
    cv::Mat normals;
    makeHeightFixture(expectedHeight, normals);
    const cv::Mat fullMask(expectedHeight.size(), CV_8U, cv::Scalar(255));

    const cv::Mat fastHeight = integrateHeight(
        normals, fullMask, HeightSolverMode::FastDct, 3.0, 800);
    const double fastError = normalizedHeightRmse(fastHeight, expectedHeight, fullMask);
    std::cout << "full_dct_normalized_height_rmse=" << fastError << '\n';
    context.check(fastError < 0.08, "full-field DCT normalized height RMSE was " + std::to_string(fastError));

    cv::Mat irregularMask(expectedHeight.size(), CV_8U, cv::Scalar(0));
    const double cx = static_cast<double>(expectedHeight.cols - 1) * 0.5;
    const double cy = static_cast<double>(expectedHeight.rows - 1) * 0.5;
    for (int y = 0; y < irregularMask.rows; ++y) {
        uchar* row = irregularMask.ptr<uchar>(y);
        for (int x = 0; x < irregularMask.cols; ++x) {
            const double ex = (static_cast<double>(x) - cx) / 34.0;
            const double ey = (static_cast<double>(y) - cy) / 27.0;
            const double hx = static_cast<double>(x) - (cx + 8.0);
            const double hy = static_cast<double>(y) - (cy - 5.0);
            if (ex * ex + ey * ey <= 1.0 && hx * hx + hy * hy > 16.0) {
                row[x] = 255;
            }
        }
    }

    const cv::Mat fastMaskedHeight = integrateHeight(
        normals, irregularMask, HeightSolverMode::FastDct, 3.0, 800);
    const double fastMaskedError = normalizedHeightRmse(fastMaskedHeight, expectedHeight, irregularMask);
    const cv::Mat robustHeight = integrateHeight(
        normals, irregularMask, HeightSolverMode::RobustMasked, 3.0, 800);
    const double robustError = normalizedHeightRmse(robustHeight, expectedHeight, irregularMask);
    std::cout << "masked_dct_normalized_height_rmse=" << fastMaskedError << '\n';
    std::cout << "masked_robust_normalized_height_rmse=" << robustError << '\n';
    context.check(
        robustError < fastMaskedError * 0.70,
        "irregular-mask robust integration must improve masked DCT RMSE by at least 30 percent");
    context.check(
        robustError < 0.07,
        "irregular-mask robust normalized height RMSE was " + std::to_string(robustError));
}

void testHeightFlatteningSemantics(TestContext& context) {
    constexpr int rows = 31;
    constexpr int cols = 37;
    const cv::Mat mask(rows, cols, CV_8U, cv::Scalar(255));
    cv::Mat plane(rows, cols, CV_32F);
    for (int y = 0; y < rows; ++y) {
        float* row = plane.ptr<float>(y);
        for (int x = 0; x < cols; ++x) {
            row[x] = static_cast<float>(2.3 + 0.13 * x - 0.08 * y);
        }
    }

    cv::Mat unchanged = plane.clone();
    applyHeightFlattening(unchanged, mask, HeightFlattenMode::None);
    context.check(
        cv::norm(unchanged, plane, cv::NORM_INF) == 0.0,
        "height flattening None must leave the integrated field byte-for-byte unchanged");

    cv::Mat leveled = plane.clone();
    applyHeightFlattening(leveled, mask, HeightFlattenMode::Plane);
    context.check(
        cv::norm(leveled, cv::NORM_INF) < 1.0e-4,
        "plane leveling must remove a least-squares affine height trend");

    const double cx = 0.5 * static_cast<double>(cols - 1);
    const double cy = 0.5 * static_cast<double>(rows - 1);
    const double scale = 0.5 * static_cast<double>(std::max(cols, rows));
    cv::Mat radial(rows, cols, CV_32F);
    cv::Mat quadratic(rows, cols, CV_32F);
    for (int y = 0; y < rows; ++y) {
        float* radialRow = radial.ptr<float>(y);
        float* quadraticRow = quadratic.ptr<float>(y);
        const double yn = (static_cast<double>(y) - cy) / scale;
        for (int x = 0; x < cols; ++x) {
            const double xn = (static_cast<double>(x) - cx) / scale;
            radialRow[x] = static_cast<float>(1.1 + 0.2 * xn - 0.3 * yn + 0.8 * (xn * xn + yn * yn));
            quadraticRow[x] = static_cast<float>(
                0.7 + 0.1 * xn - 0.2 * yn + 0.4 * xn * xn - 0.6 * yn * yn + 0.3 * xn * yn);
        }
    }
    applyHeightFlattening(radial, mask, HeightFlattenMode::Radial);
    applyHeightFlattening(quadratic, mask, HeightFlattenMode::Quadratic);
    context.check(cv::norm(radial, cv::NORM_INF) < 1.0e-4, "radial mode must remove its documented dome basis");
    context.check(
        cv::norm(quadratic, cv::NORM_INF) < 1.0e-4,
        "quadratic mode must remove its documented second-order basis");
}

} // namespace

int main() {
    TestContext context;
    testCalibrationIdentityAndValidation(context);
    testNearFieldRingModel(context);
    testRadiometricConversion(context);
    testObservationValidityMask(context);
    testEncodedNormalRecovery(context);
    testCleanLambertianCounts(context);
    testShadowOmission(context);
    testPenumbraRobustness(context);
    testNarrowHighlightRobustness(context);
    testFourLightSaturatedHighlight(context);
    testRobustHighlightCounts(context);
    testBroadGlossRobustness(context);
    testBrightDiffuseObservationIsRetained(context);
    testUnsupportedPixelsAreNotForceFit(context);
    testMitsubaMixedMaterialRecovery(context);
    testAdditionalMitsubaFixtures(context);
    testNearFieldShadowHeightRefinement(context);
    testDirectionalShadowHeightRefinement(context);
    testFiniteLedAndReferenceSurfaceGeometry(context);
    testShadowHeightRefinementNoEvidence(context);
    testMitsubaShadowHeightHoldout(context);
    testHeightIntegration(context);
    testHeightFlatteningSemantics(context);

    if (context.failures != 0) {
        std::cerr << context.failures << " scientific regression check(s) failed.\n";
        return 1;
    }
    std::cout << "All photometric and height regression checks passed.\n";
    return 0;
}
