#include "photometric.hpp"
#include "radiometry.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
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
    double pixelScaleMm = 0.01) {
    SolveResult result;
    const cv::Mat mask(images.front().size(), CV_8U, cv::Scalar(255));
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
        result.diagnostics);
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
    for (const int count : {3, 4, 8, 25}) {
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
    for (const int count : {5, 8, 25}) {
        const std::vector<cv::Vec3f> lights = makeRingLights(count);
        std::vector<cv::Mat> images = renderLambertianPlane(9, 11, lights, expected, 0.67f);
        const int injectedOutliers = count == 5 ? 1 : count == 8 ? 2 : 4;
        for (int i = 0; i < injectedOutliers; ++i) {
            images[static_cast<size_t>(i)].setTo(cv::Scalar(1.0));
        }
        const SolveResult robust = solve(images, lights, NormalSolverMode::Robust);
        const double error = meanAngularErrorDegrees(robust.normals, robust.validMask, expected);
        const double outlierCount = cv::mean(robust.diagnostics.highlightOutlierCount)[0];
        std::cout << "robust_" << count << "_light_highlight_mean_degrees=" << error << '\n';
        context.check(
            error < 0.10,
            std::to_string(count) + "-light robust highlight error was " + std::to_string(error));
        context.check(
            outlierCount >= static_cast<double>(injectedOutliers) - 0.01,
            std::to_string(count) + "-light solve did not report every injected highlight");
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
        cueRate > 0.65,
        "broad-gloss half-vector diagnostic cue rate was " + std::to_string(cueRate));
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
    testHeightIntegration(context);
    testHeightFlatteningSemantics(context);

    if (context.failures != 0) {
        std::cerr << context.failures << " scientific regression check(s) failed.\n";
        return 1;
    }
    std::cout << "All photometric and height regression checks passed.\n";
    return 0;
}
