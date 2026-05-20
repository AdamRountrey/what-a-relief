#include "neural_fusion.hpp"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kNeuralChannelsPerImage = 3;
constexpr int kDefaultFallbackNeuralSide = 1024;

[[noreturn]] void die(const std::string& message) {
    throw std::runtime_error(message);
}

cv::Vec3f normalizeVec3(const cv::Vec3f& value) {
    const float length = std::sqrt(value.dot(value));
    if (!std::isfinite(length) || length <= 1.0e-8f) {
        return cv::Vec3f(0.0f, 0.0f, 1.0f);
    }
    return value / length;
}

cv::Vec3d normalizeVec3d(const cv::Vec3d& value) {
    const double length = std::sqrt(value.dot(value));
    if (!std::isfinite(length) || length <= 1.0e-12) {
        return cv::Vec3d(0.0, 0.0, 1.0);
    }
    return value / length;
}

std::string executableDirectory() {
#ifdef _WIN32
    char path[MAX_PATH] = {};
    const DWORD length = GetModuleFileNameA(nullptr, path, static_cast<DWORD>(sizeof(path)));
    if (length > 0 && length < sizeof(path)) {
        return fs::path(std::string(path, path + length)).parent_path().string();
    }
#endif
    return fs::current_path().string();
}

std::string resolveModelPath(const Options& opt, size_t imageCount) {
    const std::string modelName = "psfcn_" + std::to_string(imageCount) + "_normalize.onnx";
    std::vector<fs::path> candidates;
    if (!opt.neuralModelPath.empty()) {
        fs::path overridePath(opt.neuralModelPath);
        if (fs::is_directory(overridePath)) {
            candidates.emplace_back(overridePath / modelName);
        } else {
            candidates.emplace_back(overridePath);
        }
    }

    const fs::path exeDir = executableDirectory();
    candidates.emplace_back(exeDir / "models" / modelName);
    candidates.emplace_back(fs::current_path() / "models" / modelName);
    candidates.emplace_back(fs::current_path() / "assets" / "models" / modelName);

    for (const fs::path& candidate : candidates) {
        if (!candidate.empty() && fs::exists(candidate)) {
            return candidate.string();
        }
    }
    die(
        "Could not find bundled PS-FCN model " + modelName + ". "
        "Looked next to the executable under models/, in the current folder's models/, and in assets/models/.");
}

cv::Mat normalizeMaskedToUnitRange(const cv::Mat& src, const cv::Mat& mask) {
    double maxValue = 0.0;
    for (int y = 0; y < src.rows; ++y) {
        const float* row = src.ptr<float>(y);
        const uchar* maskRow = mask.ptr<uchar>(y);
        for (int x = 0; x < src.cols; ++x) {
            if (maskRow[x] == 0 || !std::isfinite(row[x])) {
                continue;
            }
            maxValue = std::max(maxValue, static_cast<double>(row[x]));
        }
    }
    cv::Mat normalized(src.size(), CV_32F, cv::Scalar(0));
    if (maxValue <= 1.0e-12) {
        return normalized;
    }
    const float scale = static_cast<float>(1.0 / maxValue);
    for (int y = 0; y < src.rows; ++y) {
        const float* row = src.ptr<float>(y);
        const uchar* maskRow = mask.ptr<uchar>(y);
        float* outRow = normalized.ptr<float>(y);
        for (int x = 0; x < src.cols; ++x) {
            if (maskRow[x] != 0 && std::isfinite(row[x])) {
                outRow[x] = std::clamp(row[x] * scale, 0.0f, 1.0f);
            }
        }
    }
    return normalized;
}

cv::Mat maskedGaussianBlurFloat(const cv::Mat& src, const cv::Mat& mask, double sigma) {
    cv::Mat weighted = cv::Mat::zeros(src.size(), CV_32F);
    cv::Mat weights = cv::Mat::zeros(src.size(), CV_32F);
    for (int y = 0; y < src.rows; ++y) {
        const float* srcRow = src.ptr<float>(y);
        const uchar* maskRow = mask.ptr<uchar>(y);
        float* weightedRow = weighted.ptr<float>(y);
        float* weightsRow = weights.ptr<float>(y);
        for (int x = 0; x < src.cols; ++x) {
            if (maskRow[x] != 0 && std::isfinite(srcRow[x])) {
                weightedRow[x] = srcRow[x];
                weightsRow[x] = 1.0f;
            }
        }
    }

    cv::GaussianBlur(weighted, weighted, cv::Size(), sigma, sigma, cv::BORDER_REPLICATE);
    cv::GaussianBlur(weights, weights, cv::Size(), sigma, sigma, cv::BORDER_REPLICATE);

    cv::Mat result = cv::Mat::zeros(src.size(), CV_32F);
    for (int y = 0; y < src.rows; ++y) {
        const float* weightedRow = weighted.ptr<float>(y);
        const float* weightsRow = weights.ptr<float>(y);
        float* resultRow = result.ptr<float>(y);
        for (int x = 0; x < src.cols; ++x) {
            if (weightsRow[x] > 1.0e-6f) {
                resultRow[x] = weightedRow[x] / weightsRow[x];
            }
        }
    }
    return result;
}

cv::Mat maskedGaussianBlurVec3(const cv::Mat& src, const cv::Mat& mask, double sigma) {
    std::vector<cv::Mat> channels;
    cv::split(src, channels);
    for (cv::Mat& channel : channels) {
        channel = maskedGaussianBlurFloat(channel, mask, sigma);
    }
    cv::Mat result;
    cv::merge(channels, result);
    return result;
}

float maskedPercentile(std::vector<float>& values, float percentile) {
    if (values.empty()) {
        return 0.0f;
    }
    const size_t index = static_cast<size_t>(
        std::clamp(percentile, 0.0f, 1.0f) * static_cast<float>(values.size() - 1));
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(index), values.end());
    return values[index];
}

void normalToSlopes(
    const cv::Mat& normalMap,
    const cv::Mat& mask,
    cv::Mat& p,
    cv::Mat& q) {
    p = cv::Mat(normalMap.size(), CV_32F, cv::Scalar(0));
    q = cv::Mat(normalMap.size(), CV_32F, cv::Scalar(0));
    for (int y = 0; y < normalMap.rows; ++y) {
        const cv::Vec3f* normalRow = normalMap.ptr<cv::Vec3f>(y);
        const uchar* maskRow = mask.ptr<uchar>(y);
        float* pRow = p.ptr<float>(y);
        float* qRow = q.ptr<float>(y);
        for (int x = 0; x < normalMap.cols; ++x) {
            if (maskRow[x] == 0) {
                continue;
            }
            const cv::Vec3f normal = normalizeVec3(normalRow[x]);
            const float nz = std::max(std::abs(normal[2]), 1.0e-4f);
            pRow[x] = -normal[0] / nz;
            qRow[x] = normal[1] / nz;
        }
    }
}

cv::Mat slopesToNormals(const cv::Mat& p, const cv::Mat& q, const cv::Mat& mask) {
    cv::Mat normalMap(p.size(), CV_32FC3, cv::Scalar(0, 0, 0));
    for (int y = 0; y < p.rows; ++y) {
        const float* pRow = p.ptr<float>(y);
        const float* qRow = q.ptr<float>(y);
        const uchar* maskRow = mask.ptr<uchar>(y);
        cv::Vec3f* normalRow = normalMap.ptr<cv::Vec3f>(y);
        for (int x = 0; x < p.cols; ++x) {
            if (maskRow[x] == 0) {
                continue;
            }
            const float px = std::isfinite(pRow[x]) ? pRow[x] : 0.0f;
            const float qx = std::isfinite(qRow[x]) ? qRow[x] : 0.0f;
            const float invLength = 1.0f / std::sqrt(px * px + qx * qx + 1.0f);
            normalRow[x] = cv::Vec3f(-px * invLength, qx * invLength, invLength);
        }
    }
    return normalMap;
}

void normalizeNormalMapInPlace(cv::Mat& normalMap, const cv::Mat& mask) {
    for (int y = 0; y < normalMap.rows; ++y) {
        cv::Vec3f* row = normalMap.ptr<cv::Vec3f>(y);
        const uchar* maskRow = mask.ptr<uchar>(y);
        for (int x = 0; x < normalMap.cols; ++x) {
            if (maskRow[x] != 0) {
                row[x] = normalizeVec3(row[x]);
            } else {
                row[x] = cv::Vec3f(0.0f, 0.0f, 0.0f);
            }
        }
    }
}

cv::Mat computeClassicalConfidence(
    const cv::Mat& residual,
    const cv::Mat& classicalValidMask,
    const PhotometricDiagnostics& diagnostics) {
    const cv::Mat residualUnit = normalizeMaskedToUnitRange(residual, classicalValidMask);
    const cv::Mat shadowUnit = normalizeMaskedToUnitRange(diagnostics.shadowCount, classicalValidMask);
    const cv::Mat highlightUnit = normalizeMaskedToUnitRange(diagnostics.highlightOutlierCount, classicalValidMask);

    cv::Mat confidence(residual.size(), CV_32F, cv::Scalar(0));
    for (int y = 0; y < residual.rows; ++y) {
        const float* residualRow = residualUnit.ptr<float>(y);
        const float* robustRow = diagnostics.robustWeight.empty() ? nullptr : diagnostics.robustWeight.ptr<float>(y);
        const float* shadowRow = shadowUnit.ptr<float>(y);
        const float* highlightRow = highlightUnit.ptr<float>(y);
        const uchar* validRow = classicalValidMask.ptr<uchar>(y);
        float* outRow = confidence.ptr<float>(y);
        for (int x = 0; x < residual.cols; ++x) {
            if (validRow[x] == 0) {
                outRow[x] = 0.0f;
                continue;
            }
            float value = 1.0f - 0.90f * residualRow[x];
            if (robustRow != nullptr) {
                value = 0.55f * value + 0.45f * robustRow[x];
            }
            value -= 0.18f * shadowRow[x];
            value -= 0.14f * highlightRow[x];
            outRow[x] = std::clamp(value, 0.0f, 1.0f);
        }
    }
    return confidence;
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

cv::Mat runPsfcnInferenceAtMaxSide(
    const std::string& modelPath,
    const std::vector<cv::Mat>& images,
    const std::vector<cv::Vec3f>& lights,
    int maxSide) {
    const int originalRows = images.front().rows;
    const int originalCols = images.front().cols;
    const int originalLongSide = std::max(originalRows, originalCols);
    const double inferenceScale = maxSide <= 0 || maxSide >= originalLongSide
        ? 1.0
        : static_cast<double>(maxSide) / static_cast<double>(originalLongSide);
    const int inferenceRows = std::max(4, static_cast<int>(std::lround(static_cast<double>(originalRows) * inferenceScale)));
    const int inferenceCols = std::max(4, static_cast<int>(std::lround(static_cast<double>(originalCols) * inferenceScale)));
    const int paddedRows = ((inferenceRows + 3) / 4) * 4;
    const int paddedCols = ((inferenceCols + 3) / 4) * 4;
    const int imageCount = static_cast<int>(images.size());
    const float neuralNormalizeScale = std::sqrt(static_cast<float>(imageCount) / 32.0f);
    const int channels = imageCount * kNeuralChannelsPerImage;
    const int planeSize = paddedRows * paddedCols;

    if (inferenceScale < 0.999) {
        std::cout << "      PS-FCN neural prior runs at "
                  << inferenceCols << "x" << inferenceRows
                  << " and is upsampled to the input image size." << std::endl;
    }

    std::vector<cv::Mat> paddedImages;
    paddedImages.reserve(images.size());
    cv::Mat normAccumulator(paddedRows, paddedCols, CV_32F, cv::Scalar(0));
    for (const cv::Mat& image : images) {
        cv::Mat padded(paddedRows, paddedCols, CV_32F, cv::Scalar(0));
        cv::Mat inferenceImage;
        if (inferenceScale < 0.999) {
            cv::resize(image, inferenceImage, cv::Size(inferenceCols, inferenceRows), 0.0, 0.0, cv::INTER_AREA);
        } else {
            inferenceImage = image;
        }
        inferenceImage.copyTo(padded(cv::Rect(0, 0, inferenceImage.cols, inferenceImage.rows)));
        paddedImages.push_back(padded);
        cv::Mat squared;
        cv::multiply(padded, padded, squared);
        normAccumulator += squared * 3.0f;
    }
    cv::sqrt(normAccumulator, normAccumulator);

    int imageSizes[4] = {1, channels, paddedRows, paddedCols};
    cv::Mat imageBlob(4, imageSizes, CV_32F, cv::Scalar(0));
    cv::Mat lightBlob(4, imageSizes, CV_32F, cv::Scalar(0));
    float* imageData = reinterpret_cast<float*>(imageBlob.data);
    float* lightData = reinterpret_cast<float*>(lightBlob.data);

    for (size_t i = 0; i < paddedImages.size(); ++i) {
        const cv::Vec3f light = normalizeVec3(lights[i]);
        for (int y = 0; y < paddedRows; ++y) {
            const float* imageRow = paddedImages[i].ptr<float>(y);
            const float* normRow = normAccumulator.ptr<float>(y);
            for (int x = 0; x < paddedCols; ++x) {
                const int pixelIndex = y * paddedCols + x;
                const float denominator = std::max(normRow[x], 1.0e-10f);
                const float normalizedValue = imageRow[x] / denominator * neuralNormalizeScale;
                const int channelBase = static_cast<int>(i) * kNeuralChannelsPerImage;
                imageData[(channelBase + 0) * planeSize + pixelIndex] = normalizedValue;
                imageData[(channelBase + 1) * planeSize + pixelIndex] = normalizedValue;
                imageData[(channelBase + 2) * planeSize + pixelIndex] = normalizedValue;
                lightData[(channelBase + 0) * planeSize + pixelIndex] = light[0];
                lightData[(channelBase + 1) * planeSize + pixelIndex] = light[1];
                lightData[(channelBase + 2) * planeSize + pixelIndex] = light[2];
            }
        }
    }

    cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    net.setInput(imageBlob, "image_stack");
    net.setInput(lightBlob, "light_stack");
    cv::Mat blobOutput = net.forward("normal_map");

    std::vector<cv::Mat> outputImages;
    cv::dnn::imagesFromBlob(blobOutput, outputImages);
    if (outputImages.size() != 1 || outputImages[0].channels() != 3) {
        die("Unexpected PS-FCN output shape from bundled ONNX model.");
    }

    cv::Mat neuralNormal = outputImages[0](cv::Rect(0, 0, inferenceCols, inferenceRows)).clone();
    if (inferenceScale < 0.999) {
        cv::resize(neuralNormal, neuralNormal, cv::Size(originalCols, originalRows), 0.0, 0.0, cv::INTER_LINEAR);
    }
    normalizeNormalMapInPlace(neuralNormal, cv::Mat(neuralNormal.size(), CV_8U, cv::Scalar(255)));
    return neuralNormal;
}

cv::Mat runPsfcnInference(
    const Options& opt,
    const std::vector<cv::Mat>& images,
    const std::vector<cv::Vec3f>& lights) {
    if (images.size() < 3 || images.size() > 25 || lights.size() != images.size()) {
        die("Experimental neural fusion currently supports 3 to 25 calibrated images.");
    }

    const int originalLongSide = std::max(images.front().rows, images.front().cols);
    const std::string modelPath = resolveModelPath(opt, images.size());
    std::vector<int> attempts;
    attempts.push_back(opt.neuralMaxSide <= 0 ? originalLongSide : opt.neuralMaxSide);
    if (attempts.back() > kDefaultFallbackNeuralSide) {
        attempts.push_back(kDefaultFallbackNeuralSide);
    }
    if (attempts.back() > 768) {
        attempts.push_back(768);
    }

    std::string lastError;
    for (int maxSide : attempts) {
        try {
            return runPsfcnInferenceAtMaxSide(modelPath, images, lights, maxSide);
        } catch (const cv::Exception& e) {
            lastError = e.what();
            if (maxSide == attempts.back()) {
                break;
            }
            std::cerr << "warning: PS-FCN inference failed at max side "
                      << maxSide << "; retrying at lower resolution." << std::endl;
        } catch (const std::bad_alloc& e) {
            lastError = e.what();
            if (maxSide == attempts.back()) {
                break;
            }
            std::cerr << "warning: PS-FCN inference ran out of memory at max side "
                      << maxSide << "; retrying at lower resolution." << std::endl;
        }
    }

    die("PS-FCN inference failed at all attempted resolutions. Last error: " + lastError);
}

cv::Mat fuseNormals(
    const cv::Mat& classicalNormal,
    const cv::Mat& classicalValidMask,
    const cv::Mat& neuralNormal,
    const cv::Mat& fusionMask,
    const cv::Mat& confidence) {
    cv::Mat classicalP;
    cv::Mat classicalQ;
    cv::Mat neuralP;
    cv::Mat neuralQ;
    normalToSlopes(classicalNormal, classicalValidMask, classicalP, classicalQ);
    normalToSlopes(neuralNormal, fusionMask, neuralP, neuralQ);

    cv::Mat classicalPLow = maskedGaussianBlurFloat(classicalP, classicalValidMask, 5.0);
    cv::Mat classicalQLow = maskedGaussianBlurFloat(classicalQ, classicalValidMask, 5.0);
    cv::Mat neuralPLow = maskedGaussianBlurFloat(neuralP, fusionMask, 4.5);
    cv::Mat neuralQLow = maskedGaussianBlurFloat(neuralQ, fusionMask, 4.5);

    std::vector<float> classicalSlopeMagnitudes;
    classicalSlopeMagnitudes.reserve(static_cast<size_t>(classicalNormal.rows * classicalNormal.cols / 4));
    for (int y = 0; y < classicalP.rows; ++y) {
        const float* pRow = classicalP.ptr<float>(y);
        const float* qRow = classicalQ.ptr<float>(y);
        const uchar* maskRow = classicalValidMask.ptr<uchar>(y);
        for (int x = 0; x < classicalP.cols; ++x) {
            if (maskRow[x] != 0) {
                classicalSlopeMagnitudes.push_back(std::sqrt(pRow[x] * pRow[x] + qRow[x] * qRow[x]));
            }
        }
    }
    const float globalSlopeCap = std::max(0.2f, maskedPercentile(classicalSlopeMagnitudes, 0.98f) * 1.10f);

    cv::Mat fusedP(classicalP.size(), CV_32F, cv::Scalar(0));
    cv::Mat fusedQ(classicalQ.size(), CV_32F, cv::Scalar(0));
    for (int y = 0; y < fusedP.rows; ++y) {
        const float* classicalPRow = classicalP.ptr<float>(y);
        const float* classicalQRow = classicalQ.ptr<float>(y);
        const float* classicalPLowRow = classicalPLow.ptr<float>(y);
        const float* classicalQLowRow = classicalQLow.ptr<float>(y);
        const float* neuralPLowRow = neuralPLow.ptr<float>(y);
        const float* neuralQLowRow = neuralQLow.ptr<float>(y);
        const uchar* classicalValidRow = classicalValidMask.ptr<uchar>(y);
        const uchar* fusionMaskRow = fusionMask.ptr<uchar>(y);
        const float* confidenceRow = confidence.ptr<float>(y);
        float* fusedPRow = fusedP.ptr<float>(y);
        float* fusedQRow = fusedQ.ptr<float>(y);
        for (int x = 0; x < fusedP.cols; ++x) {
            if (fusionMaskRow[x] == 0) {
                continue;
            }

            const float confidenceValue = confidenceRow[x];
            const float neuralWeight = classicalValidRow[x] != 0
                ? (0.04f + 0.28f * (1.0f - confidenceValue))
                : 0.85f;
            const float classicalWeight = 1.0f - neuralWeight;

            const float baseP = classicalWeight * classicalPLowRow[x] + neuralWeight * neuralPLowRow[x];
            const float baseQ = classicalWeight * classicalQLowRow[x] + neuralWeight * neuralQLowRow[x];

            const float detailWeight = classicalValidRow[x] != 0
                ? (0.62f * confidenceValue + 0.12f)
                : 0.0f;
            float pValue = baseP + (classicalPRow[x] - classicalPLowRow[x]) * detailWeight;
            float qValue = baseQ + (classicalQRow[x] - classicalQLowRow[x]) * detailWeight;

            const float baseMagnitude = std::sqrt(baseP * baseP + baseQ * baseQ);
            const float localCap = std::max(0.18f, std::min(globalSlopeCap * 1.05f, baseMagnitude * 1.38f + 0.12f));
            const float fusedMagnitude = std::sqrt(pValue * pValue + qValue * qValue);
            if (fusedMagnitude > localCap) {
                const float scale = localCap / std::max(fusedMagnitude, 1.0e-6f);
                pValue *= scale;
                qValue *= scale;
            }

            fusedPRow[x] = pValue;
            fusedQRow[x] = qValue;
        }
    }
    return slopesToNormals(fusedP, fusedQ, fusionMask);
}

void refitDiffuseOutputs(
    const Options& opt,
    const std::vector<cv::Mat>& images,
    const std::vector<cv::Vec3f>& lights,
    const cv::Mat& solveMask,
    const cv::Mat& normalMap,
    cv::Point2d lightingCenter,
    cv::Mat& albedo,
    cv::Mat& residual,
    cv::Mat& validMask) {
    const int rows = normalMap.rows;
    const int cols = normalMap.cols;
    const int lightCount = static_cast<int>(lights.size());
    albedo = cv::Mat(rows, cols, CV_32F, cv::Scalar(0));
    residual = cv::Mat(rows, cols, CV_32F, cv::Scalar(0));
    validMask = solveMask.clone();

    for (int y = 0; y < rows; ++y) {
        const uchar* maskRow = solveMask.ptr<uchar>(y);
        const cv::Vec3f* normalRow = normalMap.ptr<cv::Vec3f>(y);
        float* albedoRow = albedo.ptr<float>(y);
        float* residualRow = residual.ptr<float>(y);
        uchar* validRow = validMask.ptr<uchar>(y);
        std::vector<const float*> imageRows;
        imageRows.reserve(images.size());
        for (const cv::Mat& image : images) {
            imageRows.push_back(image.ptr<float>(y));
        }

        for (int x = 0; x < cols; ++x) {
            if (maskRow[x] == 0) {
                validRow[x] = 0;
                continue;
            }
            const cv::Vec3f normal = normalizeVec3(normalRow[x]);
            double numerator = 0.0;
            double denominator = 0.0;
            int usedObservations = 0;
            std::vector<double> nDotL;
            std::vector<double> observed;
            nDotL.reserve(images.size());
            observed.reserve(images.size());

            for (int i = 0; i < lightCount; ++i) {
                const cv::Vec3f light = opt.lightingModel == LightingModel::NearFieldRing
                    ? nearFieldRingLightDirection(
                          lights[i],
                          i,
                          lightCount,
                          x,
                          y,
                          opt.ringLightRadiusMm,
                          opt.ringLightHeightMm,
                          opt.pixelScaleMm,
                          lightingCenter)
                    : normalizeVec3(lights[i]);
                const double ndotl = std::max(0.0f, normal.dot(light));
                const float intensity = imageRows[i][x];
                if (!std::isfinite(intensity) || intensity <= opt.shadowThreshold || ndotl <= 1.0e-6) {
                    continue;
                }
                numerator += ndotl * static_cast<double>(intensity);
                denominator += ndotl * ndotl;
                nDotL.push_back(ndotl);
                observed.push_back(intensity);
                ++usedObservations;
            }

            const double rho = denominator > 1.0e-8 ? numerator / denominator : 0.0;
            albedoRow[x] = static_cast<float>(std::max(0.0, rho));

            if (usedObservations == 0) {
                residualRow[x] = 0.0f;
                continue;
            }

            double error = 0.0;
            for (size_t i = 0; i < nDotL.size(); ++i) {
                const double diff = rho * nDotL[i] - observed[i];
                error += diff * diff;
            }
            residualRow[x] = static_cast<float>(std::sqrt(error / static_cast<double>(nDotL.size())));
        }
    }
}

} // namespace

void applyNeuralFusion(
    const Options& opt,
    const std::vector<cv::Mat>& images,
    const std::vector<cv::Vec3f>& lights,
    const cv::Mat& solveMask,
    cv::Point2d lightingCenter,
    cv::Mat& normalMap,
    cv::Mat& albedo,
    cv::Mat& residual,
    cv::Mat& validMask,
    PhotometricDiagnostics& diagnostics) {
    if (!opt.neuralFusion) {
        return;
    }
    if (opt.uncalibratedLighting) {
        die("Experimental neural fusion is only available for calibrated lighting.");
    }

    const cv::Mat classicalNormal = normalMap.clone();
    const cv::Mat classicalValidMask = validMask.clone();
    const cv::Mat confidence = computeClassicalConfidence(residual, classicalValidMask, diagnostics);
    const cv::Mat neuralNormal = runPsfcnInference(opt, images, lights);
    const cv::Mat fusedNormal = fuseNormals(classicalNormal, classicalValidMask, neuralNormal, solveMask, confidence);

    normalMap = fusedNormal;
    diagnostics.classicalConfidence = confidence;
    diagnostics.classicalNormal = classicalNormal;
    diagnostics.neuralNormal = neuralNormal;
    refitDiffuseOutputs(opt, images, lights, solveMask, normalMap, lightingCenter, albedo, residual, validMask);
}
