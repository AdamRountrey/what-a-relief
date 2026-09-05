#include "shadow_refinement.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace {

constexpr float kFrontLitThreshold = 0.05f;
constexpr int kMaximumRefinementLights = 16;
constexpr double kCorrectionPriorWeight = 0.02;
constexpr double kNormalSlopePriorWeight = 8.0;

struct CoarseGrid {
    cv::Size size;
    double scaleX = 1.0;
    double scaleY = 1.0;
};

struct RayResult {
    float clearance = -std::numeric_limits<float>::infinity();
    cv::Point blocker = cv::Point(-1, -1);
};

struct VisibilityResult {
    float shadowProbability = 0.0f;
    RayResult strongestRay;
};

struct ShadowPredictions {
    std::vector<cv::Mat> masks;
    std::vector<cv::Mat> probabilities;
};

struct EmitterSamples {
    std::array<cv::Vec3d, 7> positions{};
    int count = 1;
};

struct AgreementScore {
    double balancedError = 1.0;
    int shadowSamples = 0;
    int litSamples = 0;
};

void requireImage(const cv::Mat& image, const cv::Size& size, int type, const char* label) {
    if (image.empty() || image.size() != size || image.type() != type) {
        throw std::runtime_error(std::string(label) + " must match the height field.");
    }
}

cv::Vec3f normalized(const cv::Vec3f& value) {
    const float length = std::sqrt(value.dot(value));
    if (!std::isfinite(length) || length <= 1.0e-8f) {
        return cv::Vec3f(0.0f, 0.0f, 1.0f);
    }
    return value / length;
}

cv::Vec3d normalized(const cv::Vec3d& value) {
    const double length = std::sqrt(value.dot(value));
    if (!std::isfinite(length) || length <= 1.0e-12) {
        return cv::Vec3d(0.0, 0.0, 1.0);
    }
    return value / length;
}

cv::Vec3d nearFieldSourcePosition(
    const cv::Vec3f& reference,
    int lightIndex,
    int lightCount,
    const ShadowRefinementSettings& settings) {
    double radial = std::hypot(reference[0], reference[1]);
    double azimuthX = 0.0;
    double azimuthY = 0.0;
    if (radial > 1.0e-9) {
        azimuthX = reference[0] / radial;
        azimuthY = reference[1] / radial;
    } else {
        const double angle = 2.0 * CV_PI * static_cast<double>(lightIndex) /
            static_cast<double>(std::max(1, lightCount));
        azimuthX = std::cos(angle);
        azimuthY = std::sin(angle);
    }
    return cv::Vec3d(
        settings.ringLightRadiusMm * azimuthX,
        settings.ringLightRadiusMm * azimuthY,
        settings.ringLightHeightMm);
}

EmitterSamples nearFieldEmitterSamples(
    const cv::Vec3f& reference,
    int lightIndex,
    int lightCount,
    const ShadowRefinementSettings& settings) {
    const cv::Vec3d center = nearFieldSourcePosition(reference, lightIndex, lightCount, settings);
    EmitterSamples samples;
    samples.positions[0] = center;
    if (settings.ledDiameterMm <= 0.0) {
        return samples;
    }
    samples.count = 7;

    const cv::Vec3d target(0.0, 0.0, settings.referenceSurfaceZMm);
    const cv::Vec3d opticalAxis = normalized(target - center);
    cv::Vec3d axisU = opticalAxis.cross(cv::Vec3d(0.0, 0.0, 1.0));
    if (axisU.dot(axisU) <= 1.0e-12) {
        axisU = opticalAxis.cross(cv::Vec3d(1.0, 0.0, 0.0));
    }
    axisU = normalized(axisU);
    const cv::Vec3d axisV = normalized(opticalAxis.cross(axisU));
    const double sampleRadius = 0.5 * settings.ledDiameterMm * std::sqrt(0.5);
    for (int sample = 0; sample < 6; ++sample) {
        const double angle = 2.0 * CV_PI * static_cast<double>(sample) / 6.0;
        samples.positions[static_cast<size_t>(sample + 1)] = center + sampleRadius * (
            std::cos(angle) * axisU + std::sin(angle) * axisV);
    }
    return samples;
}

cv::Vec3f localLightDirection(
    const cv::Vec3f& reference,
    int lightIndex,
    int lightCount,
    int x,
    int y,
    float surfaceHeightPixels,
    const ShadowRefinementSettings& settings) {
    if (settings.lightingModel == LightingModel::Directional) {
        return normalized(reference);
    }

    const cv::Vec3d lightPosition = nearFieldSourcePosition(
        reference, lightIndex, lightCount, settings);
    const cv::Vec3d surfacePoint(
        (static_cast<double>(x) - settings.lightingCenter.x) * settings.pixelScaleMm,
        (settings.lightingCenter.y - static_cast<double>(y)) * settings.pixelScaleMm,
        settings.referenceSurfaceZMm +
            static_cast<double>(surfaceHeightPixels) * settings.pixelScaleMm);
    const cv::Vec3d displacement = lightPosition - surfacePoint;
    const double length = std::sqrt(displacement.dot(displacement));
    if (!std::isfinite(length) || length <= 1.0e-12) {
        return cv::Vec3f(0.0f, 0.0f, 1.0f);
    }
    return cv::Vec3f(
        static_cast<float>(displacement[0] / length),
        static_cast<float>(displacement[1] / length),
        static_cast<float>(displacement[2] / length));
}

cv::Mat buildEligibleCastRegion(
    const cv::Mat& heightMask,
    const cv::Mat& height,
    double referenceHeightPixels,
    const cv::Mat& normalMap,
    const cv::Mat& highlightMask,
    const cv::Mat& saturationMask,
    const cv::Vec3f& light,
    int lightIndex,
    int lightCount,
    const ShadowRefinementSettings& settings) {
    cv::Mat eligible(heightMask.size(), CV_8U, cv::Scalar(0));
    cv::parallel_for_(cv::Range(0, heightMask.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const uchar* maskRow = heightMask.ptr<uchar>(y);
            const cv::Vec3f* normalRow = normalMap.ptr<cv::Vec3f>(y);
            const uchar* highlightRow = highlightMask.empty() ? nullptr : highlightMask.ptr<uchar>(y);
            const uchar* saturationRow = saturationMask.empty() ? nullptr : saturationMask.ptr<uchar>(y);
            uchar* outputRow = eligible.ptr<uchar>(y);
            for (int x = 0; x < heightMask.cols; ++x) {
                if (maskRow[x] == 0 ||
                    (highlightRow != nullptr && highlightRow[x] != 0) ||
                    (saturationRow != nullptr && saturationRow[x] != 0)) {
                    continue;
                }
                const cv::Vec3f direction = localLightDirection(
                    light,
                    lightIndex,
                    lightCount,
                    x,
                    y,
                    height.at<float>(y, x) - static_cast<float>(referenceHeightPixels),
                    settings);
                if (normalRow[x].dot(direction) > kFrontLitThreshold) {
                    outputRow[x] = 255;
                }
            }
        }
    });
    return eligible;
}

cv::Mat removeTinyComponents(const cv::Mat& mask, int validPixels) {
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int count = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);
    const int minimumArea = std::clamp(validPixels / 500000, 3, 64);
    cv::Mat filtered(mask.size(), CV_8U, cv::Scalar(0));
    for (int label = 1; label < count; ++label) {
        if (stats.at<int>(label, cv::CC_STAT_AREA) >= minimumArea) {
            filtered.setTo(255, labels == label);
        }
    }
    return filtered;
}

cv::Mat regularizeObservedCastMask(
    const cv::Mat& rawShadow,
    const cv::Mat& rawConfidence,
    const cv::Mat& image,
    const cv::Mat& normalMap,
    const cv::Mat& eligible,
    cv::Mat& confidence) {
    cv::Mat rawCast;
    cv::bitwise_and(rawShadow, eligible, rawCast);
    cv::Mat labels;
    rawCast.convertTo(labels, CV_8U, 1.0 / 255.0);
    cv::Mat next = labels.clone();

    for (int iteration = 0; iteration < 3; ++iteration) {
        cv::parallel_for_(cv::Range(0, labels.rows), [&](const cv::Range& range) {
            for (int y = range.start; y < range.end; ++y) {
                const uchar* eligibleRow = eligible.ptr<uchar>(y);
                const uchar* rawRow = rawCast.ptr<uchar>(y);
                const float* imageRow = image.ptr<float>(y);
                const cv::Vec3f* normalRow = normalMap.ptr<cv::Vec3f>(y);
                uchar* outputRow = next.ptr<uchar>(y);
                for (int x = 0; x < labels.cols; ++x) {
                    if (eligibleRow[x] == 0) {
                        outputRow[x] = 0;
                        continue;
                    }
                    const float centerIntensity = imageRow[x];
                    const cv::Vec3f centerNormal = normalRow[x];
                    double shadowVote = rawRow[x] != 0 ? 2.4 : 0.0;
                    double totalWeight = 2.4;
                    for (int dy = -1; dy <= 1; ++dy) {
                        const int ny = y + dy;
                        if (ny < 0 || ny >= labels.rows) {
                            continue;
                        }
                        const uchar* neighborEligible = eligible.ptr<uchar>(ny);
                        const uchar* neighborLabels = labels.ptr<uchar>(ny);
                        const float* neighborImage = image.ptr<float>(ny);
                        const cv::Vec3f* neighborNormals = normalMap.ptr<cv::Vec3f>(ny);
                        for (int dx = -1; dx <= 1; ++dx) {
                            const int nx = x + dx;
                            if ((dx == 0 && dy == 0) || nx < 0 || nx >= labels.cols ||
                                neighborEligible[nx] == 0) {
                                continue;
                            }
                            const double intensityScale = 0.025 + 0.12 * std::max(
                                std::abs(static_cast<double>(centerIntensity)),
                                std::abs(static_cast<double>(neighborImage[nx])));
                            const double intensityWeight = std::exp(
                                -std::abs(static_cast<double>(centerIntensity - neighborImage[nx])) /
                                intensityScale);
                            const double normalDifference = std::max(
                                0.0,
                                1.0 - static_cast<double>(centerNormal.dot(neighborNormals[nx])));
                            const double normalWeight = std::exp(-normalDifference / 0.08);
                            const double spatialWeight = (dx != 0 && dy != 0) ? 0.70710678 : 1.0;
                            const double weight = spatialWeight * intensityWeight *
                                (0.25 + 0.75 * normalWeight);
                            totalWeight += weight;
                            shadowVote += weight * static_cast<double>(neighborLabels[nx] != 0);
                        }
                    }
                    const double probability = shadowVote / std::max(1.0e-12, totalWeight);
                    const double threshold = rawRow[x] != 0 ? 0.34 : 0.72;
                    outputRow[x] = probability >= threshold ? 1 : 0;
                }
            }
        });
        std::swap(labels, next);
    }

    labels.convertTo(labels, CV_8U, 255.0);
    cv::bitwise_and(labels, eligible, labels);
    labels = removeTinyComponents(labels, cv::countNonZero(eligible));

    confidence = cv::Mat(labels.size(), CV_32F, cv::Scalar(0));
    for (int y = 0; y < labels.rows; ++y) {
        const uchar* labelRow = labels.ptr<uchar>(y);
        const uchar* eligibleRow = eligible.ptr<uchar>(y);
        const float* rawConfidenceRow = rawConfidence.empty()
            ? nullptr
            : rawConfidence.ptr<float>(y);
        float* confidenceRow = confidence.ptr<float>(y);
        for (int x = 0; x < labels.cols; ++x) {
            if (eligibleRow[x] == 0) {
                continue;
            }
            int agreeing = 0;
            int neighbors = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                const int ny = y + dy;
                if (ny < 0 || ny >= labels.rows) {
                    continue;
                }
                const uchar* neighborEligible = eligible.ptr<uchar>(ny);
                const uchar* neighborLabels = labels.ptr<uchar>(ny);
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nx = x + dx;
                    if (nx < 0 || nx >= labels.cols || neighborEligible[nx] == 0) {
                        continue;
                    }
                    ++neighbors;
                    agreeing += (neighborLabels[nx] != 0) == (labelRow[x] != 0) ? 1 : 0;
                }
            }
            const float localAgreement = neighbors > 0
                ? static_cast<float>(agreeing) / static_cast<float>(neighbors)
                : 0.0f;
            const float inputConfidence = rawConfidenceRow == nullptr
                ? 1.0f
                : std::clamp(rawConfidenceRow[x], 0.0f, 1.0f);
            const float relabelPenalty =
                (rawShadow.at<uchar>(y, x) != 0) == (labelRow[x] != 0) ? 1.0f : 0.2f;
            confidenceRow[x] = inputConfidence * relabelPenalty *
                (0.45f + 0.55f * localAgreement);
        }
    }
    return labels;
}

CoarseGrid chooseCoarseGrid(const cv::Size& fullSize, int maximumSide) {
    const int longSide = std::max(fullSize.width, fullSize.height);
    const double ratio = longSide > maximumSide
        ? static_cast<double>(maximumSide) / static_cast<double>(longSide)
        : 1.0;
    CoarseGrid grid;
    grid.size = cv::Size(
        std::max(1, static_cast<int>(std::lround(fullSize.width * ratio))),
        std::max(1, static_cast<int>(std::lround(fullSize.height * ratio))));
    grid.scaleX = static_cast<double>(fullSize.width) / static_cast<double>(grid.size.width);
    grid.scaleY = static_cast<double>(fullSize.height) / static_cast<double>(grid.size.height);
    return grid;
}

cv::Mat resizeMaskConservatively(const cv::Mat& mask, const cv::Size& size, double threshold = 0.55) {
    cv::Mat values;
    mask.convertTo(values, CV_32F, 1.0 / 255.0);
    cv::resize(values, values, size, 0.0, 0.0, cv::INTER_AREA);
    cv::Mat result;
    cv::compare(values, threshold, result, cv::CMP_GE);
    return result;
}

cv::Mat resizeMaskedScalar(const cv::Mat& values, const cv::Mat& mask, const cv::Size& size) {
    cv::Mat weights;
    mask.convertTo(weights, CV_32F, 1.0 / 255.0);
    cv::Mat weighted = values.mul(weights);
    cv::resize(weighted, weighted, size, 0.0, 0.0, cv::INTER_AREA);
    cv::resize(weights, weights, size, 0.0, 0.0, cv::INTER_AREA);
    cv::Mat result(size, CV_32F, cv::Scalar(0));
    for (int y = 0; y < size.height; ++y) {
        const float* weightedRow = weighted.ptr<float>(y);
        const float* weightRow = weights.ptr<float>(y);
        float* resultRow = result.ptr<float>(y);
        for (int x = 0; x < size.width; ++x) {
            if (weightRow[x] > 1.0e-4f) {
                resultRow[x] = weightedRow[x] / weightRow[x];
            }
        }
    }
    return result;
}

cv::Mat resizeMaskedNormals(const cv::Mat& normals, const cv::Mat& mask, const cv::Size& size) {
    std::vector<cv::Mat> channels;
    cv::split(normals, channels);
    for (cv::Mat& channel : channels) {
        channel = resizeMaskedScalar(channel, mask, size);
    }
    cv::Mat result;
    cv::merge(channels, result);
    for (int y = 0; y < result.rows; ++y) {
        cv::Vec3f* row = result.ptr<cv::Vec3f>(y);
        for (int x = 0; x < result.cols; ++x) {
            row[x] = normalized(row[x]);
        }
    }
    return result;
}

float sampleMaskedHeight(
    const cv::Mat& height,
    const cv::Mat& mask,
    double x,
    double y,
    cv::Point& representative) {
    const int nearestX = std::clamp(static_cast<int>(std::lround(x)), 0, height.cols - 1);
    const int nearestY = std::clamp(static_cast<int>(std::lround(y)), 0, height.rows - 1);
    representative = cv::Point(nearestX, nearestY);
    if (mask.at<uchar>(nearestY, nearestX) == 0) {
        return std::numeric_limits<float>::quiet_NaN();
    }

    const int x0 = std::clamp(static_cast<int>(std::floor(x)), 0, height.cols - 1);
    const int y0 = std::clamp(static_cast<int>(std::floor(y)), 0, height.rows - 1);
    const int x1 = std::min(x0 + 1, height.cols - 1);
    const int y1 = std::min(y0 + 1, height.rows - 1);
    const double fx = std::clamp(x - x0, 0.0, 1.0);
    const double fy = std::clamp(y - y0, 0.0, 1.0);
    const int xs[4] = {x0, x1, x0, x1};
    const int ys[4] = {y0, y0, y1, y1};
    const double weights[4] = {
        (1.0 - fx) * (1.0 - fy),
        fx * (1.0 - fy),
        (1.0 - fx) * fy,
        fx * fy};
    double total = 0.0;
    double value = 0.0;
    for (int i = 0; i < 4; ++i) {
        if (mask.at<uchar>(ys[i], xs[i]) == 0) {
            continue;
        }
        total += weights[i];
        value += weights[i] * height.at<float>(ys[i], xs[i]);
    }
    return total >= 0.5
        ? static_cast<float>(value / total)
        : std::numeric_limits<float>::quiet_NaN();
}

RayResult traceDirectionalRay(
    const cv::Mat& height,
    const cv::Mat& mask,
    int x,
    int y,
    const cv::Vec3f& light,
    const CoarseGrid& grid) {
    const cv::Vec3f direction = normalized(light);
    const double radial = std::hypot(direction[0], direction[1]);
    if (radial <= 1.0e-8 || direction[2] <= 0.0f) {
        return {};
    }
    const double unitX = direction[0] / radial;
    const double unitY = -direction[1] / radial;
    const double coarseRate = std::max(
        std::abs(unitX) / grid.scaleX,
        std::abs(unitY) / grid.scaleY);
    if (coarseRate <= 1.0e-12) {
        return {};
    }
    const double horizontalPixelsPerStep = 1.0 / coarseRate;
    const double stepX = unitX * horizontalPixelsPerStep / grid.scaleX;
    const double stepY = unitY * horizontalPixelsPerStep / grid.scaleY;
    const double risePerStep = horizontalPixelsPerStep * direction[2] / radial;
    const float receiver = height.at<float>(y, x);
    RayResult result;
    const int maximumSteps = 2 * std::max(height.cols, height.rows) + 2;
    for (int step = 2; step < maximumSteps; ++step) {
        const double sampleX = static_cast<double>(x) + step * stepX;
        const double sampleY = static_cast<double>(y) + step * stepY;
        if (sampleX < 0.0 || sampleX > height.cols - 1.0 ||
            sampleY < 0.0 || sampleY > height.rows - 1.0) {
            break;
        }
        cv::Point blocker;
        const float terrain = sampleMaskedHeight(height, mask, sampleX, sampleY, blocker);
        if (!std::isfinite(terrain)) {
            continue;
        }
        const float clearance = terrain - static_cast<float>(receiver + step * risePerStep);
        if (clearance > result.clearance) {
            result.clearance = clearance;
            result.blocker = blocker;
        }
    }
    return result;
}

RayResult traceNearFieldRayToSource(
    const cv::Mat& height,
    const cv::Mat& mask,
    int x,
    int y,
    const cv::Vec3d& sourceMm,
    const ShadowRefinementSettings& settings,
    const CoarseGrid& grid) {
    const double receiverFullX = (static_cast<double>(x) + 0.5) * grid.scaleX - 0.5;
    const double receiverFullY = (static_cast<double>(y) + 0.5) * grid.scaleY - 0.5;
    const double lightFullX = settings.lightingCenter.x +
        sourceMm[0] / settings.pixelScaleMm;
    const double lightFullY = settings.lightingCenter.y -
        sourceMm[1] / settings.pixelScaleMm;
    const double dx = (lightFullX - receiverFullX) / grid.scaleX;
    const double dy = (lightFullY - receiverFullY) / grid.scaleY;
    const int steps = static_cast<int>(std::ceil(std::max(std::abs(dx), std::abs(dy))));
    if (steps <= 2) {
        return {};
    }

    const float receiver = height.at<float>(y, x);
    const double lightHeightPixels =
        (sourceMm[2] - settings.referenceSurfaceZMm) / settings.pixelScaleMm;
    RayResult result;
    for (int step = 2; step < steps; ++step) {
        const double t = static_cast<double>(step) / static_cast<double>(steps);
        const double sampleX = static_cast<double>(x) + t * dx;
        const double sampleY = static_cast<double>(y) + t * dy;
        if (sampleX < 0.0 || sampleX > height.cols - 1.0 ||
            sampleY < 0.0 || sampleY > height.rows - 1.0) {
            continue;
        }
        cv::Point blocker;
        const float terrain = sampleMaskedHeight(height, mask, sampleX, sampleY, blocker);
        if (!std::isfinite(terrain)) {
            continue;
        }
        const double rayHeight = receiver + t * (lightHeightPixels - receiver);
        const float clearance = terrain - static_cast<float>(rayHeight);
        if (clearance > result.clearance) {
            result.clearance = clearance;
            result.blocker = blocker;
        }
    }
    return result;
}

VisibilityResult traceVisibility(
    const cv::Mat& height,
    const cv::Mat& mask,
    int x,
    int y,
    const cv::Vec3f& light,
    int lightIndex,
    int lightCount,
    const ShadowRefinementSettings& settings,
    const CoarseGrid& grid,
    float tolerance) {
    VisibilityResult result;
    if (settings.lightingModel == LightingModel::Directional) {
        result.strongestRay = traceDirectionalRay(height, mask, x, y, light, grid);
        result.shadowProbability = result.strongestRay.blocker.x >= 0 &&
                result.strongestRay.clearance > tolerance
            ? 1.0f
            : 0.0f;
        return result;
    }

    const EmitterSamples samples = nearFieldEmitterSamples(
        light, lightIndex, lightCount, settings);
    double probability = 0.0;
    const float transition = std::max(0.04f, 0.75f * tolerance);
    for (int sample = 0; sample < samples.count; ++sample) {
        const cv::Vec3d& source = samples.positions[static_cast<size_t>(sample)];
        const RayResult ray = traceNearFieldRayToSource(
            height, mask, x, y, source, settings, grid);
        if (ray.clearance > result.strongestRay.clearance) {
            result.strongestRay = ray;
        }
        if (ray.blocker.x < 0) {
            continue;
        }
        if (settings.ledDiameterMm <= 0.0) {
            probability += ray.clearance > tolerance ? 1.0 : 0.0;
            continue;
        }
        const double normalizedClearance = std::clamp(
            (static_cast<double>(ray.clearance) - tolerance) /
                static_cast<double>(transition),
            -1.0,
            1.0);
        probability += 0.5 + 0.75 * normalizedClearance -
            0.25 * normalizedClearance * normalizedClearance * normalizedClearance;
    }
    result.shadowProbability = static_cast<float>(std::clamp(
        probability / static_cast<double>(samples.count), 0.0, 1.0));
    return result;
}

void predictCastShadow(
    const cv::Mat& height,
    const cv::Mat& receiverMask,
    const cv::Mat& occluderMask,
    const cv::Mat& eligible,
    const cv::Vec3f& light,
    int lightIndex,
    int lightCount,
    const ShadowRefinementSettings& settings,
    const CoarseGrid& grid,
    float tolerance,
    cv::Mat& predicted,
    cv::Mat& probability) {
    predicted = cv::Mat(height.size(), CV_8U, cv::Scalar(0));
    probability = cv::Mat(height.size(), CV_32F, cv::Scalar(0));
    cv::parallel_for_(cv::Range(0, height.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const uchar* receiverRow = receiverMask.ptr<uchar>(y);
            const uchar* eligibleRow = eligible.ptr<uchar>(y);
            uchar* outputRow = predicted.ptr<uchar>(y);
            float* probabilityRow = probability.ptr<float>(y);
            for (int x = 0; x < height.cols; ++x) {
                if (receiverRow[x] == 0 || eligibleRow[x] == 0) {
                    continue;
                }
                const VisibilityResult visibility = traceVisibility(
                    height,
                    occluderMask,
                    x,
                    y,
                    light,
                    lightIndex,
                    lightCount,
                    settings,
                    grid,
                    tolerance);
                probabilityRow[x] = visibility.shadowProbability;
                if (visibility.shadowProbability >= 0.5f) {
                    outputRow[x] = 255;
                }
            }
        }
    });
}

cv::Mat innerBoundary(const cv::Mat& mask) {
    cv::Mat eroded;
    cv::erode(mask, eroded, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
    cv::Mat boundary;
    cv::subtract(mask, eroded, boundary);
    return boundary;
}

cv::Mat outerBoundary(const cv::Mat& mask, const cv::Mat& eligible) {
    cv::Mat dilated;
    cv::dilate(mask, dilated, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
    cv::Mat inverse;
    cv::bitwise_not(mask, inverse);
    cv::bitwise_and(dilated, inverse, dilated);
    cv::bitwise_and(dilated, eligible, dilated);
    return dilated;
}

cv::Mat smoothMasked(const cv::Mat& values, const cv::Mat& mask, double sigma) {
    cv::Mat weights;
    mask.convertTo(weights, CV_32F, 1.0 / 255.0);
    cv::Mat weighted = values.mul(weights);
    cv::GaussianBlur(weighted, weighted, cv::Size(), sigma, sigma, cv::BORDER_REPLICATE);
    cv::GaussianBlur(weights, weights, cv::Size(), sigma, sigma, cv::BORDER_REPLICATE);
    cv::Mat result(values.size(), CV_32F, cv::Scalar(0));
    for (int y = 0; y < values.rows; ++y) {
        const float* weightedRow = weighted.ptr<float>(y);
        const float* weightRow = weights.ptr<float>(y);
        const uchar* maskRow = mask.ptr<uchar>(y);
        float* resultRow = result.ptr<float>(y);
        for (int x = 0; x < values.cols; ++x) {
            if (maskRow[x] != 0 && weightRow[x] > 1.0e-5f) {
                resultRow[x] = weightedRow[x] / weightRow[x];
            }
        }
    }
    return result;
}

double maskedMean(const cv::Mat& values, const cv::Mat& mask) {
    double sum = 0.0;
    int count = 0;
    for (int y = 0; y < values.rows; ++y) {
        const float* valueRow = values.ptr<float>(y);
        const uchar* maskRow = mask.ptr<uchar>(y);
        for (int x = 0; x < values.cols; ++x) {
            if (maskRow[x] != 0 && std::isfinite(valueRow[x])) {
                sum += valueRow[x];
                ++count;
            }
        }
    }
    return count > 0 ? sum / static_cast<double>(count) : 0.0;
}

double referencePlaneHeight(
    const cv::Mat& height,
    const cv::Mat& mask,
    const cv::Mat& normalMap) {
    std::vector<float> flatSamples;
    std::vector<float> allSamples;
    const int validPixels = cv::countNonZero(mask);
    flatSamples.reserve(static_cast<size_t>(validPixels / 2));
    allSamples.reserve(static_cast<size_t>(validPixels));
    for (int y = 0; y < height.rows; ++y) {
        const float* heightRow = height.ptr<float>(y);
        const uchar* maskRow = mask.ptr<uchar>(y);
        const cv::Vec3f* normalRow = normalMap.ptr<cv::Vec3f>(y);
        for (int x = 0; x < height.cols; ++x) {
            if (maskRow[x] == 0 || !std::isfinite(heightRow[x])) {
                continue;
            }
            allSamples.push_back(heightRow[x]);
            const cv::Vec3f normal = normalRow[x];
            if (normal[2] > 0.985f && std::abs(normal[0]) < 0.12f &&
                std::abs(normal[1]) < 0.12f) {
                flatSamples.push_back(heightRow[x]);
            }
        }
    }
    const size_t minimumFlatSamples = static_cast<size_t>(std::max(32, validPixels / 50));
    if (flatSamples.size() >= minimumFlatSamples) {
        const size_t middle = flatSamples.size() / 2;
        std::nth_element(
            flatSamples.begin(),
            flatSamples.begin() + static_cast<std::ptrdiff_t>(middle),
            flatSamples.end());
        return flatSamples[middle];
    }
    if (allSamples.empty()) {
        return 0.0;
    }
    const size_t low = allSamples.size() / 10;
    std::nth_element(
        allSamples.begin(),
        allSamples.begin() + static_cast<std::ptrdiff_t>(low),
        allSamples.end());
    return allSamples[low];
}

double percentileRange(const cv::Mat& values, const cv::Mat& mask) {
    std::vector<float> samples;
    samples.reserve(static_cast<size_t>(cv::countNonZero(mask)));
    for (int y = 0; y < values.rows; ++y) {
        const float* valueRow = values.ptr<float>(y);
        const uchar* maskRow = mask.ptr<uchar>(y);
        for (int x = 0; x < values.cols; ++x) {
            if (maskRow[x] != 0 && std::isfinite(valueRow[x])) {
                samples.push_back(valueRow[x]);
            }
        }
    }
    if (samples.size() < 4) {
        return 0.0;
    }
    const size_t lowIndex = samples.size() / 20;
    const size_t highIndex = samples.size() - 1 - lowIndex;
    std::nth_element(samples.begin(), samples.begin() + static_cast<std::ptrdiff_t>(lowIndex), samples.end());
    const float low = samples[lowIndex];
    std::nth_element(samples.begin(), samples.begin() + static_cast<std::ptrdiff_t>(highIndex), samples.end());
    return std::max(0.0, static_cast<double>(samples[highIndex] - low));
}

std::vector<int> selectRefinementLights(
    const std::vector<cv::Mat>& observedCast,
    const std::vector<cv::Mat>& eligible,
    const std::vector<cv::Vec3f>& lights,
    int& coherentShadowLights) {
    std::vector<int> allCandidates;
    std::vector<int> shadowCandidates;
    for (size_t i = 0; i < observedCast.size(); ++i) {
        if (cv::countNonZero(eligible[i]) < 16) {
            continue;
        }
        allCandidates.push_back(static_cast<int>(i));
        if (cv::countNonZero(observedCast[i]) >= 3) {
            shadowCandidates.push_back(static_cast<int>(i));
        }
    }
    coherentShadowLights = static_cast<int>(shadowCandidates.size());
    auto sortByAzimuth = [&](std::vector<int>& indices) {
        std::sort(indices.begin(), indices.end(), [&](int first, int second) {
        const double firstAngle = std::atan2(lights[static_cast<size_t>(first)][1], lights[static_cast<size_t>(first)][0]);
        const double secondAngle = std::atan2(lights[static_cast<size_t>(second)][1], lights[static_cast<size_t>(second)][0]);
        return firstAngle < secondAngle;
        });
    };
    sortByAzimuth(allCandidates);
    sortByAzimuth(shadowCandidates);
    if (static_cast<int>(allCandidates.size()) <= kMaximumRefinementLights) {
        return allCandidates;
    }

    std::vector<int> selected;
    selected.reserve(kMaximumRefinementLights);
    const int shadowSlots = std::min(
        static_cast<int>(shadowCandidates.size()),
        std::min(12, kMaximumRefinementLights));
    for (int i = 0; i < shadowSlots; ++i) {
        const size_t index = static_cast<size_t>(std::floor(
            (static_cast<double>(i) + 0.5) * shadowCandidates.size() /
            static_cast<double>(std::max(1, shadowSlots))));
        selected.push_back(shadowCandidates[std::min(index, shadowCandidates.size() - 1)]);
    }
    for (int i = 0; i < kMaximumRefinementLights &&
            static_cast<int>(selected.size()) < kMaximumRefinementLights; ++i) {
        const size_t index = static_cast<size_t>(std::floor(
            (static_cast<double>(i) + 0.5) * allCandidates.size() /
            static_cast<double>(kMaximumRefinementLights)));
        const int candidate = allCandidates[std::min(index, allCandidates.size() - 1)];
        if (std::find(selected.begin(), selected.end(), candidate) == selected.end()) {
            selected.push_back(candidate);
        }
    }
    sortByAzimuth(selected);
    return selected;
}

AgreementScore scoreAgreement(
    const ShadowPredictions& predicted,
    const std::vector<cv::Mat>& observed,
    const std::vector<cv::Mat>& confidence,
    const std::vector<cv::Mat>& eligible,
    const std::vector<int>& indices,
    cv::Mat* mismatchCount = nullptr) {
    double falseNegative = 0.0;
    double falsePositive = 0.0;
    double shadowWeight = 0.0;
    double litWeight = 0.0;
    long long shadowCount = 0;
    long long litCount = 0;
    if (mismatchCount != nullptr) {
        *mismatchCount = cv::Mat(observed.front().size(), CV_32F, cv::Scalar(0));
    }
    for (const int index : indices) {
        cv::Mat shadowEvaluation = observed[static_cast<size_t>(index)];
        cv::Mat shadowDilated;
        cv::dilate(
            shadowEvaluation,
            shadowDilated,
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
        cv::Mat litEvaluation;
        cv::bitwise_not(shadowDilated, litEvaluation);
        cv::bitwise_and(litEvaluation, eligible[static_cast<size_t>(index)], litEvaluation);
        for (int y = 0; y < shadowEvaluation.rows; ++y) {
            const float* predictedRow = predicted.probabilities[static_cast<size_t>(index)].ptr<float>(y);
            const uchar* shadowRow = shadowEvaluation.ptr<uchar>(y);
            const uchar* litRow = litEvaluation.ptr<uchar>(y);
            const float* confidenceRow = confidence.empty()
                ? nullptr
                : confidence[static_cast<size_t>(index)].ptr<float>(y);
            float* mismatchRow = mismatchCount == nullptr ? nullptr : mismatchCount->ptr<float>(y);
            for (int x = 0; x < shadowEvaluation.cols; ++x) {
                if (shadowRow[x] != 0) {
                    ++shadowCount;
                    const double weight = confidenceRow == nullptr
                        ? 1.0
                        : std::clamp(static_cast<double>(confidenceRow[x]), 0.0, 1.0);
                    shadowWeight += weight;
                    falseNegative += weight * (1.0 - predictedRow[x]);
                    if (mismatchRow != nullptr) {
                        mismatchRow[x] += static_cast<float>(weight * (1.0 - predictedRow[x]));
                    }
                } else if (litRow[x] != 0) {
                    ++litCount;
                    const double weight = confidenceRow == nullptr
                        ? 1.0
                        : std::clamp(static_cast<double>(confidenceRow[x]), 0.0, 1.0);
                    litWeight += weight;
                    falsePositive += weight * predictedRow[x];
                    if (mismatchRow != nullptr) {
                        mismatchRow[x] += static_cast<float>(weight * predictedRow[x]);
                    }
                }
            }
        }
    }
    AgreementScore score;
    score.shadowSamples = static_cast<int>(std::min<long long>(shadowCount, std::numeric_limits<int>::max()));
    score.litSamples = static_cast<int>(std::min<long long>(litCount, std::numeric_limits<int>::max()));
    if (shadowWeight > 1.0e-8 && litWeight > 1.0e-8) {
        score.balancedError = 0.5 * (
            falseNegative / shadowWeight + falsePositive / litWeight);
    }
    return score;
}

ShadowPredictions predictSelectedLights(
    const cv::Mat& height,
    const cv::Mat& receiverMask,
    const cv::Mat& occluderMask,
    const std::vector<cv::Mat>& eligible,
    const std::vector<cv::Vec3f>& lights,
    const std::vector<int>& indices,
    const ShadowRefinementSettings& settings,
    const CoarseGrid& grid,
    float tolerance) {
    ShadowPredictions predicted;
    predicted.masks.resize(lights.size());
    predicted.probabilities.resize(lights.size());
    for (const int index : indices) {
        predictCastShadow(
            height,
            receiverMask,
            occluderMask,
            eligible[static_cast<size_t>(index)],
            lights[static_cast<size_t>(index)],
            index,
            static_cast<int>(lights.size()),
            settings,
            grid,
            tolerance,
            predicted.masks[static_cast<size_t>(index)],
            predicted.probabilities[static_cast<size_t>(index)]);
    }
    return predicted;
}

std::vector<double> perLightBalancedErrors(
    const ShadowPredictions& predicted,
    const std::vector<cv::Mat>& observed,
    const std::vector<cv::Mat>& confidence,
    const std::vector<cv::Mat>& eligible,
    const std::vector<int>& indices) {
    std::vector<double> errors;
    errors.reserve(indices.size());
    for (const int index : indices) {
        errors.push_back(scoreAgreement(
            predicted,
            observed,
            confidence,
            eligible,
            std::vector<int>{index}).balancedError);
    }
    return errors;
}

void incrementMaskCount(cv::Mat& count, const cv::Mat& mask) {
    for (int y = 0; y < count.rows; ++y) {
        float* countRow = count.ptr<float>(y);
        const uchar* maskRow = mask.ptr<uchar>(y);
        for (int x = 0; x < count.cols; ++x) {
            if (maskRow[x] != 0) {
                countRow[x] += 1.0f;
            }
        }
    }
}

int azimuthSector(const cv::Vec3f& light) {
    double angle = std::atan2(light[1], light[0]);
    if (angle < 0.0) {
        angle += 2.0 * CV_PI;
    }
    return std::clamp(static_cast<int>(std::floor(2.0 * angle / CV_PI)), 0, 3);
}

void buildEvidenceDiagnostics(
    const std::vector<cv::Mat>& observed,
    const std::vector<cv::Mat>& confidence,
    const std::vector<cv::Mat>& eligible,
    const std::vector<cv::Vec3f>& lights,
    const std::vector<int>& indices,
    cv::Mat& observability,
    cv::Mat& edgeSupport) {
    const cv::Size size = observed.front().size();
    observability = cv::Mat(size, CV_32F, cv::Scalar(0));
    edgeSupport = cv::Mat(size, CV_32F, cv::Scalar(0));
    std::vector<cv::Mat> edges(observed.size());
    for (const int index : indices) {
        edges[static_cast<size_t>(index)] = innerBoundary(observed[static_cast<size_t>(index)]);
    }

    for (int y = 0; y < size.height; ++y) {
        float* observabilityRow = observability.ptr<float>(y);
        float* edgeRow = edgeSupport.ptr<float>(y);
        for (int x = 0; x < size.width; ++x) {
            double shadowEvidence = 0.0;
            double litEvidence = 0.0;
            unsigned int shadowSectors = 0;
            unsigned int litSectors = 0;
            for (const int index : indices) {
                if (eligible[static_cast<size_t>(index)].at<uchar>(y, x) == 0) {
                    continue;
                }
                const double weight = confidence.empty()
                    ? 1.0
                    : std::clamp(
                        static_cast<double>(confidence[static_cast<size_t>(index)].at<float>(y, x)),
                        0.0,
                        1.0);
                if (weight <= 0.05) {
                    continue;
                }
                const unsigned int sectorBit = 1u << azimuthSector(lights[static_cast<size_t>(index)]);
                if (observed[static_cast<size_t>(index)].at<uchar>(y, x) != 0) {
                    shadowEvidence += weight;
                    if (weight >= 0.25) {
                        shadowSectors |= sectorBit;
                    }
                } else {
                    litEvidence += weight;
                    if (weight >= 0.25) {
                        litSectors |= sectorBit;
                    }
                }
                if (!edges[static_cast<size_t>(index)].empty() &&
                    edges[static_cast<size_t>(index)].at<uchar>(y, x) != 0) {
                    edgeRow[x] += static_cast<float>(weight);
                }
            }
            const double total = shadowEvidence + litEvidence;
            if (shadowEvidence <= 0.0 || litEvidence <= 0.0 || total <= 0.0) {
                continue;
            }
            const double balance = 2.0 * std::min(shadowEvidence, litEvidence) / total;
            const int sectorCount = static_cast<int>(
                std::bitset<4>(shadowSectors | litSectors).count());
            const double directionalSupport = std::min(1.0, sectorCount / 3.0);
            const double evidenceSupport = std::min(1.0, total / 3.0);
            observabilityRow[x] = static_cast<float>(
                balance * directionalSupport * evidenceSupport);
        }
    }
}

std::vector<cv::Mat> buildCorrectionBasis(const cv::Size& size, const cv::Mat& mask) {
    std::vector<cv::Mat> basis;
    auto addBasis = [&](const std::function<double(double, double)>& function) {
        cv::Mat field(size, CV_32F, cv::Scalar(0));
        for (int y = 0; y < size.height; ++y) {
            float* row = field.ptr<float>(y);
            const uchar* maskRow = mask.ptr<uchar>(y);
            const double yn = size.height > 1
                ? 2.0 * static_cast<double>(y) / static_cast<double>(size.height - 1) - 1.0
                : 0.0;
            for (int x = 0; x < size.width; ++x) {
                if (maskRow[x] == 0) {
                    continue;
                }
                const double xn = size.width > 1
                    ? 2.0 * static_cast<double>(x) / static_cast<double>(size.width - 1) - 1.0
                    : 0.0;
                row[x] = static_cast<float>(function(xn, yn));
            }
        }
        field -= static_cast<float>(maskedMean(field, mask));
        double maximum = 0.0;
        for (int y = 0; y < field.rows; ++y) {
            const float* row = field.ptr<float>(y);
            const uchar* maskRow = mask.ptr<uchar>(y);
            for (int x = 0; x < field.cols; ++x) {
                if (maskRow[x] != 0) {
                    maximum = std::max(maximum, std::abs(static_cast<double>(row[x])));
                }
            }
        }
        if (maximum > 1.0e-8) {
            field *= static_cast<float>(1.0 / maximum);
            basis.push_back(std::move(field));
        }
    };

    addBasis([](double x, double) { return x; });
    addBasis([](double, double y) { return y; });
    addBasis([](double x, double) { return x * x; });
    addBasis([](double, double y) { return y * y; });
    addBasis([](double x, double y) { return x * y; });
    return basis;
}

double normalSlopeMeanSquaredError(
    const cv::Mat& height,
    const cv::Mat& normals,
    const cv::Mat& receiverMask,
    const CoarseGrid& grid) {
    double squaredError = 0.0;
    int samples = 0;
    for (int y = 0; y < height.rows; ++y) {
        const uchar* maskRow = receiverMask.ptr<uchar>(y);
        const cv::Vec3f* normalRow = normals.ptr<cv::Vec3f>(y);
        for (int x = 0; x < height.cols; ++x) {
            if (maskRow[x] == 0 || normalRow[x][2] <= 0.15f) {
                continue;
            }
            const int left = x > 0 && receiverMask.at<uchar>(y, x - 1) != 0 ? x - 1 : x;
            const int right = x + 1 < height.cols && receiverMask.at<uchar>(y, x + 1) != 0 ? x + 1 : x;
            const int up = y > 0 && receiverMask.at<uchar>(y - 1, x) != 0 ? y - 1 : y;
            const int down = y + 1 < height.rows && receiverMask.at<uchar>(y + 1, x) != 0 ? y + 1 : y;
            if (left != right) {
                const double actualP =
                    (height.at<float>(y, right) - height.at<float>(y, left)) /
                    (static_cast<double>(right - left) * grid.scaleX);
                const double targetP = std::clamp(
                    -static_cast<double>(normalRow[x][0]) / normalRow[x][2],
                    -3.0,
                    3.0);
                const double difference = actualP - targetP;
                squaredError += std::min(4.0, difference * difference);
                ++samples;
            }
            if (up != down) {
                const double actualQ =
                    (height.at<float>(down, x) - height.at<float>(up, x)) /
                    (static_cast<double>(down - up) * grid.scaleY);
                const double targetQ = std::clamp(
                    static_cast<double>(normalRow[x][1]) / normalRow[x][2],
                    -3.0,
                    3.0);
                const double difference = actualQ - targetQ;
                squaredError += std::min(4.0, difference * difference);
                ++samples;
            }
        }
    }
    return samples > 0 ? squaredError / static_cast<double>(samples) : 4.0;
}

double constraintObjective(
    const cv::Mat& height,
    const cv::Mat& occlusionMask,
    const cv::Mat& receiverMask,
    const cv::Mat& normals,
    const std::vector<cv::Mat>& shadowBoundaries,
    const std::vector<cv::Mat>& litBoundaries,
    const std::vector<cv::Mat>& evidenceConfidence,
    const std::vector<cv::Vec3f>& lights,
    const std::vector<int>& fitIndices,
    const ShadowRefinementSettings& settings,
    const CoarseGrid& grid,
    float visibilityTolerance,
    float targetMargin,
    float coarseSpacing,
    const cv::Mat& correction,
    float maximumCorrection) {
    double shadowLoss = 0.0;
    double litLoss = 0.0;
    double shadowWeight = 0.0;
    double litWeight = 0.0;
    const double normalization = 1.0 / std::max(0.25f, coarseSpacing);
    const double maximumNormalizedViolation = 3.0;
    const float shadowTarget = visibilityTolerance + targetMargin;
    const float litTarget = visibilityTolerance - targetMargin;
    for (const int index : fitIndices) {
        const cv::Mat& shadowBoundary = shadowBoundaries[static_cast<size_t>(index)];
        const cv::Mat& litBoundary = litBoundaries[static_cast<size_t>(index)];
        for (int y = 0; y < height.rows; ++y) {
            const uchar* shadowRow = shadowBoundary.ptr<uchar>(y);
            const uchar* litRow = litBoundary.ptr<uchar>(y);
            const float* confidenceRow = evidenceConfidence.empty()
                ? nullptr
                : evidenceConfidence[static_cast<size_t>(index)].ptr<float>(y);
            for (int x = 0; x < height.cols; ++x) {
                if (shadowRow[x] == 0 && litRow[x] == 0) {
                    continue;
                }
                const VisibilityResult visibility = traceVisibility(
                    height,
                    occlusionMask,
                    x,
                    y,
                    lights[static_cast<size_t>(index)],
                    index,
                    static_cast<int>(lights.size()),
                    settings,
                    grid,
                    visibilityTolerance);
                const RayResult& ray = visibility.strongestRay;
                if (ray.blocker.x < 0) {
                    continue;
                }
                const double weight = confidenceRow == nullptr
                    ? 1.0
                    : std::clamp(static_cast<double>(confidenceRow[x]), 0.0, 1.0);
                if (weight <= 1.0e-6) {
                    continue;
                }
                if (settings.lightingModel == LightingModel::NearFieldRing &&
                    settings.ledDiameterMm > 0.0) {
                    if (shadowRow[x] != 0) {
                        const double violation = std::max(
                            0.0, 0.65 - static_cast<double>(visibility.shadowProbability));
                        shadowLoss += weight * violation * violation;
                        shadowWeight += weight;
                    } else {
                        const double violation = std::max(
                            0.0, static_cast<double>(visibility.shadowProbability) - 0.35);
                        litLoss += weight * violation * violation;
                        litWeight += weight;
                    }
                    continue;
                }
                if (shadowRow[x] != 0) {
                    const double violation = std::max(
                        0.0,
                        static_cast<double>(shadowTarget - ray.clearance) * normalization);
                    shadowLoss += weight * std::min(
                        maximumNormalizedViolation * maximumNormalizedViolation,
                        violation * violation);
                    shadowWeight += weight;
                } else {
                    const double violation = std::max(
                        0.0,
                        static_cast<double>(ray.clearance - litTarget) * normalization);
                    litLoss += weight * std::min(
                        maximumNormalizedViolation * maximumNormalizedViolation,
                        violation * violation);
                    litWeight += weight;
                }
            }
        }
    }
    if (shadowWeight <= 1.0e-8 || litWeight <= 1.0e-8) {
        return std::numeric_limits<double>::infinity();
    }
    const double data = 0.5 * (
        shadowLoss / shadowWeight + litLoss / litWeight);
    const double correctionEnergy = cv::sum(correction.mul(correction))[0] /
        static_cast<double>(std::max(1, cv::countNonZero(receiverMask)));
    const double regularization = kCorrectionPriorWeight * correctionEnergy /
        std::max(1.0e-8, static_cast<double>(maximumCorrection * maximumCorrection));
    return data + regularization + kNormalSlopePriorWeight *
        normalSlopeMeanSquaredError(height, normals, receiverMask, grid);
}

cv::Mat resizeFloatToFull(const cv::Mat& values, const cv::Size& fullSize) {
    cv::Mat result;
    cv::resize(values, result, fullSize, 0.0, 0.0, cv::INTER_LINEAR);
    return result;
}

cv::Mat resizeMaskToFull(const cv::Mat& mask, const cv::Size& fullSize, const cv::Mat& fullMask) {
    cv::Mat result;
    cv::resize(mask, result, fullSize, 0.0, 0.0, cv::INTER_NEAREST);
    cv::bitwise_and(result, fullMask, result);
    return result;
}

} // namespace

void refineHeightFromCastShadows(
    cv::Mat& height,
    const cv::Mat& receiverMask,
    const cv::Mat& occluderMask,
    const cv::Mat& normalMap,
    const std::vector<cv::Mat>& images,
    const std::vector<cv::Vec3f>& lights,
    const ShadowRefinementSettings& settings,
    PhotometricDiagnostics& diagnostics,
    const std::function<void(const std::string&)>& progress) {
    if (height.empty() || height.type() != CV_32F) {
        throw std::runtime_error("Shadow height refinement requires a single-channel float height field.");
    }
    requireImage(receiverMask, height.size(), CV_8U, "Shadow refinement receiver mask");
    requireImage(occluderMask, height.size(), CV_8U, "Shadow refinement occluder mask");
    requireImage(normalMap, height.size(), CV_32FC3, "Shadow refinement normal map");
    if (images.size() != lights.size() || images.size() < 6 ||
        diagnostics.shadowObservationMasks.size() != lights.size()) {
        throw std::runtime_error(
            "Shadow height refinement requires at least six calibrated images and per-light robust shadow masks.");
    }
    if (!diagnostics.shadowObservationConfidence.empty() &&
        diagnostics.shadowObservationConfidence.size() != lights.size()) {
        throw std::runtime_error(
            "Shadow observation confidence must be empty or contain one map per light.");
    }
    if (settings.maximumCoarseSide < 32 || settings.iterations < 1) {
        throw std::runtime_error("Shadow height refinement settings are invalid.");
    }
    if (settings.lightingModel == LightingModel::NearFieldRing &&
        (!std::isfinite(settings.pixelScaleMm) || settings.pixelScaleMm <= 0.0 ||
         !std::isfinite(settings.ringLightRadiusMm) || settings.ringLightRadiusMm <= 0.0 ||
         !std::isfinite(settings.ringLightHeightMm) || settings.ringLightHeightMm <= 0.0 ||
         !std::isfinite(settings.referenceSurfaceZMm) ||
         settings.referenceSurfaceZMm >= settings.ringLightHeightMm ||
         !std::isfinite(settings.ledDiameterMm) || settings.ledDiameterMm < 0.0)) {
        throw std::runtime_error(
            "Near-field shadow refinement requires positive ring geometry and pixel scale, "
            "a finite reference Z below the lights, and a non-negative LED diameter.");
    }
    for (const cv::Mat& image : images) {
        requireImage(image, height.size(), CV_32F, "Shadow refinement source image");
    }

    diagnostics.shadowHeightRefinementApplied = false;
    diagnostics.shadowHeightRefinementDecision = "running";
    diagnostics.shadowMismatchRateBefore = -1.0;
    diagnostics.shadowMismatchRateAfter = -1.0;
    diagnostics.shadowHoldoutMismatchRateBefore = -1.0;
    diagnostics.shadowHoldoutMismatchRateAfter = -1.0;
    diagnostics.shadowNormalSlopeRmsBefore = -1.0;
    diagnostics.shadowNormalSlopeRmsAfter = -1.0;
    diagnostics.shadowCorrectionRms = 0.0;
    diagnostics.shadowRefinementConstraintCount = 0;
    diagnostics.shadowRefinementShadowSamples = 0;
    diagnostics.shadowRefinementLitSamples = 0;
    diagnostics.shadowHeightCorrection = cv::Mat(height.size(), CV_32F, cv::Scalar(0));
    diagnostics.shadowConstraintCount = cv::Mat(height.size(), CV_32F, cv::Scalar(0));
    diagnostics.shadowMismatchBefore = cv::Mat(height.size(), CV_32F, cv::Scalar(0));
    diagnostics.shadowMismatchAfter = cv::Mat(height.size(), CV_32F, cv::Scalar(0));
    diagnostics.shadowObservability = cv::Mat(height.size(), CV_32F, cv::Scalar(0));
    diagnostics.shadowEdgeSupport = cv::Mat(height.size(), CV_32F, cv::Scalar(0));
    diagnostics.shadowOccluderSupport = occluderMask.clone();
    diagnostics.shadowSelectedStepFraction = 0.0;
    diagnostics.shadowWorstHoldoutDelta = 0.0;
    diagnostics.shadowObservedCastMasks.assign(lights.size(), cv::Mat());
    diagnostics.shadowPredictedBeforeMasks.assign(lights.size(), cv::Mat());
    diagnostics.shadowPredictedAfterMasks.assign(lights.size(), cv::Mat());
    diagnostics.shadowPredictedBeforeProbability.assign(lights.size(), cv::Mat());
    diagnostics.shadowPredictedAfterProbability.assign(lights.size(), cv::Mat());
    diagnostics.shadowRefinementLightIndices.clear();

    if (progress) {
        progress("regularizing per-light cast-shadow evidence");
    }
    const double referenceHeightPixels = referencePlaneHeight(height, receiverMask, normalMap);
    std::vector<cv::Mat> fullEligible(lights.size());
    std::vector<cv::Mat> fullObserved(lights.size());
    std::vector<cv::Mat> fullConfidence(lights.size());
    for (size_t i = 0; i < lights.size(); ++i) {
        const cv::Mat highlight = diagnostics.highlightObservationMasks.size() == lights.size()
            ? diagnostics.highlightObservationMasks[i]
            : cv::Mat();
        const cv::Mat saturation = diagnostics.saturationObservationMasks.size() == lights.size()
            ? diagnostics.saturationObservationMasks[i]
            : cv::Mat();
        fullEligible[i] = buildEligibleCastRegion(
            receiverMask,
            height,
            referenceHeightPixels,
            normalMap,
            highlight,
            saturation,
            lights[i],
            static_cast<int>(i),
            static_cast<int>(lights.size()),
            settings);
        const cv::Mat rawConfidence = diagnostics.shadowObservationConfidence.empty()
            ? cv::Mat()
            : diagnostics.shadowObservationConfidence[i];
        if (!rawConfidence.empty()) {
            requireImage(rawConfidence, height.size(), CV_32F, "Shadow observation confidence");
        }
        fullObserved[i] = regularizeObservedCastMask(
            diagnostics.shadowObservationMasks[i],
            rawConfidence,
            images[i],
            normalMap,
            fullEligible[i],
            fullConfidence[i]);
    }
    diagnostics.shadowObservedCastMasks = fullObserved;

    const CoarseGrid grid = chooseCoarseGrid(height.size(), settings.maximumCoarseSide);
    cv::Mat coarseMask = resizeMaskConservatively(receiverMask, grid.size, 0.70);
    if (cv::countNonZero(coarseMask) < 64) {
        throw std::runtime_error("Shadow height refinement has too little valid surface area.");
    }
    cv::Mat coarseHeight = resizeMaskedScalar(height, receiverMask, grid.size);
    const cv::Mat coarseNormals = resizeMaskedNormals(normalMap, receiverMask, grid.size);
    coarseHeight -= static_cast<float>(referenceHeightPixels);
    cv::Mat supportedOccluderMask = occluderMask.clone();
    cv::Mat occlusionMask = resizeMaskConservatively(supportedOccluderMask, grid.size, 0.55);
    cv::Mat occlusionBaseHeight = resizeMaskedScalar(height, occluderMask, grid.size);
    occlusionBaseHeight -= static_cast<float>(referenceHeightPixels);
    occlusionBaseHeight.setTo(0.0f, occlusionMask == 0);
    diagnostics.shadowOccluderSupport = resizeMaskToFull(
        occlusionMask, height.size(), supportedOccluderMask);
    ShadowRefinementSettings coarseSettings = settings;

    std::vector<cv::Mat> coarseEligible(lights.size());
    std::vector<cv::Mat> coarseObserved(lights.size());
    std::vector<cv::Mat> coarseConfidence(lights.size());
    for (size_t i = 0; i < lights.size(); ++i) {
        coarseEligible[i] = resizeMaskConservatively(fullEligible[i], grid.size, 0.60);
        cv::bitwise_and(coarseEligible[i], coarseMask, coarseEligible[i]);
        coarseObserved[i] = resizeMaskConservatively(fullObserved[i], grid.size, 0.42);
        cv::bitwise_and(coarseObserved[i], coarseEligible[i], coarseObserved[i]);
        cv::resize(fullConfidence[i], coarseConfidence[i], grid.size, 0.0, 0.0, cv::INTER_AREA);
        coarseConfidence[i].setTo(0.0f, coarseEligible[i] == 0);
    }

    int coherentShadowLights = 0;
    const std::vector<int> selected = selectRefinementLights(
        coarseObserved, coarseEligible, lights, coherentShadowLights);
    diagnostics.shadowRefinementLightIndices = selected;
    if (!selected.empty()) {
        cv::Mat observabilityCoarse;
        cv::Mat edgeSupportCoarse;
        buildEvidenceDiagnostics(
            coarseObserved,
            coarseConfidence,
            coarseEligible,
            lights,
            selected,
            observabilityCoarse,
            edgeSupportCoarse);
        diagnostics.shadowObservability = resizeFloatToFull(observabilityCoarse, height.size());
        diagnostics.shadowEdgeSupport = resizeFloatToFull(edgeSupportCoarse, height.size());
    }
    if (selected.size() < 6 || coherentShadowLights < 6) {
        diagnostics.shadowHeightRefinementDecision =
            "rejected_insufficient_coherent_shadow_lights";
        if (progress) {
            progress("fewer than six lights have coherent cast-shadow regions; height left unchanged");
        }
        return;
    }

    std::vector<int> fitIndices;
    std::vector<int> holdoutIndices;
    for (size_t i = 0; i < selected.size(); ++i) {
        if (selected.size() >= 6 && i % 4 == 3) {
            holdoutIndices.push_back(selected[i]);
        } else {
            fitIndices.push_back(selected[i]);
        }
    }
    if (fitIndices.size() < 4) {
        fitIndices = selected;
        holdoutIndices.clear();
    }

    const float coarseSpacing = static_cast<float>(0.5 * (grid.scaleX + grid.scaleY));
    const float visibilityTolerance = std::max(0.10f, 0.12f * coarseSpacing);
    const float targetMargin = std::max(0.08f, 0.08f * coarseSpacing);
    if (progress) {
        progress("ray-casting the integrated surface toward calibrated lights");
    }
    const ShadowPredictions predictedBefore = predictSelectedLights(
        occlusionBaseHeight,
        coarseMask,
        occlusionMask,
        coarseEligible,
        lights,
        selected,
        coarseSettings,
        grid,
        visibilityTolerance);
    cv::Mat mismatchBeforeCoarse;
    const AgreementScore overallBefore = scoreAgreement(
        predictedBefore,
        coarseObserved,
        coarseConfidence,
        coarseEligible,
        selected,
        &mismatchBeforeCoarse);
    const AgreementScore fitBefore = scoreAgreement(
        predictedBefore, coarseObserved, coarseConfidence, coarseEligible, fitIndices);
    const AgreementScore holdoutBefore = holdoutIndices.empty()
        ? AgreementScore{}
        : scoreAgreement(
            predictedBefore, coarseObserved, coarseConfidence, coarseEligible, holdoutIndices);
    diagnostics.shadowMismatchRateBefore = overallBefore.balancedError;
    diagnostics.shadowMismatchRateAfter = overallBefore.balancedError;
    diagnostics.shadowHoldoutMismatchRateBefore = holdoutIndices.empty()
        ? -1.0
        : holdoutBefore.balancedError;
    diagnostics.shadowHoldoutMismatchRateAfter = diagnostics.shadowHoldoutMismatchRateBefore;
    diagnostics.shadowRefinementShadowSamples = overallBefore.shadowSamples;
    diagnostics.shadowRefinementLitSamples = overallBefore.litSamples;
    diagnostics.shadowNormalSlopeRmsBefore = std::sqrt(normalSlopeMeanSquaredError(
        coarseHeight, coarseNormals, coarseMask, grid));
    diagnostics.shadowNormalSlopeRmsAfter = diagnostics.shadowNormalSlopeRmsBefore;
    diagnostics.shadowMismatchBefore = resizeFloatToFull(mismatchBeforeCoarse, height.size());
    diagnostics.shadowMismatchAfter = diagnostics.shadowMismatchBefore.clone();
    for (const int index : selected) {
        diagnostics.shadowPredictedBeforeMasks[static_cast<size_t>(index)] = resizeMaskToFull(
            predictedBefore.masks[static_cast<size_t>(index)], height.size(), receiverMask);
        diagnostics.shadowPredictedAfterMasks[static_cast<size_t>(index)] =
            diagnostics.shadowPredictedBeforeMasks[static_cast<size_t>(index)].clone();
        diagnostics.shadowPredictedBeforeProbability[static_cast<size_t>(index)] =
            resizeFloatToFull(
                predictedBefore.probabilities[static_cast<size_t>(index)], height.size());
        diagnostics.shadowPredictedBeforeProbability[static_cast<size_t>(index)].setTo(
            0.0f, receiverMask == 0);
        diagnostics.shadowPredictedAfterProbability[static_cast<size_t>(index)] =
            diagnostics.shadowPredictedBeforeProbability[static_cast<size_t>(index)].clone();
    }

    std::vector<cv::Mat> shadowBoundaries(lights.size());
    std::vector<cv::Mat> litBoundaries(lights.size());
    cv::Mat constraintCountCoarse(coarseHeight.size(), CV_32F, cv::Scalar(0));
    int totalShadowConstraints = 0;
    int totalLitConstraints = 0;
    for (const int index : fitIndices) {
        shadowBoundaries[static_cast<size_t>(index)] = innerBoundary(coarseObserved[static_cast<size_t>(index)]);
        litBoundaries[static_cast<size_t>(index)] = outerBoundary(
            coarseObserved[static_cast<size_t>(index)],
            coarseEligible[static_cast<size_t>(index)]);
        totalShadowConstraints += cv::countNonZero(shadowBoundaries[static_cast<size_t>(index)]);
        totalLitConstraints += cv::countNonZero(litBoundaries[static_cast<size_t>(index)]);
        incrementMaskCount(constraintCountCoarse, shadowBoundaries[static_cast<size_t>(index)]);
        incrementMaskCount(constraintCountCoarse, litBoundaries[static_cast<size_t>(index)]);
    }
    diagnostics.shadowRefinementConstraintCount = totalShadowConstraints + totalLitConstraints;
    diagnostics.shadowConstraintCount = resizeFloatToFull(constraintCountCoarse, height.size());
    if (totalShadowConstraints < 8 || totalLitConstraints < 8) {
        diagnostics.shadowHeightRefinementDecision =
            "rejected_insufficient_shadow_edge_constraints";
        if (progress) {
            progress("too few reliable shadow edges; height left unchanged");
        }
        return;
    }

    cv::Mat correction(coarseHeight.size(), CV_32F, cv::Scalar(0));
    cv::Mat workingHeight = occlusionBaseHeight.clone();
    const double heightRange = percentileRange(coarseHeight, coarseMask);
    const float maximumCorrection = static_cast<float>(std::clamp(
        0.35 * heightRange + 1.5 * coarseSpacing,
        1.5 * static_cast<double>(coarseSpacing),
        18.0 * static_cast<double>(coarseSpacing)));
    const std::vector<cv::Mat> basis = buildCorrectionBasis(coarseHeight.size(), coarseMask);
    float coordinateStep = static_cast<float>(std::clamp(
        0.12 * heightRange + 0.35 * coarseSpacing,
        0.45 * static_cast<double>(coarseSpacing),
        2.5 * static_cast<double>(coarseSpacing)));
    double objective = constraintObjective(
        workingHeight,
        occlusionMask,
        coarseMask,
        coarseNormals,
        shadowBoundaries,
        litBoundaries,
        coarseConfidence,
        lights,
        fitIndices,
        coarseSettings,
        grid,
        visibilityTolerance,
        targetMargin,
        coarseSpacing,
        correction,
        maximumCorrection);
    for (int iteration = 0; iteration < settings.iterations; ++iteration) {
        const double startObjective = objective;
        for (const cv::Mat& mode : basis) {
            cv::Mat bestCorrection = correction;
            double bestObjective = objective;
            for (const float sign : {-1.0f, 1.0f}) {
                cv::Mat candidate = correction + sign * coordinateStep * mode;
                candidate -= static_cast<float>(maskedMean(candidate, coarseMask));
                cv::max(candidate, -maximumCorrection, candidate);
                cv::min(candidate, maximumCorrection, candidate);
                candidate.setTo(0.0f, coarseMask == 0);
                const cv::Mat candidateHeight = occlusionBaseHeight + candidate;
                const double candidateObjective = constraintObjective(
                    candidateHeight,
                    occlusionMask,
                    coarseMask,
                    coarseNormals,
                    shadowBoundaries,
                    litBoundaries,
                    coarseConfidence,
                    lights,
                    fitIndices,
                    coarseSettings,
                    grid,
                    visibilityTolerance,
                    targetMargin,
                    coarseSpacing,
                    candidate,
                    maximumCorrection);
                if (candidateObjective + 1.0e-10 < bestObjective) {
                    bestObjective = candidateObjective;
                    bestCorrection = std::move(candidate);
                }
            }
            correction = bestCorrection;
            objective = bestObjective;
        }
        workingHeight = occlusionBaseHeight + correction;
        if (progress) {
            progress(
                "coarse shadow-basis pass " + std::to_string(iteration + 1) + "/" +
                std::to_string(settings.iterations) + ", objective " +
                std::to_string(startObjective) + " -> " + std::to_string(objective));
        }
        coordinateStep *= 0.62f;
    }

    correction = smoothMasked(correction, coarseMask, 0.65);
    correction -= static_cast<float>(maskedMean(correction, coarseMask));
    cv::Mat proposedFullCorrection = resizeFloatToFull(correction, height.size());
    const double fullSigma = std::max(1.0, 0.85 * std::max(grid.scaleX, grid.scaleY));
    proposedFullCorrection = smoothMasked(proposedFullCorrection, receiverMask, fullSigma);
    proposedFullCorrection -= static_cast<float>(maskedMean(proposedFullCorrection, receiverMask));
    proposedFullCorrection.setTo(0.0f, receiverMask == 0);

    if (progress) {
        progress("validating exact exported corrections against each held-out light");
    }
    const double slopeMeanSquaredBefore = normalSlopeMeanSquaredError(
        coarseHeight, coarseNormals, coarseMask, grid);
    const std::vector<double> holdoutPerLightBefore = perLightBalancedErrors(
        predictedBefore,
        coarseObserved,
        coarseConfidence,
        coarseEligible,
        holdoutIndices);

    bool accepted = false;
    bool bestFitImproved = false;
    bool bestHoldoutAccepted = holdoutIndices.empty();
    bool bestOverallImproved = false;
    bool bestNormalPriorAccepted = false;
    double bestAttemptError = std::numeric_limits<double>::infinity();
    double selectedSlopeMeanSquared = slopeMeanSquaredBefore;
    double selectedWorstHoldoutDelta = 0.0;
    AgreementScore selectedOverall = overallBefore;
    AgreementScore selectedHoldout = holdoutBefore;
    ShadowPredictions selectedPredictions = predictedBefore;
    cv::Mat selectedMismatch = mismatchBeforeCoarse.clone();
    cv::Mat fullCorrection(height.size(), CV_32F, cv::Scalar(0));

    for (const float fraction : {0.25f, 0.50f, 0.75f, 1.00f}) {
        cv::Mat candidateCorrection = proposedFullCorrection * fraction;
        const cv::Mat candidateFullHeight = height + candidateCorrection;
        cv::Mat candidateCoarseHeight = resizeMaskedScalar(
            candidateFullHeight, receiverMask, grid.size);
        candidateCoarseHeight -= static_cast<float>(
            referencePlaneHeight(candidateFullHeight, receiverMask, normalMap));
        cv::Mat candidateOcclusionHeight = resizeMaskedScalar(
            candidateFullHeight, occluderMask, grid.size);
        candidateOcclusionHeight -= static_cast<float>(
            referencePlaneHeight(candidateFullHeight, receiverMask, normalMap));
        candidateOcclusionHeight.setTo(0.0f, occlusionMask == 0);

        const ShadowPredictions candidatePredictions = predictSelectedLights(
            candidateOcclusionHeight,
            coarseMask,
            occlusionMask,
            coarseEligible,
            lights,
            selected,
            coarseSettings,
            grid,
            visibilityTolerance);
        cv::Mat candidateMismatch;
        const AgreementScore overallAfter = scoreAgreement(
            candidatePredictions,
            coarseObserved,
            coarseConfidence,
            coarseEligible,
            selected,
            &candidateMismatch);
        const AgreementScore fitAfter = scoreAgreement(
            candidatePredictions,
            coarseObserved,
            coarseConfidence,
            coarseEligible,
            fitIndices);
        const AgreementScore holdoutAfter = holdoutIndices.empty()
            ? AgreementScore{}
            : scoreAgreement(
                candidatePredictions,
                coarseObserved,
                coarseConfidence,
                coarseEligible,
                holdoutIndices);
        const std::vector<double> holdoutPerLightAfter = perLightBalancedErrors(
            candidatePredictions,
            coarseObserved,
            coarseConfidence,
            coarseEligible,
            holdoutIndices);
        double worstHoldoutDelta = 0.0;
        for (size_t i = 0; i < holdoutPerLightAfter.size(); ++i) {
            worstHoldoutDelta = std::max(
                worstHoldoutDelta,
                holdoutPerLightAfter[i] - holdoutPerLightBefore[i]);
        }
        const double slopeMeanSquaredAfter = normalSlopeMeanSquaredError(
            candidateCoarseHeight, coarseNormals, coarseMask, grid);
        const bool fitImproved = fitAfter.balancedError + 0.002 < fitBefore.balancedError;
        const bool holdoutAccepted = holdoutIndices.empty() ||
            (holdoutAfter.balancedError <= holdoutBefore.balancedError + 0.005 &&
             worstHoldoutDelta <= 0.01);
        const bool overallImproved =
            overallAfter.balancedError + 0.001 < overallBefore.balancedError;
        const bool normalPriorAccepted = std::sqrt(slopeMeanSquaredAfter) <=
            std::sqrt(slopeMeanSquaredBefore) + 1.0e-5;
        const bool candidateAccepted =
            fitImproved && holdoutAccepted && overallImproved && normalPriorAccepted;

        if (overallAfter.balancedError < bestAttemptError) {
            bestAttemptError = overallAfter.balancedError;
            bestFitImproved = fitImproved;
            bestHoldoutAccepted = holdoutAccepted;
            bestOverallImproved = overallImproved;
            bestNormalPriorAccepted = normalPriorAccepted;
        }
        if (progress) {
            progress(
                "step " + std::to_string(fraction) + ": fit " +
                std::to_string(fitBefore.balancedError) + " -> " +
                std::to_string(fitAfter.balancedError) + ", holdout " +
                (holdoutIndices.empty()
                    ? std::string("not available")
                    : std::to_string(holdoutBefore.balancedError) + " -> " +
                        std::to_string(holdoutAfter.balancedError)) +
                ", worst held-out delta " + std::to_string(worstHoldoutDelta) +
                ", all " + std::to_string(overallBefore.balancedError) + " -> " +
                std::to_string(overallAfter.balancedError));
        }
        if (candidateAccepted && !accepted) {
            accepted = true;
            selectedSlopeMeanSquared = slopeMeanSquaredAfter;
            selectedWorstHoldoutDelta = worstHoldoutDelta;
            selectedOverall = overallAfter;
            selectedHoldout = holdoutAfter;
            selectedPredictions = candidatePredictions;
            selectedMismatch = candidateMismatch;
            fullCorrection = candidateCorrection;
            diagnostics.shadowSelectedStepFraction = fraction;
        }
    }

    if (accepted) {
        diagnostics.shadowHeightRefinementDecision = "accepted";
    } else if (!bestFitImproved) {
        diagnostics.shadowHeightRefinementDecision = "rejected_fit_did_not_improve";
    } else if (!bestHoldoutAccepted) {
        diagnostics.shadowHeightRefinementDecision = "rejected_withheld_lights_worsened";
    } else if (!bestOverallImproved) {
        diagnostics.shadowHeightRefinementDecision = "rejected_overall_agreement_did_not_improve";
    } else if (!bestNormalPriorAccepted) {
        diagnostics.shadowHeightRefinementDecision = "rejected_normal_slope_inconsistency";
    } else {
        diagnostics.shadowHeightRefinementDecision = "rejected_validation_gate";
    }
    if (!accepted) {
        correction.setTo(0.0f);
        fullCorrection.setTo(0.0f);
        selectedMismatch = mismatchBeforeCoarse.clone();
        selectedPredictions = predictedBefore;
        selectedOverall = overallBefore;
        selectedHoldout = holdoutBefore;
        selectedSlopeMeanSquared = slopeMeanSquaredBefore;
        selectedWorstHoldoutDelta = 0.0;
    }

    diagnostics.shadowHeightRefinementApplied = accepted;
    diagnostics.shadowMismatchRateBefore = overallBefore.balancedError;
    diagnostics.shadowMismatchRateAfter = selectedOverall.balancedError;
    diagnostics.shadowHoldoutMismatchRateBefore = holdoutIndices.empty()
        ? -1.0
        : holdoutBefore.balancedError;
    diagnostics.shadowHoldoutMismatchRateAfter = holdoutIndices.empty()
        ? -1.0
        : (accepted ? selectedHoldout.balancedError : holdoutBefore.balancedError);
    diagnostics.shadowNormalSlopeRmsBefore = std::sqrt(slopeMeanSquaredBefore);
    diagnostics.shadowNormalSlopeRmsAfter = accepted
        ? std::sqrt(selectedSlopeMeanSquared)
        : diagnostics.shadowNormalSlopeRmsBefore;
    diagnostics.shadowWorstHoldoutDelta = selectedWorstHoldoutDelta;
    diagnostics.shadowRefinementShadowSamples = overallBefore.shadowSamples;
    diagnostics.shadowRefinementLitSamples = overallBefore.litSamples;

    diagnostics.shadowHeightCorrection = fullCorrection;
    diagnostics.shadowCorrectionRms = std::sqrt(
        cv::sum(fullCorrection.mul(fullCorrection))[0] /
        static_cast<double>(std::max(1, cv::countNonZero(receiverMask))));
    if (accepted) {
        height += fullCorrection;
    }

    diagnostics.shadowMismatchBefore = resizeFloatToFull(mismatchBeforeCoarse, height.size());
    diagnostics.shadowMismatchAfter = resizeFloatToFull(selectedMismatch, height.size());
    for (const int index : selected) {
        diagnostics.shadowPredictedBeforeMasks[static_cast<size_t>(index)] = resizeMaskToFull(
            predictedBefore.masks[static_cast<size_t>(index)], height.size(), receiverMask);
        diagnostics.shadowPredictedAfterMasks[static_cast<size_t>(index)] = resizeMaskToFull(
            selectedPredictions.masks[static_cast<size_t>(index)], height.size(), receiverMask);
        diagnostics.shadowPredictedAfterProbability[static_cast<size_t>(index)] =
            resizeFloatToFull(
                selectedPredictions.probabilities[static_cast<size_t>(index)], height.size());
        diagnostics.shadowPredictedAfterProbability[static_cast<size_t>(index)].setTo(
            0.0f, receiverMask == 0);
    }

    if (progress) {
        progress(accepted
            ? "accepted shadow-constrained broad-height correction"
            : "shadow correction failed the fit/holdout safety gate; height left unchanged");
    }
}
