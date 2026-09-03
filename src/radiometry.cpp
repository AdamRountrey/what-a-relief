#include "radiometry.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

double integerScaleForDepth(int depth) {
    switch (depth) {
    case CV_8U:
        return 1.0 / 255.0;
    case CV_8S:
        return 1.0 / 127.0;
    case CV_16U:
        return 1.0 / 65535.0;
    case CV_16S:
        return 1.0 / 32767.0;
    default:
        return 1.0;
    }
}

float srgbToLinear(float encoded) {
    const float value = std::clamp(encoded, 0.0f, 1.0f);
    if (value <= 0.04045f) {
        return value / 12.92f;
    }
    return std::pow((value + 0.055f) / 1.055f, 2.4f);
}

void linearizeSrgbChannel(cv::Mat& channel) {
    for (int y = 0; y < channel.rows; ++y) {
        float* row = channel.ptr<float>(y);
        for (int x = 0; x < channel.cols; ++x) {
            row[x] = srgbToLinear(row[x]);
        }
    }
}

} // namespace

cv::Mat convertToLinearColor(const cv::Mat& input, bool srgb) {
    if (input.empty()) {
        throw std::runtime_error("Cannot convert an empty image to linear color.");
    }

    cv::Mat color;
    if (input.channels() == 1) {
        cv::cvtColor(input, color, cv::COLOR_GRAY2BGR);
    } else if (input.channels() == 3) {
        color = input;
    } else if (input.channels() == 4) {
        cv::cvtColor(input, color, cv::COLOR_BGRA2BGR);
    } else {
        throw std::runtime_error("Linear color conversion requires one, three, or four image channels.");
    }

    cv::Mat f32;
    color.convertTo(
        f32,
        CV_32FC3,
        integerScaleForDepth(color.depth()));

    std::vector<cv::Mat> channels;
    cv::split(f32, channels);
    if (srgb) {
        for (cv::Mat& channel : channels) {
            linearizeSrgbChannel(channel);
        }
    }
    cv::merge(channels, f32);
    return f32;
}

cv::Mat convertToLinearLuminance(const cv::Mat& input, bool srgb) {
    if (input.empty()) {
        throw std::runtime_error("Cannot convert an empty image to luminance.");
    }

    if (input.channels() == 1) {
        cv::Mat luminance;
        input.convertTo(luminance, CV_32F, integerScaleForDepth(input.depth()));
        if (srgb) {
            linearizeSrgbChannel(luminance);
        }
        cv::max(luminance, 0.0f, luminance);
        return luminance;
    }

    const cv::Mat f32 = convertToLinearColor(input, srgb);
    std::vector<cv::Mat> channels;
    cv::split(f32, channels);

    cv::Mat luminance =
        0.0722f * channels[0] + 0.7152f * channels[1] + 0.2126f * channels[2];
    cv::max(luminance, 0.0f, luminance);
    return luminance;
}

double normalizeRelativeIntensityStack(std::vector<cv::Mat>& images, bool scalePeakToOne) {
    double peak = 0.0;
    for (cv::Mat& image : images) {
        if (image.depth() != CV_32F) {
            throw std::runtime_error("Relative intensity normalization requires floating-point images.");
        }
        cv::Mat samples = image.reshape(1);
        for (int y = 0; y < samples.rows; ++y) {
            float* row = samples.ptr<float>(y);
            for (int x = 0; x < samples.cols; ++x) {
                if (!std::isfinite(row[x]) || row[x] < 0.0f) {
                    row[x] = 0.0f;
                } else {
                    peak = std::max(peak, static_cast<double>(row[x]));
                }
            }
        }
    }
    if (!std::isfinite(peak) || peak <= std::numeric_limits<float>::epsilon()) {
        throw std::runtime_error("Input image stack contains no positive finite intensity values.");
    }

    if (scalePeakToOne || peak > 1.0) {
        const float scale = static_cast<float>(1.0 / peak);
        for (cv::Mat& image : images) {
            image *= scale;
        }
    }
    return peak;
}
