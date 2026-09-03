#include "image_io.hpp"
#include "checked_io.hpp"
#include "radiometry.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
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

bool insideSphere(const Sphere& sphere, int x, int y, double margin = 0.0) {
    const double dx = static_cast<double>(x) - sphere.cx;
    const double dy = static_cast<double>(y) - sphere.cy;
    const double r = sphere.radius + margin;
    return dx * dx + dy * dy <= r * r;
}

std::string lowerAscii(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

double imageDescriptionUnitMm(const std::string& description) {
    const std::string lower = lowerAscii(description);
    const size_t pos = lower.find("unit=");
    if (pos == std::string::npos) {
        return 0.0;
    }
    const std::string unit = lower.substr(pos + 5, 32);
    if (unit.find("micron") != std::string::npos ||
        unit.find("micrometer") != std::string::npos ||
        unit.find("um") != std::string::npos) {
        return 0.001;
    }
    if (unit.find("millimeter") != std::string::npos || unit.find("mm") != std::string::npos) {
        return 1.0;
    }
    if (unit.find("nanometer") != std::string::npos || unit.find("nm") != std::string::npos) {
        return 0.000001;
    }
    if (unit.find("centimeter") != std::string::npos || unit.find("centimetre") != std::string::npos ||
        unit.find("cm") != std::string::npos) {
        return 10.0;
    }
    if (unit.find("meter") != std::string::npos || unit.find("metre") != std::string::npos) {
        return 1000.0;
    }
    return 0.0;
}

double averagePositive(double a, double b) {
    if (a > 0.0 && b > 0.0) {
        return 0.5 * (a + b);
    }
    return a > 0.0 ? a : b;
}

class ClassicTiffReader {
public:
    explicit ClassicTiffReader(std::vector<std::uint8_t> bytesInput)
        : bytes(std::move(bytesInput)) {
        if (bytes.size() < 8) {
            return;
        }
        if (bytes[0] == 'I' && bytes[1] == 'I') {
            little = true;
        } else if (bytes[0] == 'M' && bytes[1] == 'M') {
            little = false;
        } else {
            return;
        }
        valid = readU16(2) == 42;
    }

    bool isValid() const {
        return valid;
    }

    std::uint32_t firstIfdOffset() const {
        return readU32(4);
    }

    std::uint16_t readU16(size_t offset) const {
        if (offset + 2 > bytes.size()) {
            return 0;
        }
        if (little) {
            return static_cast<std::uint16_t>(bytes[offset] | (bytes[offset + 1] << 8));
        }
        return static_cast<std::uint16_t>((bytes[offset] << 8) | bytes[offset + 1]);
    }

    std::uint32_t readU32(size_t offset) const {
        if (offset + 4 > bytes.size()) {
            return 0;
        }
        if (little) {
            return static_cast<std::uint32_t>(bytes[offset]) |
                (static_cast<std::uint32_t>(bytes[offset + 1]) << 8) |
                (static_cast<std::uint32_t>(bytes[offset + 2]) << 16) |
                (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
        }
        return (static_cast<std::uint32_t>(bytes[offset]) << 24) |
            (static_cast<std::uint32_t>(bytes[offset + 1]) << 16) |
            (static_cast<std::uint32_t>(bytes[offset + 2]) << 8) |
            static_cast<std::uint32_t>(bytes[offset + 3]);
    }

    double readDouble(size_t offset) const {
        if (offset + 8 > bytes.size()) {
            return 0.0;
        }
        std::uint64_t raw = 0;
        if (little) {
            for (int i = 7; i >= 0; --i) {
                raw = (raw << 8) | bytes[offset + static_cast<size_t>(i)];
            }
        } else {
            for (int i = 0; i < 8; ++i) {
                raw = (raw << 8) | bytes[offset + static_cast<size_t>(i)];
            }
        }
        double value = 0.0;
        static_assert(sizeof(value) == sizeof(raw), "Unexpected double size");
        std::memcpy(&value, &raw, sizeof(value));
        return value;
    }

    double readRational(size_t offset) const {
        const std::uint32_t numerator = readU32(offset);
        const std::uint32_t denominator = readU32(offset + 4);
        if (denominator == 0) {
            return 0.0;
        }
        return static_cast<double>(numerator) / static_cast<double>(denominator);
    }

    const std::vector<std::uint8_t>& data() const {
        return bytes;
    }

private:
    std::vector<std::uint8_t> bytes;
    bool little = true;
    bool valid = false;
};

size_t tiffTypeSize(std::uint16_t type) {
    switch (type) {
    case 1:
    case 2:
        return 1;
    case 3:
        return 2;
    case 4:
    case 9:
        return 4;
    case 5:
    case 10:
    case 12:
        return 8;
    default:
        return 0;
    }
}

size_t tiffEntryValueOffset(const ClassicTiffReader& reader, size_t entry, std::uint16_t type, std::uint32_t count) {
    const size_t total = tiffTypeSize(type) * static_cast<size_t>(count);
    if (total == 0) {
        return 0;
    }
    if (total <= 4) {
        return entry + 8;
    }
    return reader.readU32(entry + 8);
}

double epsgLinearUnitMm(std::uint16_t code) {
    // Common EPSG length-unit codes used by GeoTIFF. Unknown units are not guessed.
    switch (code) {
    case 9001: // metre
        return 1000.0;
    case 9002: // international foot
        return 304.8;
    case 9003: // US survey foot
        return 1200.0 / 3.937;
    case 9030: // nautical mile
        return 1852000.0;
    case 9036: // kilometre
        return 1000000.0;
    default:
        return 0.0;
    }
}

double geoTiffModelUnitMm(
    const ClassicTiffReader& reader,
    size_t geoKeyOffset,
    std::uint32_t geoKeyCount,
    size_t geoDoubleOffset,
    std::uint32_t geoDoubleCount) {
    constexpr std::uint16_t kProjLinearUnitsGeoKey = 3076;
    constexpr std::uint16_t kProjLinearUnitSizeGeoKey = 3077;
    constexpr std::uint16_t kGeoDoubleParamsTag = 34736;
    constexpr std::uint16_t kUserDefined = 32767;

    if (geoKeyCount < 4 || geoKeyOffset > reader.data().size() ||
        static_cast<size_t>(geoKeyCount) > (reader.data().size() - geoKeyOffset) / 2) {
        return 0.0;
    }

    const std::uint16_t keyCount = reader.readU16(geoKeyOffset + 6);
    if (static_cast<size_t>(keyCount) > (static_cast<size_t>(geoKeyCount) - 4) / 4) {
        return 0.0;
    }

    std::uint16_t linearUnitCode = 0;
    double userUnitMeters = 0.0;
    for (std::uint16_t i = 0; i < keyCount; ++i) {
        const size_t key = geoKeyOffset + 8 + static_cast<size_t>(i) * 8;
        const std::uint16_t keyId = reader.readU16(key);
        const std::uint16_t location = reader.readU16(key + 2);
        const std::uint16_t count = reader.readU16(key + 4);
        const std::uint16_t valueOffset = reader.readU16(key + 6);
        if (keyId == kProjLinearUnitsGeoKey && location == 0 && count == 1) {
            linearUnitCode = valueOffset;
        } else if (
            keyId == kProjLinearUnitSizeGeoKey && location == kGeoDoubleParamsTag && count >= 1 &&
            valueOffset < geoDoubleCount && geoDoubleOffset <= reader.data().size() &&
            static_cast<size_t>(valueOffset) < (reader.data().size() - geoDoubleOffset) / 8) {
            userUnitMeters = reader.readDouble(geoDoubleOffset + static_cast<size_t>(valueOffset) * 8);
        }
    }

    const double registeredUnitMm = epsgLinearUnitMm(linearUnitCode);
    if (registeredUnitMm > 0.0) {
        return registeredUnitMm;
    }
    if (linearUnitCode == kUserDefined && userUnitMeters > 0.0 && std::isfinite(userUnitMeters)) {
        return 1000.0 * userUnitMeters;
    }
    return 0.0;
}

double readClassicTiffPixelScaleMm(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return 0.0;
    }
    std::vector<std::uint8_t> bytes(
        (std::istreambuf_iterator<char>(in)),
        std::istreambuf_iterator<char>());
    ClassicTiffReader reader(std::move(bytes));
    if (!reader.isValid()) {
        return 0.0;
    }

    const size_t ifd = reader.firstIfdOffset();
    if (ifd + 2 > reader.data().size()) {
        return 0.0;
    }
    const std::uint16_t entries = reader.readU16(ifd);
    double xResolution = 0.0;
    double yResolution = 0.0;
    double modelScaleX = 0.0;
    double modelScaleY = 0.0;
    int resolutionUnit = 0;
    std::string imageDescription;
    size_t geoKeyOffset = 0;
    std::uint32_t geoKeyCount = 0;
    size_t geoDoubleOffset = 0;
    std::uint32_t geoDoubleCount = 0;

    for (std::uint16_t i = 0; i < entries; ++i) {
        const size_t entry = ifd + 2 + static_cast<size_t>(i) * 12;
        if (entry + 12 > reader.data().size()) {
            break;
        }
        const std::uint16_t tag = reader.readU16(entry);
        const std::uint16_t type = reader.readU16(entry + 2);
        const std::uint32_t count = reader.readU32(entry + 4);
        const size_t valueOffset = tiffEntryValueOffset(reader, entry, type, count);
        if (valueOffset == 0 || valueOffset >= reader.data().size()) {
            continue;
        }
        if (tag == 282 && type == 5 && count >= 1) {
            xResolution = reader.readRational(valueOffset);
        } else if (tag == 283 && type == 5 && count >= 1) {
            yResolution = reader.readRational(valueOffset);
        } else if (tag == 296 && type == 3 && count >= 1) {
            resolutionUnit = reader.readU16(valueOffset);
        } else if (tag == 270 && type == 2 && count > 0) {
            const size_t readable = std::min(static_cast<size_t>(count), reader.data().size() - valueOffset);
            imageDescription.assign(
                reinterpret_cast<const char*>(reader.data().data() + valueOffset),
                readable);
        } else if (tag == 33550 && type == 12 && count >= 2) {
            modelScaleX = reader.readDouble(valueOffset);
            modelScaleY = reader.readDouble(valueOffset + 8);
        } else if (tag == 34735 && type == 3 && count >= 4) {
            geoKeyOffset = valueOffset;
            geoKeyCount = count;
        } else if (tag == 34736 && type == 12 && count >= 1) {
            geoDoubleOffset = valueOffset;
            geoDoubleCount = count;
        }
    }

    const double averageResolution = averagePositive(xResolution, yResolution);
    if (averageResolution > 0.0) {
        if (resolutionUnit == 2) {
            return 25.4 / averageResolution;
        }
        if (resolutionUnit == 3) {
            return 10.0 / averageResolution;
        }
        const double descriptionUnitMm = imageDescriptionUnitMm(imageDescription);
        if (descriptionUnitMm > 0.0) {
            return descriptionUnitMm / averageResolution;
        }
    }

    const double modelScale = averagePositive(modelScaleX, modelScaleY);
    if (modelScale > 0.0 && std::isfinite(modelScale)) {
        const double geoUnitMm = geoTiffModelUnitMm(
            reader,
            geoKeyOffset,
            geoKeyCount,
            geoDoubleOffset,
            geoDoubleCount);
        if (geoUnitMm > 0.0) {
            return modelScale * geoUnitMm;
        }
        const double descriptionUnitMm = imageDescriptionUnitMm(imageDescription);
        if (descriptionUnitMm > 0.0) {
            return modelScale * descriptionUnitMm;
        }
    }
    return 0.0;
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

cv::Mat hillshadeTo8U(const cv::Mat& normalMap, const cv::Mat& mask, const cv::Vec3f& lightDirection) {
    const float length = std::sqrt(lightDirection.dot(lightDirection));
    const cv::Vec3f light = length > 1.0e-6f ? lightDirection / length : cv::Vec3f(-0.5f, 0.5f, 0.70710678f);
    constexpr double kAmbient = 0.16;
    constexpr double kDiffuse = 0.82;
    constexpr double kGamma = 0.78;
    cv::Mat out(normalMap.rows, normalMap.cols, CV_8U, cv::Scalar(0));
    for (int y = 0; y < normalMap.rows; ++y) {
        const cv::Vec3f* nrow = normalMap.ptr<cv::Vec3f>(y);
        const uchar* mrow = mask.ptr<uchar>(y);
        uchar* orow = out.ptr<uchar>(y);
        for (int x = 0; x < normalMap.cols; ++x) {
            if (mrow[x] == 0) {
                continue;
            }
            const double ndotl =
                static_cast<double>(nrow[x][0]) * light[0] +
                static_cast<double>(nrow[x][1]) * light[1] +
                static_cast<double>(nrow[x][2]) * light[2];
            const double hillshade = std::pow(std::clamp(ndotl, 0.0, 1.0), kGamma);
            const double value = kAmbient + kDiffuse * hillshade;
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

void writeNormalSet(
    const fs::path& outDir,
    const std::string& prefix,
    const cv::Mat& normalMap,
    const cv::Mat& mask) {
    writeImageChecked(outDir / (prefix + "_normal_rgb.png"), normalRgbTo8U(normalMap, mask));
    writeImageChecked(outDir / (prefix + "_normal_x.png"), normalComponentTo8U(normalMap, mask, 0, true));
    writeImageChecked(outDir / (prefix + "_normal_y.png"), normalComponentTo8U(normalMap, mask, 1, true));
    writeImageChecked(outDir / (prefix + "_hillshade_ul.png"), hillshadeTo8U(normalMap, mask, cv::Vec3f(-0.5f, 0.5f, 0.70710678f)));
    writeImageChecked(outDir / (prefix + "_normal_z.png"), normalComponentTo8U(normalMap, mask, 2, false));
}

cv::Vec3f normalizeVector(const cv::Vec3f& v) {
    const float length = std::sqrt(v.dot(v));
    if (length <= 1.0e-6f) {
        return cv::Vec3f(0.0f, 0.0f, 1.0f);
    }
    return v / length;
}

float smoothstep(float edge0, float edge1, float x) {
    const float t = std::clamp((x - edge0) / std::max(1.0e-6f, edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

cv::Vec3f mixColor(const cv::Vec3f& a, const cv::Vec3f& b, float t) {
    return a * (1.0f - t) + b * t;
}

cv::Vec3f reflectVector(const cv::Vec3f& incident, const cv::Vec3f& normal) {
    return incident - 2.0f * incident.dot(normal) * normal;
}

float maskedAbsPercentile(const cv::Mat& src, const cv::Mat& mask, float percentileValue) {
    std::vector<float> values;
    values.reserve(static_cast<size_t>(src.rows * src.cols / 4));
    for (int y = 0; y < src.rows; ++y) {
        const float* row = src.ptr<float>(y);
        const uchar* mrow = mask.ptr<uchar>(y);
        for (int x = 0; x < src.cols; ++x) {
            if (mrow[x] != 0 && std::isfinite(row[x])) {
                values.push_back(std::abs(row[x]));
            }
        }
    }
    if (values.empty()) {
        return 1.0f;
    }
    const size_t index = static_cast<size_t>(
        std::clamp(percentileValue, 0.0f, 1.0f) * static_cast<float>(values.size() - 1));
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(index), values.end());
    return std::max(values[index], 1.0e-6f);
}

cv::Mat maskedGaussianBlur(const cv::Mat& src, const cv::Mat& mask, double sigma) {
    cv::Mat maskFloat;
    mask.convertTo(maskFloat, CV_32F, 1.0 / 255.0);

    cv::Mat weighted(src.size(), CV_32F, cv::Scalar(0));
    for (int y = 0; y < src.rows; ++y) {
        const float* srcRow = src.ptr<float>(y);
        const float* maskRow = maskFloat.ptr<float>(y);
        float* weightedRow = weighted.ptr<float>(y);
        for (int x = 0; x < src.cols; ++x) {
            weightedRow[x] = std::isfinite(srcRow[x]) ? srcRow[x] * maskRow[x] : 0.0f;
        }
    }

    cv::Mat numerator;
    cv::Mat denominator;
    cv::GaussianBlur(weighted, numerator, cv::Size(0, 0), sigma);
    cv::GaussianBlur(maskFloat, denominator, cv::Size(0, 0), sigma);

    cv::Mat out(src.size(), CV_32F, cv::Scalar(0));
    for (int y = 0; y < src.rows; ++y) {
        const float* numRow = numerator.ptr<float>(y);
        const float* denRow = denominator.ptr<float>(y);
        float* outRow = out.ptr<float>(y);
        for (int x = 0; x < src.cols; ++x) {
            outRow[x] = denRow[x] > 1.0e-6f ? numRow[x] / denRow[x] : 0.0f;
        }
    }
    return out;
}

cv::Mat maskedGaussianBlurVec3(const cv::Mat& src, const cv::Mat& mask, double sigma) {
    std::vector<cv::Mat> channels;
    cv::split(src, channels);
    for (cv::Mat& channel : channels) {
        channel = maskedGaussianBlur(channel, mask, sigma);
    }
    cv::Mat out;
    cv::merge(channels, out);
    return out;
}

float maskedColorLuminancePercentile(const cv::Mat& image, const cv::Mat& mask, float percentileValue) {
    std::vector<float> values;
    values.reserve(static_cast<size_t>(image.rows * image.cols / 4));
    for (int y = 0; y < image.rows; ++y) {
        const cv::Vec3b* row = image.ptr<cv::Vec3b>(y);
        const uchar* mrow = mask.ptr<uchar>(y);
        for (int x = 0; x < image.cols; ++x) {
            if (mrow[x] == 0) {
                continue;
            }
            const cv::Vec3b bgr = row[x];
            values.push_back(0.0722f * static_cast<float>(bgr[0]) +
                0.7152f * static_cast<float>(bgr[1]) +
                0.2126f * static_cast<float>(bgr[2]));
        }
    }
    if (values.empty()) {
        return 0.0f;
    }
    const size_t index = static_cast<size_t>(
        std::clamp(percentileValue, 0.0f, 1.0f) * static_cast<float>(values.size() - 1));
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(index), values.end());
    return values[index];
}

cv::Mat stretchColorByMaskedLuminance(const cv::Mat& image, const cv::Mat& mask) {
    const float low = maskedColorLuminancePercentile(image, mask, 0.01f);
    const float high = maskedColorLuminancePercentile(image, mask, 0.992f);
    if (high - low < 8.0f) {
        return image;
    }

    const float outLow = 6.0f;
    const float outHigh = 248.0f;
    const float scale = (outHigh - outLow) / (high - low);
    cv::Mat stretched = image.clone();
    for (int y = 0; y < image.rows; ++y) {
        const uchar* mrow = mask.ptr<uchar>(y);
        cv::Vec3b* row = stretched.ptr<cv::Vec3b>(y);
        for (int x = 0; x < image.cols; ++x) {
            if (mrow[x] == 0) {
                continue;
            }
            for (int c = 0; c < 3; ++c) {
                const float v = (static_cast<float>(row[x][c]) - low) * scale + outLow;
                row[x][c] = static_cast<uchar>(std::clamp(v, 0.0f, 255.0f));
            }
        }
    }
    return stretched;
}

cv::Mat liquidMetalTo8U(const cv::Mat& normalMap, const cv::Mat& mask) {
    cv::Mat smoothNormals = maskedGaussianBlurVec3(normalMap, mask, 0.45);
    const cv::Vec3f view(0.0f, 0.0f, 1.0f);
    const cv::Vec3f keyLight = normalizeVector(cv::Vec3f(-0.45f, 0.35f, 0.82f));
    const cv::Vec3f fillLight = normalizeVector(cv::Vec3f(0.58f, -0.42f, 0.70f));
    const cv::Vec3f darkSteel(0.035f, 0.045f, 0.050f);
    const cv::Vec3f silver(0.72f, 0.76f, 0.78f);
    const cv::Vec3f coolCyan(0.42f, 0.72f, 0.80f);
    const cv::Vec3f warmGold(0.95f, 0.76f, 0.42f);

    cv::Mat out(normalMap.rows, normalMap.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int y = 0; y < normalMap.rows; ++y) {
        const cv::Vec3f* rawRow = normalMap.ptr<cv::Vec3f>(y);
        const cv::Vec3f* nrow = smoothNormals.ptr<cv::Vec3f>(y);
        const uchar* mrow = mask.ptr<uchar>(y);
        cv::Vec3b* outRow = out.ptr<cv::Vec3b>(y);
        for (int x = 0; x < normalMap.cols; ++x) {
            if (mrow[x] == 0) {
                continue;
            }

            const cv::Vec3f n = normalizeVector(mixColor(rawRow[x], nrow[x], 0.18f));
            const cv::Vec3f reflectedView = normalizeVector(reflectVector(-view, n));
            const float ndv = std::clamp(n.dot(view), 0.0f, 1.0f);
            const float key = std::max(0.0f, n.dot(keyLight));
            const float fill = std::max(0.0f, n.dot(fillLight));
            const float stripe = smoothstep(0.32f, 0.86f, 0.5f + 0.5f * (reflectedView[0] * 0.70f + reflectedView[1] * 0.30f));

            cv::Vec3f base = mixColor(darkSteel, silver, stripe);
            base = mixColor(base, coolCyan, 0.20f * smoothstep(0.05f, 0.90f, reflectedView[1] * 0.5f + 0.5f));
            base = mixColor(base, warmGold, 0.16f * smoothstep(0.25f, 1.00f, reflectedView[0] * -0.5f + 0.5f));

            const cv::Vec3f keyReflection = reflectVector(-keyLight, n);
            const cv::Vec3f fillReflection = reflectVector(-fillLight, n);
            const float specKey = std::pow(std::max(0.0f, keyReflection.dot(view)), 60.0f);
            const float specFill = std::pow(std::max(0.0f, fillReflection.dot(view)), 32.0f);
            const float rim = std::pow(1.0f - ndv, 2.0f);

            cv::Vec3f rgb = base * (0.42f + 0.46f * key + 0.18f * fill);
            rgb += cv::Vec3f(1.0f, 0.96f, 0.88f) * (1.00f * specKey + 0.36f * specFill);
            rgb += coolCyan * (0.20f * rim);

            for (int c = 0; c < 3; ++c) {
                rgb[c] = 1.0f - std::exp(-std::max(0.0f, rgb[c]) * 1.35f);
                rgb[c] = std::pow(std::clamp(rgb[c], 0.0f, 1.0f), 1.0f / 2.2f);
            }
            outRow[x] = cv::Vec3b(
                static_cast<uchar>(std::clamp(rgb[2] * 255.0f, 0.0f, 255.0f)),
                static_cast<uchar>(std::clamp(rgb[1] * 255.0f, 0.0f, 255.0f)),
                static_cast<uchar>(std::clamp(rgb[0] * 255.0f, 0.0f, 255.0f)));
        }
    }
    return stretchColorByMaskedLuminance(out, mask);
}

void writePfm(const fs::path& path, const cv::Mat& image, const cv::Mat& mask) {
    CheckedOutputFile checked(path);
    std::ostream& out = checked.stream();
    out << "Pf\n" << image.cols << " " << image.rows << "\n-1.0\n";
    for (int y = image.rows - 1; y >= 0; --y) {
        const float* row = image.ptr<float>(y);
        const uchar* mrow = mask.ptr<uchar>(y);
        for (int x = 0; x < image.cols; ++x) {
            float v = (mrow[x] != 0 && std::isfinite(row[x])) ? row[x] : 0.0f;
            out.write(reinterpret_cast<const char*>(&v), sizeof(float));
        }
    }
    checked.commit();
}

void writeLightsCsv(
    const fs::path& path,
    const Options& opt,
    const std::vector<cv::Vec3f>& lights,
    const std::vector<HighlightEstimate>& estimates) {
    CheckedOutputFile checked(path);
    std::ostream& out = checked.stream();
    if (opt.uncalibratedLighting) {
        out << "mode,uncalibrated_unknown_lighting\n";
        out << "note,No sphere or calibrated light vectors were used. Normals and height are relative and ambiguous.\n";
        out << "image\n";
        for (const std::string& imagePath : opt.imagePaths) {
            out << '"' << imagePath << '"' << '\n';
        }
        checked.commit();
        return;
    }
    out << "lighting_model," << (opt.lightingModel == LightingModel::NearFieldRing ? "near_field_ring" : "directional") << "\n";
    if (opt.lightingModel == LightingModel::NearFieldRing) {
        out << "ring_light_radius_mm," << opt.ringLightRadiusMm << "\n";
        out << "ring_light_height_mm," << opt.ringLightHeightMm << "\n";
    }
    if (opt.pixelScaleMm > 0.0) {
        out << "pixel_scale_mm_per_pixel," << opt.pixelScaleMm << "\n";
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
    checked.commit();
}

void writeLightVectorsCsv(const fs::path& path, const std::vector<cv::Vec3f>& lights) {
    CheckedOutputFile checked(path);
    std::ostream& out = checked.stream();
    out << "x,y,z\n";
    out << std::fixed << std::setprecision(8);
    for (const cv::Vec3f& light : lights) {
        out << light[0] << ',' << light[1] << ',' << light[2] << '\n';
    }
    checked.commit();
}

void writePlyMesh(
    const fs::path& path,
    const cv::Mat& height,
    const cv::Mat& mask,
    const cv::Mat& vertexColor,
    int step,
    double heightScale,
    const std::function<void(const std::string&)>& progress) {
    const int rows = height.rows;
    const int cols = height.cols;
    cv::Mat index(rows, cols, CV_32S, cv::Scalar(-1));

    reportProgress(progress, "PLY: indexing vertices...");
    int vertexCount = 0;
    for (int y = 0; y < rows; y += step) {
        const float* hrow = height.ptr<float>(y);
        const uchar* mrow = mask.ptr<uchar>(y);
        int* irow = index.ptr<int>(y);
        for (int x = 0; x < cols; x += step) {
            if (mrow[x] != 0 && std::isfinite(hrow[x])) {
                irow[x] = vertexCount++;
            }
        }
    }

    reportProgress(progress, "PLY: counting faces...");
    int faceCount = 0;
    for (int y = 0; y + step < rows; y += step) {
        const int* row0 = index.ptr<int>(y);
        const int* row1 = index.ptr<int>(y + step);
        for (int x = 0; x + step < cols; x += step) {
            const int a = row0[x];
            const int b = row0[x + step];
            const int c = row1[x];
            const int d = row1[x + step];
            if (a >= 0 && b >= 0 && c >= 0) {
                ++faceCount;
            }
            if (b >= 0 && d >= 0 && c >= 0) {
                ++faceCount;
            }
        }
    }

    CheckedOutputFile checked(path);
    std::ostream& out = checked.stream();

    out << "ply\n";
    out << "format binary_little_endian 1.0\n";
    out << "comment generated by What A Relief\n";
    out << "element vertex " << vertexCount << "\n";
    out << "property float x\n";
    out << "property float y\n";
    out << "property float z\n";
    out << "property uchar red\n";
    out << "property uchar green\n";
    out << "property uchar blue\n";
    out << "element face " << faceCount << "\n";
    out << "property list uchar int vertex_indices\n";
    out << "end_header\n";

    reportProgress(progress, "PLY: writing binary vertices...");
    for (int y = 0; y < rows; y += step) {
        const float* hrow = height.ptr<float>(y);
        const int* irow = index.ptr<int>(y);
        const uchar* colorRow = vertexColor.empty() ? nullptr : vertexColor.ptr<uchar>(y);
        for (int x = 0; x < cols; x += step) {
            if (irow[x] >= 0) {
                const float vx = static_cast<float>(x);
                const float vy = static_cast<float>(-y);
                const float vz = static_cast<float>(static_cast<double>(hrow[x]) * heightScale);
                const std::uint8_t color = colorRow == nullptr ? 200 : colorRow[x];
                out.write(reinterpret_cast<const char*>(&vx), sizeof(vx));
                out.write(reinterpret_cast<const char*>(&vy), sizeof(vy));
                out.write(reinterpret_cast<const char*>(&vz), sizeof(vz));
                out.write(reinterpret_cast<const char*>(&color), sizeof(color));
                out.write(reinterpret_cast<const char*>(&color), sizeof(color));
                out.write(reinterpret_cast<const char*>(&color), sizeof(color));
            }
        }
    }

    reportProgress(progress, "PLY: writing binary faces...");
    for (int y = 0; y + step < rows; y += step) {
        const int* row0 = index.ptr<int>(y);
        const int* row1 = index.ptr<int>(y + step);
        for (int x = 0; x + step < cols; x += step) {
            const int a = row0[x];
            const int b = row0[x + step];
            const int c = row1[x];
            const int d = row1[x + step];
            if (a >= 0 && b >= 0 && c >= 0) {
                const std::uint8_t count = 3;
                const std::int32_t ia = static_cast<std::int32_t>(a);
                const std::int32_t ib = static_cast<std::int32_t>(c);
                const std::int32_t ic = static_cast<std::int32_t>(b);
                out.write(reinterpret_cast<const char*>(&count), sizeof(count));
                out.write(reinterpret_cast<const char*>(&ia), sizeof(ia));
                out.write(reinterpret_cast<const char*>(&ib), sizeof(ib));
                out.write(reinterpret_cast<const char*>(&ic), sizeof(ic));
            }
            if (b >= 0 && d >= 0 && c >= 0) {
                const std::uint8_t count = 3;
                const std::int32_t ia = static_cast<std::int32_t>(b);
                const std::int32_t ib = static_cast<std::int32_t>(c);
                const std::int32_t ic = static_cast<std::int32_t>(d);
                out.write(reinterpret_cast<const char*>(&count), sizeof(count));
                out.write(reinterpret_cast<const char*>(&ia), sizeof(ia));
                out.write(reinterpret_cast<const char*>(&ib), sizeof(ib));
                out.write(reinterpret_cast<const char*>(&ic), sizeof(ic));
            }
        }
    }

    if (!out) {
        die("Failed while writing PLY mesh: " + path.string());
    }
    checked.commit();
    reportProgress(
        progress,
        "PLY: done (" + std::to_string(vertexCount) + " vertices, " +
            std::to_string(faceCount) + " faces).");
}

struct PlyBoundaryEdge {
    std::int32_t a = -1;
    std::int32_t b = -1;
    int count = 0;
};

std::uint64_t plyEdgeKey(std::int32_t a, std::int32_t b) {
    const std::uint32_t lo = static_cast<std::uint32_t>(std::min(a, b));
    const std::uint32_t hi = static_cast<std::uint32_t>(std::max(a, b));
    return (static_cast<std::uint64_t>(lo) << 32) | static_cast<std::uint64_t>(hi);
}

void addPlyEdge(std::unordered_map<std::uint64_t, PlyBoundaryEdge>& edges, std::int32_t a, std::int32_t b) {
    PlyBoundaryEdge& edge = edges[plyEdgeKey(a, b)];
    if (edge.count == 0) {
        edge.a = a;
        edge.b = b;
    }
    ++edge.count;
}

void addPlyTriangleEdges(std::unordered_map<std::uint64_t, PlyBoundaryEdge>& edges, std::int32_t a, std::int32_t b, std::int32_t c) {
    addPlyEdge(edges, a, b);
    addPlyEdge(edges, b, c);
    addPlyEdge(edges, c, a);
}

void writePlyTriangle(std::ostream& out, std::int32_t a, std::int32_t b, std::int32_t c) {
    const std::uint8_t count = 3;
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    out.write(reinterpret_cast<const char*>(&a), sizeof(a));
    out.write(reinterpret_cast<const char*>(&b), sizeof(b));
    out.write(reinterpret_cast<const char*>(&c), sizeof(c));
}

void writePlyQuad(std::ostream& out, std::int32_t a, std::int32_t b, std::int32_t c, std::int32_t d) {
    const std::uint8_t count = 4;
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    out.write(reinterpret_cast<const char*>(&a), sizeof(a));
    out.write(reinterpret_cast<const char*>(&b), sizeof(b));
    out.write(reinterpret_cast<const char*>(&c), sizeof(c));
    out.write(reinterpret_cast<const char*>(&d), sizeof(d));
}

void writePrintablePlyMesh(
    const fs::path& path,
    const cv::Mat& height,
    const cv::Mat& mask,
    const cv::Mat& vertexColor,
    int step,
    double heightScale,
    double pixelScaleMm,
    double baseThicknessMm,
    const std::function<void(const std::string&)>& progress) {
    const int rows = height.rows;
    const int cols = height.cols;
    const double xyScale = pixelScaleMm > 0.0 ? pixelScaleMm : 1.0;
    const double zScale = xyScale * heightScale;
    const double baseThickness = pixelScaleMm > 0.0 ? baseThicknessMm : baseThicknessMm / xyScale;
    cv::Mat index(rows, cols, CV_32S, cv::Scalar(-1));

    reportProgress(progress, "Printable PLY: indexing top vertices...");
    int topVertexCount = 0;
    double minZ = std::numeric_limits<double>::infinity();
    for (int y = 0; y < rows; y += step) {
        const float* hrow = height.ptr<float>(y);
        const uchar* mrow = mask.ptr<uchar>(y);
        int* irow = index.ptr<int>(y);
        for (int x = 0; x < cols; x += step) {
            if (mrow[x] != 0 && std::isfinite(hrow[x])) {
                irow[x] = topVertexCount++;
                minZ = std::min(minZ, static_cast<double>(hrow[x]) * zScale);
            }
        }
    }
    if (topVertexCount < 3 || !std::isfinite(minZ)) {
        die("Printable PLY needs at least three finite height pixels inside the geometry mask.");
    }

    reportProgress(progress, "Printable PLY: counting faces...");
    std::unordered_map<std::uint64_t, PlyBoundaryEdge> edges;
    edges.reserve(static_cast<std::size_t>(topVertexCount) * 3);
    int topFaceCount = 0;
    for (int y = 0; y + step < rows; y += step) {
        const int* row0 = index.ptr<int>(y);
        const int* row1 = index.ptr<int>(y + step);
        for (int x = 0; x + step < cols; x += step) {
            const int a = row0[x];
            const int b = row0[x + step];
            const int c = row1[x];
            const int d = row1[x + step];
            if (a >= 0 && b >= 0 && c >= 0) {
                addPlyTriangleEdges(edges, a, c, b);
                ++topFaceCount;
            }
            if (b >= 0 && d >= 0 && c >= 0) {
                addPlyTriangleEdges(edges, b, c, d);
                ++topFaceCount;
            }
        }
    }
    int boundaryFaceCount = 0;
    for (const auto& item : edges) {
        if (item.second.count == 1) {
            ++boundaryFaceCount;
        }
    }
    if (topFaceCount == 0) {
        die("Printable PLY has no surface faces. Use a smaller mesh step or a larger height mask.");
    }
    const int vertexCount = topVertexCount * 2;
    const int faceCount = topFaceCount * 2 + boundaryFaceCount;
    const double baseZ = minZ - std::max(0.0, baseThickness);

    CheckedOutputFile checked(path);
    std::ostream& out = checked.stream();

    out << "ply\n";
    out << "format binary_little_endian 1.0\n";
    out << "comment generated by What A Relief printable solid export\n";
    out << "comment units " << (pixelScaleMm > 0.0 ? "millimeters" : "input_pixels") << "\n";
    out << "element vertex " << vertexCount << "\n";
    out << "property float x\n";
    out << "property float y\n";
    out << "property float z\n";
    out << "property uchar red\n";
    out << "property uchar green\n";
    out << "property uchar blue\n";
    out << "element face " << faceCount << "\n";
    out << "property list uchar int vertex_indices\n";
    out << "end_header\n";

    reportProgress(progress, "Printable PLY: writing vertices...");
    for (int layer = 0; layer < 2; ++layer) {
        for (int y = 0; y < rows; y += step) {
            const float* hrow = height.ptr<float>(y);
            const int* irow = index.ptr<int>(y);
            const uchar* colorRow = vertexColor.empty() ? nullptr : vertexColor.ptr<uchar>(y);
            for (int x = 0; x < cols; x += step) {
                if (irow[x] >= 0) {
                    const float vx = static_cast<float>(static_cast<double>(x) * xyScale);
                    const float vy = static_cast<float>(static_cast<double>(-y) * xyScale);
                    const float vz = layer == 0
                        ? static_cast<float>(static_cast<double>(hrow[x]) * zScale)
                        : static_cast<float>(baseZ);
                    const std::uint8_t color = colorRow == nullptr ? 200 : colorRow[x];
                    out.write(reinterpret_cast<const char*>(&vx), sizeof(vx));
                    out.write(reinterpret_cast<const char*>(&vy), sizeof(vy));
                    out.write(reinterpret_cast<const char*>(&vz), sizeof(vz));
                    out.write(reinterpret_cast<const char*>(&color), sizeof(color));
                    out.write(reinterpret_cast<const char*>(&color), sizeof(color));
                    out.write(reinterpret_cast<const char*>(&color), sizeof(color));
                }
            }
        }
    }

    reportProgress(progress, "Printable PLY: writing closed faces...");
    for (int y = 0; y + step < rows; y += step) {
        const int* row0 = index.ptr<int>(y);
        const int* row1 = index.ptr<int>(y + step);
        for (int x = 0; x + step < cols; x += step) {
            const int a = row0[x];
            const int b = row0[x + step];
            const int c = row1[x];
            const int d = row1[x + step];
            if (a >= 0 && b >= 0 && c >= 0) {
                writePlyTriangle(out, a, c, b);
                writePlyTriangle(out, b + topVertexCount, c + topVertexCount, a + topVertexCount);
            }
            if (b >= 0 && d >= 0 && c >= 0) {
                writePlyTriangle(out, b, c, d);
                writePlyTriangle(out, d + topVertexCount, c + topVertexCount, b + topVertexCount);
            }
        }
    }
    for (const auto& item : edges) {
        const PlyBoundaryEdge& edge = item.second;
        if (edge.count == 1) {
            writePlyQuad(out, edge.b, edge.a, edge.a + topVertexCount, edge.b + topVertexCount);
        }
    }

    if (!out) {
        die("Failed while writing printable PLY mesh: " + path.string());
    }
    checked.commit();
    reportProgress(
        progress,
        "Printable PLY: done (" + std::to_string(vertexCount) + " vertices, " +
            std::to_string(faceCount) + " faces).");
}

} // namespace

double readPixelScaleMmFromImage(const std::string& path) {
    const std::string ext = lowerAscii(fs::path(path).extension().string());
    if (ext == ".tif" || ext == ".tiff") {
        return readClassicTiffPixelScaleMm(path);
    }
    return 0.0;
}

std::vector<cv::Mat> loadLuminanceImages(const std::vector<std::string>& paths, bool srgb) {
    std::vector<cv::Mat> images;
    images.reserve(paths.size());
    cv::Size expected;
    for (const std::string& path : paths) {
        cv::Mat raw = cv::imread(path, cv::IMREAD_UNCHANGED);
        if (raw.empty()) {
            die("Failed to read image: " + path);
        }
        cv::Mat gray = convertToLinearLuminance(raw, srgb);
        if (expected.empty()) {
            expected = gray.size();
        } else if (gray.size() != expected) {
            die("All input images must have identical dimensions. Mismatch at: " + path);
        }
        images.push_back(gray);
    }
    normalizeRelativeIntensityStack(images);
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

void applyCropToMask(cv::Mat& mask, const cv::Rect& crop) {
    const cv::Rect imageRect(0, 0, mask.cols, mask.rows);
    const cv::Rect clamped = crop & imageRect;
    if (clamped.empty()) {
        die("Crop rectangle does not overlap the image.");
    }

    cv::Mat cropMask(mask.size(), CV_8U, cv::Scalar(0));
    cropMask(clamped).setTo(cv::Scalar(255));
    cv::bitwise_and(mask, cropMask, mask);
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
    const PhotometricDiagnostics& diagnostics,
    const cv::Mat& height,
    const cv::Mat& heightMask,
    const std::function<void(const std::string&)>& progress) {
    const fs::path outDir = opt.outputDir;
    fs::create_directories(outDir);
    reportProgress(progress, "Writing light CSV files...");
    writeLightsCsv(outDir / "lights.csv", opt, lights, estimates);
    writeLightVectorsCsv(outDir / "light_vectors.csv", lights);
    reportProgress(progress, "Writing image outputs...");
    const cv::Mat albedo8 = normalizeFloatTo8U(albedo, validMask, true);
    writeImageChecked(outDir / "normal_rgb.png", normalRgbTo8U(normalMap, validMask));
    writeImageChecked(outDir / "normal_x.png", normalComponentTo8U(normalMap, validMask, 0, true));
    writeImageChecked(outDir / "normal_y.png", normalComponentTo8U(normalMap, validMask, 1, true));
    writeImageChecked(outDir / "hillshade_ul.png", hillshadeTo8U(normalMap, validMask, cv::Vec3f(-0.5f, 0.5f, 0.70710678f)));
    writeImageChecked(outDir / "normal_z.png", normalComponentTo8U(normalMap, validMask, 2, false));
    writeImageChecked(outDir / "albedo.png", albedo8);
    writeImageChecked(outDir / "residual.png", normalizeFloatTo8U(residual, validMask, true));
    writeImageChecked(outDir / "valid_mask.png", validMask);
    writeImageChecked(outDir / "liquid_metal.png", liquidMetalTo8U(normalMap, validMask));
    if (opt.neuralFusion && !diagnostics.neuralNormal.empty()) {
        writeNormalSet(outDir, "fused", normalMap, validMask);
        if (!diagnostics.classicalNormal.empty()) {
            const cv::Mat& classicalMask = diagnostics.classicalValidMask.empty()
                ? validMask
                : diagnostics.classicalValidMask;
            writeNormalSet(outDir, "classical", diagnostics.classicalNormal, classicalMask);
        }
        const cv::Mat neuralMask = diagnostics.neuralValidMask.empty()
            ? cv::Mat(diagnostics.neuralNormal.size(), CV_8U, cv::Scalar(0))
            : diagnostics.neuralValidMask;
        writeNormalSet(outDir, "neural", diagnostics.neuralNormal, neuralMask);
        writeImageChecked(outDir / "neural_valid_mask.png", neuralMask);
        writeImageChecked(outDir / "fused_classical_confidence.png", normalizeFloatTo8U(diagnostics.classicalConfidence, validMask, true));
    }
    if (opt.solverMode == NormalSolverMode::Robust && !diagnostics.robustWeight.empty()) {
        writeImageChecked(outDir / "robust_weight.png", normalizeFloatTo8U(diagnostics.robustWeight, validMask, true));
        if (!diagnostics.robustFallbackMask.empty()) {
            writeImageChecked(outDir / "robust_fallback_mask.png", diagnostics.robustFallbackMask);
        }
        writeImageChecked(outDir / "shadow_count.png", normalizeFloatTo8U(diagnostics.shadowCount, validMask, true));
        writeImageChecked(outDir / "highlight_outlier_count.png", normalizeFloatTo8U(diagnostics.highlightOutlierCount, validMask, true));
    }
    if (opt.specularDiagnostics && !diagnostics.specularCueMask.empty()) {
        writeImageChecked(outDir / "specular_cue_mask.png", diagnostics.specularCueMask);
    }
    if (!height.empty()) {
        reportProgress(progress, "Writing height outputs...");
        const cv::Mat& geometryMask = heightMask.empty() ? validMask : heightMask;
        writeImageChecked(outDir / "height_mask.png", geometryMask);
        writeImageChecked(outDir / "height.png", normalizeFloatTo8U(height, geometryMask, false));
        writePfm(outDir / "height.pfm", height, geometryMask);
        if (!opt.meshPath.empty()) {
            writePlyMesh(opt.meshPath, height, geometryMask, albedo8, opt.meshStep, opt.heightScale, progress);
        }
        if (!opt.printableMeshPath.empty()) {
            writePrintablePlyMesh(
                opt.printableMeshPath,
                height,
                geometryMask,
                albedo8,
                opt.meshStep,
                opt.heightScale,
                opt.pixelScaleMm,
                opt.printableThicknessMm,
                progress);
        }
    }
}
