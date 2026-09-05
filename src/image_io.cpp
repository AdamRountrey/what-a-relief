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
#include <type_traits>
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

cv::Mat definiteSaturationMask(const cv::Mat& raw) {
    double whiteLevel = 0.0;
    if (raw.depth() == CV_8U) {
        whiteLevel = 255.0;
    } else if (raw.depth() == CV_16U) {
        whiteLevel = 65535.0;
    } else {
        return cv::Mat(raw.size(), CV_8U, cv::Scalar(0));
    }

    std::vector<cv::Mat> channels;
    cv::split(raw, channels);
    cv::Mat saturated(raw.size(), CV_8U, cv::Scalar(0));
    for (const cv::Mat& channel : channels) {
        cv::Mat channelSaturated;
        cv::compare(channel, whiteLevel, channelSaturated, cv::CMP_GE);
        cv::bitwise_or(saturated, channelSaturated, saturated);
    }
    return saturated;
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
    std::vector<double> rowMinimums(static_cast<size_t>(src.rows), std::numeric_limits<double>::infinity());
    std::vector<double> rowMaximums(static_cast<size_t>(src.rows), -std::numeric_limits<double>::infinity());
    cv::parallel_for_(cv::Range(0, src.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const float* row = src.ptr<float>(y);
            const uchar* mrow = mask.ptr<uchar>(y);
            double rowMinimum = std::numeric_limits<double>::infinity();
            double rowMaximum = -std::numeric_limits<double>::infinity();
            for (int x = 0; x < src.cols; ++x) {
                if (mrow[x] == 0 || !std::isfinite(row[x])) {
                    continue;
                }
                rowMinimum = std::min(rowMinimum, static_cast<double>(row[x]));
                rowMaximum = std::max(rowMaximum, static_cast<double>(row[x]));
            }
            rowMinimums[static_cast<size_t>(y)] = rowMinimum;
            rowMaximums[static_cast<size_t>(y)] = rowMaximum;
        }
    });

    minValue = std::numeric_limits<double>::infinity();
    maxValue = -std::numeric_limits<double>::infinity();
    for (int y = 0; y < src.rows; ++y) {
        minValue = std::min(minValue, rowMinimums[static_cast<size_t>(y)]);
        maxValue = std::max(maxValue, rowMaximums[static_cast<size_t>(y)]);
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
    cv::parallel_for_(cv::Range(0, src.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
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
    });
    return out;
}

cv::Mat normalComponentTo8U(const cv::Mat& normalMap, const cv::Mat& mask, int component, bool signedComponent) {
    cv::Mat out(normalMap.rows, normalMap.cols, CV_8U, cv::Scalar(0));
    cv::parallel_for_(cv::Range(0, normalMap.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
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
    });
    return out;
}

cv::Mat hillshadeTo8U(const cv::Mat& normalMap, const cv::Mat& mask, const cv::Vec3f& lightDirection) {
    const float length = std::sqrt(lightDirection.dot(lightDirection));
    const cv::Vec3f light = length > 1.0e-6f ? lightDirection / length : cv::Vec3f(-0.5f, 0.5f, 0.70710678f);
    constexpr double kAmbient = 0.16;
    constexpr double kDiffuse = 0.82;
    constexpr double kGamma = 0.78;
    cv::Mat out(normalMap.rows, normalMap.cols, CV_8U, cv::Scalar(0));
    cv::parallel_for_(cv::Range(0, normalMap.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
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
    });
    return out;
}

cv::Mat normalRgbTo8U(const cv::Mat& normalMap, const cv::Mat& mask) {
    cv::Mat out(normalMap.rows, normalMap.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::parallel_for_(cv::Range(0, normalMap.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
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
    });
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

cv::Mat signedFloatTo8U(const cv::Mat& src, const cv::Mat& mask) {
    const float extent = maskedAbsPercentile(src, mask, 0.995f);
    cv::Mat out(src.size(), CV_8U, cv::Scalar(0));
    cv::parallel_for_(cv::Range(0, src.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const float* srcRow = src.ptr<float>(y);
            const uchar* maskRow = mask.ptr<uchar>(y);
            uchar* outputRow = out.ptr<uchar>(y);
            for (int x = 0; x < src.cols; ++x) {
                if (maskRow[x] == 0 || !std::isfinite(srcRow[x])) {
                    continue;
                }
                const float normalized = std::clamp(srcRow[x] / extent, -1.0f, 1.0f);
                outputRow[x] = static_cast<uchar>(std::lround(127.5f * (normalized + 1.0f)));
            }
        }
    });
    return out;
}

cv::Mat maskedGaussianBlur(const cv::Mat& src, const cv::Mat& mask, double sigma) {
    cv::Mat maskFloat;
    mask.convertTo(maskFloat, CV_32F, 1.0 / 255.0);

    cv::Mat weighted(src.size(), CV_32F, cv::Scalar(0));
    cv::parallel_for_(cv::Range(0, src.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const float* srcRow = src.ptr<float>(y);
            const float* maskRow = maskFloat.ptr<float>(y);
            float* weightedRow = weighted.ptr<float>(y);
            for (int x = 0; x < src.cols; ++x) {
                weightedRow[x] = std::isfinite(srcRow[x]) ? srcRow[x] * maskRow[x] : 0.0f;
            }
        }
    });

    cv::Mat numerator;
    cv::Mat denominator;
    cv::GaussianBlur(weighted, numerator, cv::Size(0, 0), sigma);
    cv::GaussianBlur(maskFloat, denominator, cv::Size(0, 0), sigma);

    cv::Mat out(src.size(), CV_32F, cv::Scalar(0));
    cv::parallel_for_(cv::Range(0, src.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const float* numRow = numerator.ptr<float>(y);
            const float* denRow = denominator.ptr<float>(y);
            float* outRow = out.ptr<float>(y);
            for (int x = 0; x < src.cols; ++x) {
                outRow[x] = denRow[x] > 1.0e-6f ? numRow[x] / denRow[x] : 0.0f;
            }
        }
    });
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

std::pair<float, float> maskedColorLuminanceRange(
    const cv::Mat& image,
    const cv::Mat& mask,
    float lowPercentile,
    float highPercentile) {
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
        return {0.0f, 0.0f};
    }
    const size_t lowIndex = static_cast<size_t>(
        std::clamp(lowPercentile, 0.0f, 1.0f) * static_cast<float>(values.size() - 1));
    const size_t highIndex = static_cast<size_t>(
        std::clamp(highPercentile, 0.0f, 1.0f) * static_cast<float>(values.size() - 1));
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(lowIndex), values.end());
    const float low = values[lowIndex];
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(highIndex), values.end());
    return {low, values[highIndex]};
}

cv::Mat stretchColorByMaskedLuminance(const cv::Mat& image, const cv::Mat& mask) {
    const auto [low, high] = maskedColorLuminanceRange(image, mask, 0.01f, 0.992f);
    if (high - low < 8.0f) {
        return image;
    }

    const float outLow = 6.0f;
    const float outHigh = 248.0f;
    const float scale = (outHigh - outLow) / (high - low);
    cv::Mat stretched = image.clone();
    cv::parallel_for_(cv::Range(0, image.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
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
    });
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
    cv::parallel_for_(cv::Range(0, normalMap.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
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
    });
    return stretchColorByMaskedLuminance(out, mask);
}

void writePfm(const fs::path& path, const cv::Mat& image, const cv::Mat& mask) {
    CheckedOutputFile checked(path);
    std::ostream& out = checked.stream();
    out << "Pf\n" << image.cols << " " << image.rows << "\n-1.0\n";
    std::vector<float> outputRow(static_cast<size_t>(image.cols));
    for (int y = image.rows - 1; y >= 0; --y) {
        const float* row = image.ptr<float>(y);
        const uchar* mrow = mask.ptr<uchar>(y);
        for (int x = 0; x < image.cols; ++x) {
            outputRow[static_cast<size_t>(x)] =
                (mrow[x] != 0 && std::isfinite(row[x])) ? row[x] : 0.0f;
        }
        out.write(
            reinterpret_cast<const char*>(outputRow.data()),
            static_cast<std::streamsize>(outputRow.size() * sizeof(float)));
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
        out << "ring_light_led_diameter_mm," << opt.shadowLedDiameterMm << "\n";
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

class PlyBinaryWriter {
public:
    explicit PlyBinaryWriter(std::ostream& out)
        : out_(out), buffer_(4 * 1024 * 1024) {}

    void writeVertex(float x, float y, float z, std::uint8_t color) {
        ensure(sizeof(float) * 3 + 3);
        appendUnchecked(x);
        appendUnchecked(y);
        appendUnchecked(z);
        buffer_[used_++] = static_cast<char>(color);
        buffer_[used_++] = static_cast<char>(color);
        buffer_[used_++] = static_cast<char>(color);
    }

    void writeTriangle(std::int32_t a, std::int32_t b, std::int32_t c) {
        ensure(1 + sizeof(std::int32_t) * 3);
        buffer_[used_++] = 3;
        appendUnchecked(a);
        appendUnchecked(b);
        appendUnchecked(c);
    }

    void writeQuad(std::int32_t a, std::int32_t b, std::int32_t c, std::int32_t d) {
        ensure(1 + sizeof(std::int32_t) * 4);
        buffer_[used_++] = 4;
        appendUnchecked(a);
        appendUnchecked(b);
        appendUnchecked(c);
        appendUnchecked(d);
    }

    void flush() {
        if (used_ == 0) {
            return;
        }
        out_.write(buffer_.data(), static_cast<std::streamsize>(used_));
        used_ = 0;
    }

private:
    template <typename T>
    void appendUnchecked(const T& value) {
        static_assert(std::is_trivially_copyable<T>::value, "PLY fields must be trivially copyable");
        std::memcpy(buffer_.data() + used_, &value, sizeof(T));
        used_ += sizeof(T);
    }

    void ensure(size_t bytes) {
        if (used_ + bytes > buffer_.size()) {
            flush();
        }
    }

    std::ostream& out_;
    std::vector<char> buffer_;
    size_t used_ = 0;
};

struct PlyGridTopology {
    int step = 1;
    int sampleRows = 0;
    int sampleCols = 0;
    std::int32_t vertexCount = 0;
    std::uint64_t topFaceCount = 0;
    std::uint64_t boundaryFaceCount = 0;
    double minimumHeight = std::numeric_limits<double>::infinity();
    double maximumHeight = -std::numeric_limits<double>::infinity();
    std::vector<std::int32_t> indices;
    std::vector<std::int8_t> horizontalEdgeBalance;
    std::vector<std::int8_t> verticalEdgeBalance;
    std::vector<std::int8_t> diagonalEdgeBalance;

    std::int32_t index(int row, int col) const {
        return indices[static_cast<size_t>(row) * static_cast<size_t>(sampleCols) + static_cast<size_t>(col)];
    }
};

void addEdgeBalance(std::int8_t& balance, int contribution) {
    balance = static_cast<std::int8_t>(static_cast<int>(balance) + contribution);
}

PlyGridTopology buildPlyGridTopology(
    const cv::Mat& height,
    const cv::Mat& mask,
    int step,
    bool includeBoundary,
    const std::function<void(const std::string&)>& progress) {
    if (height.empty() || height.type() != CV_32F || mask.type() != CV_8U || mask.size() != height.size()) {
        die("PLY height and mask inputs must be nonempty, matching single-channel images.");
    }
    if (step < 1) {
        die("PLY mesh step must be at least 1.");
    }

    PlyGridTopology topology;
    topology.step = step;
    topology.sampleRows = (height.rows - 1) / step + 1;
    topology.sampleCols = (height.cols - 1) / step + 1;
    const size_t sampleCount =
        static_cast<size_t>(topology.sampleRows) * static_cast<size_t>(topology.sampleCols);
    topology.indices.assign(sampleCount, -1);

    reportProgress(progress, "PLY: indexing sampled vertices...");
    for (int row = 0; row < topology.sampleRows; ++row) {
        const int y = row * step;
        const float* heightRow = height.ptr<float>(y);
        const uchar* maskRow = mask.ptr<uchar>(y);
        for (int col = 0; col < topology.sampleCols; ++col) {
            const int x = col * step;
            if (maskRow[x] == 0 || !std::isfinite(heightRow[x])) {
                continue;
            }
            if (topology.vertexCount == std::numeric_limits<std::int32_t>::max()) {
                die("PLY has too many vertices for 32-bit face indices. Increase the mesh step.");
            }
            topology.indices[static_cast<size_t>(row) * static_cast<size_t>(topology.sampleCols) +
                static_cast<size_t>(col)] = topology.vertexCount++;
            topology.minimumHeight = std::min(topology.minimumHeight, static_cast<double>(heightRow[x]));
            topology.maximumHeight = std::max(topology.maximumHeight, static_cast<double>(heightRow[x]));
        }
    }

    if (includeBoundary) {
        topology.horizontalEdgeBalance.assign(
            static_cast<size_t>(topology.sampleRows) *
                static_cast<size_t>(std::max(0, topology.sampleCols - 1)),
            0);
        topology.verticalEdgeBalance.assign(
            static_cast<size_t>(std::max(0, topology.sampleRows - 1)) *
                static_cast<size_t>(topology.sampleCols),
            0);
        topology.diagonalEdgeBalance.assign(
            static_cast<size_t>(std::max(0, topology.sampleRows - 1)) *
                static_cast<size_t>(std::max(0, topology.sampleCols - 1)),
            0);
    }

    reportProgress(progress, includeBoundary ? "PLY: building shared faces and printable boundary..." : "PLY: counting faces...");
    const int cellCols = std::max(0, topology.sampleCols - 1);
    for (int row = 0; row + 1 < topology.sampleRows; ++row) {
        for (int col = 0; col + 1 < topology.sampleCols; ++col) {
            const std::int32_t a = topology.index(row, col);
            const std::int32_t b = topology.index(row, col + 1);
            const std::int32_t c = topology.index(row + 1, col);
            const std::int32_t d = topology.index(row + 1, col + 1);
            if (a >= 0 && b >= 0 && c >= 0) {
                ++topology.topFaceCount;
                if (includeBoundary) {
                    addEdgeBalance(topology.verticalEdgeBalance[
                        static_cast<size_t>(row) * static_cast<size_t>(topology.sampleCols) +
                        static_cast<size_t>(col)], 1);
                    addEdgeBalance(topology.diagonalEdgeBalance[
                        static_cast<size_t>(row) * static_cast<size_t>(cellCols) +
                        static_cast<size_t>(col)], -1);
                    addEdgeBalance(topology.horizontalEdgeBalance[
                        static_cast<size_t>(row) * static_cast<size_t>(cellCols) +
                        static_cast<size_t>(col)], -1);
                }
            }
            if (b >= 0 && d >= 0 && c >= 0) {
                ++topology.topFaceCount;
                if (includeBoundary) {
                    addEdgeBalance(topology.diagonalEdgeBalance[
                        static_cast<size_t>(row) * static_cast<size_t>(cellCols) +
                        static_cast<size_t>(col)], 1);
                    addEdgeBalance(topology.horizontalEdgeBalance[
                        static_cast<size_t>(row + 1) * static_cast<size_t>(cellCols) +
                        static_cast<size_t>(col)], 1);
                    addEdgeBalance(topology.verticalEdgeBalance[
                        static_cast<size_t>(row) * static_cast<size_t>(topology.sampleCols) +
                        static_cast<size_t>(col + 1)], -1);
                }
            }
        }
    }
    if (includeBoundary) {
        const auto countBoundary = [](const std::vector<std::int8_t>& balances) {
            return static_cast<std::uint64_t>(std::count_if(
                balances.begin(), balances.end(), [](std::int8_t value) { return value != 0; }));
        };
        topology.boundaryFaceCount =
            countBoundary(topology.horizontalEdgeBalance) +
            countBoundary(topology.verticalEdgeBalance) +
            countBoundary(topology.diagonalEdgeBalance);
    }
    return topology;
}

struct PrintableSurfaceFill {
    cv::Mat height;
    cv::Mat mask;
    cv::Mat vertexColor;
    cv::Mat fillMask;
    size_t filledPixelCount = 0;
};

cv::Mat finiteGeometryMask(const cv::Mat& height, const cv::Mat& mask) {
    cv::Mat finiteMask(mask.size(), CV_8U, cv::Scalar(0));
    cv::parallel_for_(cv::Range(0, mask.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const float* hrow = height.ptr<float>(y);
            const uchar* mrow = mask.ptr<uchar>(y);
            uchar* out = finiteMask.ptr<uchar>(y);
            for (int x = 0; x < mask.cols; ++x) {
                out[x] = mrow[x] != 0 && std::isfinite(hrow[x]) ? 255 : 0;
            }
        }
    });
    return finiteMask;
}

cv::Mat enclosedMaskHoles(const cv::Mat& validMask) {
    cv::Mat exterior;
    cv::compare(validMask, 0, exterior, cv::CMP_EQ);
    if (exterior.empty()) {
        return exterior;
    }

    const auto floodExterior = [&](int x, int y) {
        if (exterior.at<uchar>(y, x) == 255) {
            cv::floodFill(
                exterior,
                cv::Point(x, y),
                cv::Scalar(128),
                nullptr,
                cv::Scalar(),
                cv::Scalar(),
                8);
        }
    };
    for (int x = 0; x < exterior.cols; ++x) {
        floodExterior(x, 0);
        if (exterior.rows > 1) {
            floodExterior(x, exterior.rows - 1);
        }
    }
    for (int y = 1; y + 1 < exterior.rows; ++y) {
        floodExterior(0, y);
        if (exterior.cols > 1) {
            floodExterior(exterior.cols - 1, y);
        }
    }

    cv::Mat holes;
    cv::compare(exterior, 255, holes, cv::CMP_EQ);
    return holes;
}

std::vector<cv::Point> maskPoints(const cv::Mat& mask) {
    std::vector<cv::Point> points;
    points.reserve(static_cast<size_t>(cv::countNonZero(mask)));
    for (int y = 0; y < mask.rows; ++y) {
        const uchar* row = mask.ptr<uchar>(y);
        for (int x = 0; x < mask.cols; ++x) {
            if (row[x] != 0) {
                points.emplace_back(x, y);
            }
        }
    }
    return points;
}

bool maskContains(const cv::Mat& mask, int x, int y) {
    return x >= 0 && y >= 0 && x < mask.cols && y < mask.rows && mask.at<uchar>(y, x) != 0;
}

void initializeHarmonicField(
    cv::Mat& field,
    const cv::Mat& sourceMask,
    const cv::Mat& domainMask,
    const std::vector<cv::Point>& holes) {
    cv::Mat known = sourceMask.clone();
    cv::Mat queued(sourceMask.size(), CV_8U, cv::Scalar(0));
    std::vector<cv::Point> queue;
    queue.reserve(holes.size());

    const auto hasKnownNeighbor = [&](const cv::Point& point) {
        return maskContains(known, point.x - 1, point.y) ||
            maskContains(known, point.x + 1, point.y) ||
            maskContains(known, point.x, point.y - 1) ||
            maskContains(known, point.x, point.y + 1);
    };
    for (const cv::Point& point : holes) {
        if (hasKnownNeighbor(point)) {
            queue.push_back(point);
            queued.at<uchar>(point) = 255;
        }
    }

    const int dx[4] = {-1, 1, 0, 0};
    const int dy[4] = {0, 0, -1, 1};
    size_t head = 0;
    while (head < queue.size()) {
        const cv::Point point = queue[head++];
        float sum = 0.0f;
        int count = 0;
        for (int direction = 0; direction < 4; ++direction) {
            const int nx = point.x + dx[direction];
            const int ny = point.y + dy[direction];
            if (maskContains(known, nx, ny)) {
                sum += field.at<float>(ny, nx);
                ++count;
            }
        }
        if (count == 0) {
            continue;
        }
        field.at<float>(point) = sum / static_cast<float>(count);
        known.at<uchar>(point) = 255;

        for (int direction = 0; direction < 4; ++direction) {
            const int nx = point.x + dx[direction];
            const int ny = point.y + dy[direction];
            if (maskContains(domainMask, nx, ny) && !maskContains(known, nx, ny) &&
                queued.at<uchar>(ny, nx) == 0) {
                queue.emplace_back(nx, ny);
                queued.at<uchar>(ny, nx) = 255;
            }
        }
    }
}

void relaxHarmonicField(
    cv::Mat& field,
    const cv::Mat& domainMask,
    const std::vector<cv::Point>& holes,
    int iterations) {
    for (int iteration = 0; iteration < iterations; ++iteration) {
        float maximumChange = 0.0f;
        for (int parity = 0; parity < 2; ++parity) {
            for (const cv::Point& point : holes) {
                if (((point.x + point.y) & 1) != parity) {
                    continue;
                }
                float sum = 0.0f;
                int count = 0;
                const int dx[4] = {-1, 1, 0, 0};
                const int dy[4] = {0, 0, -1, 1};
                for (int direction = 0; direction < 4; ++direction) {
                    const int nx = point.x + dx[direction];
                    const int ny = point.y + dy[direction];
                    if (maskContains(domainMask, nx, ny)) {
                        sum += field.at<float>(ny, nx);
                        ++count;
                    }
                }
                if (count > 0) {
                    const float candidate = sum / static_cast<float>(count);
                    float& value = field.at<float>(point);
                    maximumChange = std::max(maximumChange, std::abs(candidate - value));
                    value = candidate;
                }
            }
        }
        if (maximumChange < 1.0e-5f) {
            break;
        }
    }
}

void estimateSurfaceSlopes(
    const cv::Mat& height,
    const cv::Mat& sourceMask,
    const cv::Mat& normalMap,
    double slopeCap,
    cv::Mat& p,
    cv::Mat& q) {
    p = cv::Mat(height.size(), CV_32F, cv::Scalar(0));
    q = cv::Mat(height.size(), CV_32F, cv::Scalar(0));
    const bool normalsAvailable = normalMap.type() == CV_32FC3 && normalMap.size() == height.size();

    cv::parallel_for_(cv::Range(0, height.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const float* hrow = height.ptr<float>(y);
            float* prow = p.ptr<float>(y);
            float* qrow = q.ptr<float>(y);
            for (int x = 0; x < height.cols; ++x) {
                if (!maskContains(sourceMask, x, y)) {
                    continue;
                }

                bool haveP = false;
                bool haveQ = false;
                if (maskContains(sourceMask, x - 1, y) && maskContains(sourceMask, x + 1, y)) {
                    prow[x] = 0.5f * (hrow[x + 1] - hrow[x - 1]);
                    haveP = true;
                } else if (maskContains(sourceMask, x + 1, y)) {
                    prow[x] = hrow[x + 1] - hrow[x];
                    haveP = true;
                } else if (maskContains(sourceMask, x - 1, y)) {
                    prow[x] = hrow[x] - hrow[x - 1];
                    haveP = true;
                }

                if (maskContains(sourceMask, x, y - 1) && maskContains(sourceMask, x, y + 1)) {
                    qrow[x] = 0.5f * (height.at<float>(y + 1, x) - height.at<float>(y - 1, x));
                    haveQ = true;
                } else if (maskContains(sourceMask, x, y + 1)) {
                    qrow[x] = height.at<float>(y + 1, x) - hrow[x];
                    haveQ = true;
                } else if (maskContains(sourceMask, x, y - 1)) {
                    qrow[x] = hrow[x] - height.at<float>(y - 1, x);
                    haveQ = true;
                }

                if ((!haveP || !haveQ) && normalsAvailable) {
                    const cv::Vec3f normal = normalMap.at<cv::Vec3f>(y, x);
                    if (std::isfinite(normal[0]) && std::isfinite(normal[1]) &&
                        std::isfinite(normal[2]) && normal[2] > 1.0e-4f) {
                        if (!haveP) {
                            prow[x] = -normal[0] / normal[2];
                        }
                        if (!haveQ) {
                            qrow[x] = normal[1] / normal[2];
                        }
                    }
                }

                if (slopeCap > 0.0) {
                    const float magnitude = std::sqrt(prow[x] * prow[x] + qrow[x] * qrow[x]);
                    if (magnitude > static_cast<float>(slopeCap)) {
                        const float scale = static_cast<float>(slopeCap) / magnitude;
                        prow[x] *= scale;
                        qrow[x] *= scale;
                    }
                }
            }
        }
    });
}

float slopeAdjustedNeighborHeight(
    const cv::Mat& height,
    const cv::Mat& p,
    const cv::Mat& q,
    const cv::Point& point,
    int nx,
    int ny) {
    const float neighborHeight = height.at<float>(ny, nx);
    if (nx < point.x) {
        return neighborHeight + 0.5f * (p.at<float>(ny, nx) + p.at<float>(point));
    }
    if (nx > point.x) {
        return neighborHeight - 0.5f * (p.at<float>(ny, nx) + p.at<float>(point));
    }
    if (ny < point.y) {
        return neighborHeight + 0.5f * (q.at<float>(ny, nx) + q.at<float>(point));
    }
    return neighborHeight - 0.5f * (q.at<float>(ny, nx) + q.at<float>(point));
}

void fillHeightFromSlopes(
    cv::Mat& height,
    const cv::Mat& sourceMask,
    const cv::Mat& domainMask,
    const cv::Mat& p,
    const cv::Mat& q,
    const std::vector<cv::Point>& holes) {
    cv::Mat known = sourceMask.clone();
    cv::Mat queued(sourceMask.size(), CV_8U, cv::Scalar(0));
    std::vector<cv::Point> queue;
    queue.reserve(holes.size());
    const int dx[4] = {-1, 1, 0, 0};
    const int dy[4] = {0, 0, -1, 1};

    for (const cv::Point& point : holes) {
        for (int direction = 0; direction < 4; ++direction) {
            if (maskContains(known, point.x + dx[direction], point.y + dy[direction])) {
                queue.push_back(point);
                queued.at<uchar>(point) = 255;
                break;
            }
        }
    }

    size_t head = 0;
    while (head < queue.size()) {
        const cv::Point point = queue[head++];
        float sum = 0.0f;
        int count = 0;
        for (int direction = 0; direction < 4; ++direction) {
            const int nx = point.x + dx[direction];
            const int ny = point.y + dy[direction];
            if (maskContains(known, nx, ny)) {
                sum += slopeAdjustedNeighborHeight(height, p, q, point, nx, ny);
                ++count;
            }
        }
        if (count == 0) {
            continue;
        }
        height.at<float>(point) = sum / static_cast<float>(count);
        known.at<uchar>(point) = 255;
        for (int direction = 0; direction < 4; ++direction) {
            const int nx = point.x + dx[direction];
            const int ny = point.y + dy[direction];
            if (maskContains(domainMask, nx, ny) && !maskContains(known, nx, ny) &&
                queued.at<uchar>(ny, nx) == 0) {
                queue.emplace_back(nx, ny);
                queued.at<uchar>(ny, nx) = 255;
            }
        }
    }

    double minimumHeight = 0.0;
    double maximumHeight = 0.0;
    cv::minMaxLoc(height, &minimumHeight, &maximumHeight, nullptr, nullptr, sourceMask);
    const float tolerance = static_cast<float>(
        std::max(1.0e-5, (maximumHeight - minimumHeight) * 1.0e-6));
    int iterations = 180;
    if (holes.size() > 2000000) {
        iterations = 60;
    } else if (holes.size() > 500000) {
        iterations = 100;
    }
    for (int iteration = 0; iteration < iterations; ++iteration) {
        float maximumChange = 0.0f;
        for (int parity = 0; parity < 2; ++parity) {
            for (const cv::Point& point : holes) {
                if (((point.x + point.y) & 1) != parity) {
                    continue;
                }
                float sum = 0.0f;
                int count = 0;
                for (int direction = 0; direction < 4; ++direction) {
                    const int nx = point.x + dx[direction];
                    const int ny = point.y + dy[direction];
                    if (maskContains(domainMask, nx, ny)) {
                        sum += slopeAdjustedNeighborHeight(height, p, q, point, nx, ny);
                        ++count;
                    }
                }
                if (count > 0) {
                    const float candidate = sum / static_cast<float>(count);
                    float& value = height.at<float>(point);
                    const float updated = value + 1.25f * (candidate - value);
                    maximumChange = std::max(maximumChange, std::abs(updated - value));
                    value = updated;
                }
            }
        }
        if (maximumChange < tolerance) {
            break;
        }
    }
}

PrintableSurfaceFill fillPrintableSurfaceHoles(
    const cv::Mat& height,
    const cv::Mat& geometryMask,
    const cv::Mat& normalMap,
    const cv::Mat& vertexColor,
    double slopeCap,
    const std::function<void(const std::string&)>& progress) {
    PrintableSurfaceFill result;
    const cv::Mat sourceMask = finiteGeometryMask(height, geometryMask);
    result.fillMask = enclosedMaskHoles(sourceMask);
    result.filledPixelCount = static_cast<size_t>(cv::countNonZero(result.fillMask));
    result.height = height.clone();
    result.mask = sourceMask.clone();
    result.vertexColor = vertexColor.clone();
    if (result.filledPixelCount == 0) {
        reportProgress(progress, "Printable PLY: no enclosed surface holes found.");
        return result;
    }

    reportProgress(
        progress,
        "Printable PLY: reconstructing " + std::to_string(result.filledPixelCount) +
            " enclosed surface pixels...");
    result.mask.setTo(cv::Scalar(255), result.fillMask);
    const std::vector<cv::Point> holes = maskPoints(result.fillMask);

    cv::Mat p;
    cv::Mat q;
    estimateSurfaceSlopes(height, sourceMask, normalMap, slopeCap, p, q);
    initializeHarmonicField(p, sourceMask, result.mask, holes);
    initializeHarmonicField(q, sourceMask, result.mask, holes);
    relaxHarmonicField(p, result.mask, holes, 60);
    relaxHarmonicField(q, result.mask, holes, 60);
    fillHeightFromSlopes(result.height, sourceMask, result.mask, p, q, holes);

    if (!result.vertexColor.empty() && result.vertexColor.type() == CV_8U &&
        result.vertexColor.size() == height.size()) {
        cv::Mat colorFloat;
        result.vertexColor.convertTo(colorFloat, CV_32F);
        initializeHarmonicField(colorFloat, sourceMask, result.mask, holes);
        relaxHarmonicField(colorFloat, result.mask, holes, 60);
        for (const cv::Point& point : holes) {
            result.vertexColor.at<uchar>(point) =
                cv::saturate_cast<uchar>(colorFloat.at<float>(point));
        }
    }
    return result;
}

void writePlyMesh(
    const fs::path& path,
    const cv::Mat& height,
    const cv::Mat& vertexColor,
    double heightScale,
    const PlyGridTopology& topology,
    const std::function<void(const std::string&)>& progress) {
    CheckedOutputFile checked(path);
    std::ostream& out = checked.stream();

    out << "ply\n";
    out << "format binary_little_endian 1.0\n";
    out << "comment generated by what-a-relief\n";
    out << "element vertex " << topology.vertexCount << "\n";
    out << "property float x\n";
    out << "property float y\n";
    out << "property float z\n";
    out << "property uchar red\n";
    out << "property uchar green\n";
    out << "property uchar blue\n";
    out << "element face " << topology.topFaceCount << "\n";
    out << "property list uchar int vertex_indices\n";
    out << "end_header\n";

    PlyBinaryWriter binary(out);
    reportProgress(progress, "PLY: writing binary vertices...");
    for (int row = 0; row < topology.sampleRows; ++row) {
        const int y = row * topology.step;
        const float* hrow = height.ptr<float>(y);
        const uchar* colorRow = vertexColor.empty() ? nullptr : vertexColor.ptr<uchar>(y);
        for (int col = 0; col < topology.sampleCols; ++col) {
            const int x = col * topology.step;
            if (topology.index(row, col) >= 0) {
                const float vx = static_cast<float>(x);
                const float vy = static_cast<float>(-y);
                const float vz = static_cast<float>(static_cast<double>(hrow[x]) * heightScale);
                const std::uint8_t color = colorRow == nullptr ? 200 : colorRow[x];
                binary.writeVertex(vx, vy, vz, color);
            }
        }
    }

    reportProgress(progress, "PLY: writing binary faces...");
    for (int row = 0; row + 1 < topology.sampleRows; ++row) {
        for (int col = 0; col + 1 < topology.sampleCols; ++col) {
            const std::int32_t a = topology.index(row, col);
            const std::int32_t b = topology.index(row, col + 1);
            const std::int32_t c = topology.index(row + 1, col);
            const std::int32_t d = topology.index(row + 1, col + 1);
            if (a >= 0 && b >= 0 && c >= 0) {
                binary.writeTriangle(a, c, b);
            }
            if (b >= 0 && d >= 0 && c >= 0) {
                binary.writeTriangle(b, c, d);
            }
        }
    }
    binary.flush();

    if (!out) {
        die("Failed while writing PLY mesh: " + path.string());
    }
    checked.commit();
    reportProgress(
        progress,
        "PLY: done (" + std::to_string(topology.vertexCount) + " vertices, " +
            std::to_string(topology.topFaceCount) + " faces).");
}

void writeBoundaryQuad(
    PlyBinaryWriter& binary,
    std::int8_t balance,
    std::int32_t canonicalStart,
    std::int32_t canonicalEnd,
    std::int32_t bottomOffset) {
    if (balance > 0) {
        binary.writeQuad(
            canonicalEnd,
            canonicalStart,
            canonicalStart + bottomOffset,
            canonicalEnd + bottomOffset);
    } else if (balance < 0) {
        binary.writeQuad(
            canonicalStart,
            canonicalEnd,
            canonicalEnd + bottomOffset,
            canonicalStart + bottomOffset);
    }
}

void writePrintablePlyMesh(
    const fs::path& path,
    const cv::Mat& height,
    const cv::Mat& vertexColor,
    double heightScale,
    double pixelScaleMm,
    double baseThicknessMm,
    const PlyGridTopology& topology,
    const std::function<void(const std::string&)>& progress) {
    const double xyScale = pixelScaleMm > 0.0 ? pixelScaleMm : 1.0;
    const double zScale = xyScale * heightScale;
    const double baseThickness = pixelScaleMm > 0.0 ? baseThicknessMm : baseThicknessMm / xyScale;
    if (topology.vertexCount < 3 ||
        !std::isfinite(topology.minimumHeight) || !std::isfinite(topology.maximumHeight)) {
        die("Printable PLY needs at least three finite height pixels inside the geometry mask.");
    }
    if (topology.topFaceCount == 0) {
        die("Printable PLY has no surface faces. Use a smaller mesh step or a larger height mask.");
    }
    if (topology.vertexCount > std::numeric_limits<std::int32_t>::max() / 2) {
        die("Printable PLY has too many vertices for 32-bit face indices. Increase the mesh step.");
    }
    const std::int32_t topVertexCount = topology.vertexCount;
    const std::int64_t vertexCount = static_cast<std::int64_t>(topVertexCount) * 2;
    const std::uint64_t faceCount = topology.topFaceCount * 2 + topology.boundaryFaceCount;
    const double minZ = zScale >= 0.0
        ? topology.minimumHeight * zScale
        : topology.maximumHeight * zScale;
    const double baseZ = minZ - std::max(0.0, baseThickness);

    CheckedOutputFile checked(path);
    std::ostream& out = checked.stream();

    out << "ply\n";
    out << "format binary_little_endian 1.0\n";
    out << "comment generated by what-a-relief printable solid export\n";
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

    PlyBinaryWriter binary(out);
    reportProgress(progress, "Printable PLY: writing vertices...");
    for (int layer = 0; layer < 2; ++layer) {
        for (int row = 0; row < topology.sampleRows; ++row) {
            const int y = row * topology.step;
            const float* hrow = height.ptr<float>(y);
            const uchar* colorRow = vertexColor.empty() ? nullptr : vertexColor.ptr<uchar>(y);
            for (int col = 0; col < topology.sampleCols; ++col) {
                const int x = col * topology.step;
                if (topology.index(row, col) >= 0) {
                    const float vx = static_cast<float>(static_cast<double>(x) * xyScale);
                    const float vy = static_cast<float>(static_cast<double>(-y) * xyScale);
                    const float vz = layer == 0
                        ? static_cast<float>(static_cast<double>(hrow[x]) * zScale)
                        : static_cast<float>(baseZ);
                    const std::uint8_t color = colorRow == nullptr ? 200 : colorRow[x];
                    binary.writeVertex(vx, vy, vz, color);
                }
            }
        }
    }

    reportProgress(progress, "Printable PLY: writing closed faces...");
    for (int row = 0; row + 1 < topology.sampleRows; ++row) {
        for (int col = 0; col + 1 < topology.sampleCols; ++col) {
            const std::int32_t a = topology.index(row, col);
            const std::int32_t b = topology.index(row, col + 1);
            const std::int32_t c = topology.index(row + 1, col);
            const std::int32_t d = topology.index(row + 1, col + 1);
            if (a >= 0 && b >= 0 && c >= 0) {
                binary.writeTriangle(a, c, b);
                binary.writeTriangle(b + topVertexCount, c + topVertexCount, a + topVertexCount);
            }
            if (b >= 0 && d >= 0 && c >= 0) {
                binary.writeTriangle(b, c, d);
                binary.writeTriangle(d + topVertexCount, c + topVertexCount, b + topVertexCount);
            }
        }
    }

    const int cellCols = std::max(0, topology.sampleCols - 1);
    for (int row = 0; row < topology.sampleRows; ++row) {
        for (int col = 0; col + 1 < topology.sampleCols; ++col) {
            const std::int8_t balance = topology.horizontalEdgeBalance[
                static_cast<size_t>(row) * static_cast<size_t>(cellCols) + static_cast<size_t>(col)];
            writeBoundaryQuad(
                binary,
                balance,
                topology.index(row, col),
                topology.index(row, col + 1),
                topVertexCount);
        }
    }
    for (int row = 0; row + 1 < topology.sampleRows; ++row) {
        for (int col = 0; col < topology.sampleCols; ++col) {
            const std::int8_t balance = topology.verticalEdgeBalance[
                static_cast<size_t>(row) * static_cast<size_t>(topology.sampleCols) + static_cast<size_t>(col)];
            writeBoundaryQuad(
                binary,
                balance,
                topology.index(row, col),
                topology.index(row + 1, col),
                topVertexCount);
        }
    }
    for (int row = 0; row + 1 < topology.sampleRows; ++row) {
        for (int col = 0; col + 1 < topology.sampleCols; ++col) {
            const std::int8_t balance = topology.diagonalEdgeBalance[
                static_cast<size_t>(row) * static_cast<size_t>(cellCols) + static_cast<size_t>(col)];
            writeBoundaryQuad(
                binary,
                balance,
                topology.index(row, col + 1),
                topology.index(row + 1, col),
                topVertexCount);
        }
    }
    binary.flush();

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

std::vector<cv::Mat> loadLuminanceImages(
    const std::vector<std::string>& paths,
    bool srgb,
    std::vector<cv::Mat>* saturationMasks) {
    std::vector<cv::Mat> images;
    images.reserve(paths.size());
    if (saturationMasks != nullptr) {
        saturationMasks->clear();
        saturationMasks->reserve(paths.size());
    }
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
        if (saturationMasks != nullptr) {
            saturationMasks->push_back(definiteSaturationMask(raw));
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
        if (!diagnostics.unsupportedMask.empty()) {
            writeImageChecked(outDir / "robust_unsupported_mask.png", diagnostics.unsupportedMask);
        }
        if (!diagnostics.effectiveInlierCount.empty()) {
            cv::Mat inlierCount8;
            diagnostics.effectiveInlierCount.convertTo(
                inlierCount8,
                CV_8U,
                255.0 / static_cast<double>(std::max<size_t>(1, lights.size())));
            writeImageChecked(outDir / "robust_inlier_count.png", inlierCount8);
        }
        if (!diagnostics.localConditionNumber.empty()) {
            cv::Mat condition8;
            cv::min(diagnostics.localConditionNumber, 100.0f, condition8);
            condition8.convertTo(condition8, CV_8U, 2.55);
            writeImageChecked(outDir / "robust_local_condition.png", condition8);
        }
        writeImageChecked(outDir / "shadow_count.png", normalizeFloatTo8U(diagnostics.shadowCount, validMask, true));
        writeImageChecked(outDir / "highlight_outlier_count.png", normalizeFloatTo8U(diagnostics.highlightOutlierCount, validMask, true));
        writeImageChecked(outDir / "saturation_count.png", normalizeFloatTo8U(diagnostics.saturationCount, validMask, true));
        writeImageChecked(outDir / "model_mismatch_count.png", normalizeFloatTo8U(diagnostics.modelMismatchCount, validMask, true));
    }
    if (opt.specularDiagnostics && !diagnostics.specularCueMask.empty()) {
        writeImageChecked(outDir / "specular_cue_mask.png", diagnostics.specularCueMask);
        if (diagnostics.shadowObservationMasks.size() == lights.size() &&
            diagnostics.highlightObservationMasks.size() == lights.size() &&
            diagnostics.saturationObservationMasks.size() == lights.size()) {
            const fs::path observationDir = outDir / "robust_observations";
            fs::create_directories(observationDir);
            for (size_t i = 0; i < lights.size(); ++i) {
                std::ostringstream number;
                number << std::setw(3) << std::setfill('0') << (i + 1);
                writeImageChecked(
                    observationDir / ("light_" + number.str() + "_shadow.png"),
                    diagnostics.shadowObservationMasks[i]);
                writeImageChecked(
                    observationDir / ("light_" + number.str() + "_highlight.png"),
                    diagnostics.highlightObservationMasks[i]);
                writeImageChecked(
                    observationDir / ("light_" + number.str() + "_saturation.png"),
                    diagnostics.saturationObservationMasks[i]);
            }
        }
    }
    if (opt.shadowHeightRefinement && !diagnostics.shadowHeightCorrection.empty()) {
        reportProgress(progress, "Writing shadow-constrained height audit...");
        const cv::Mat& geometryMask = heightMask.empty() ? validMask : heightMask;
        writeImageChecked(
            outDir / "shadow_height_correction.png",
            signedFloatTo8U(diagnostics.shadowHeightCorrection, geometryMask));
        writePfm(
            outDir / "shadow_height_correction.pfm",
            diagnostics.shadowHeightCorrection,
            geometryMask);
        writeImageChecked(
            outDir / "shadow_constraint_count.png",
            normalizeFloatTo8U(diagnostics.shadowConstraintCount, geometryMask, true));
        writeImageChecked(
            outDir / "shadow_mismatch_before.png",
            normalizeFloatTo8U(diagnostics.shadowMismatchBefore, geometryMask, true));
        writeImageChecked(
            outDir / "shadow_mismatch_after.png",
            normalizeFloatTo8U(diagnostics.shadowMismatchAfter, geometryMask, true));
        if (!diagnostics.shadowObservability.empty()) {
            cv::Mat observability8;
            diagnostics.shadowObservability.convertTo(observability8, CV_8U, 255.0);
            writeImageChecked(outDir / "shadow_observability.png", observability8);
        }
        if (!diagnostics.shadowEdgeSupport.empty()) {
            writeImageChecked(
                outDir / "shadow_edge_support.png",
                normalizeFloatTo8U(diagnostics.shadowEdgeSupport, geometryMask, true));
        }
        if (!diagnostics.shadowOccluderSupport.empty()) {
            writeImageChecked(
                outDir / "shadow_occluder_support.png",
                diagnostics.shadowOccluderSupport);
        }

        const fs::path shadowDirectory = outDir / "shadow_refinement";
        fs::create_directories(shadowDirectory);
        for (const int index : diagnostics.shadowRefinementLightIndices) {
            if (index < 0 || static_cast<size_t>(index) >= lights.size()) {
                continue;
            }
            std::ostringstream number;
            number << std::setw(3) << std::setfill('0') << (index + 1);
            const size_t lightIndex = static_cast<size_t>(index);
            if (lightIndex < diagnostics.shadowObservedCastMasks.size() &&
                !diagnostics.shadowObservedCastMasks[lightIndex].empty()) {
                writeImageChecked(
                    shadowDirectory / ("light_" + number.str() + "_observed_cast.png"),
                    diagnostics.shadowObservedCastMasks[lightIndex]);
            }
            if (lightIndex < diagnostics.shadowObservationConfidence.size() &&
                !diagnostics.shadowObservationConfidence[lightIndex].empty()) {
                cv::Mat confidence8;
                diagnostics.shadowObservationConfidence[lightIndex].convertTo(
                    confidence8, CV_8U, 255.0);
                writeImageChecked(
                    shadowDirectory / ("light_" + number.str() + "_evidence_confidence.png"),
                    confidence8);
            }
            if (lightIndex < diagnostics.shadowPredictedBeforeMasks.size() &&
                !diagnostics.shadowPredictedBeforeMasks[lightIndex].empty()) {
                writeImageChecked(
                    shadowDirectory / ("light_" + number.str() + "_predicted_before.png"),
                    diagnostics.shadowPredictedBeforeMasks[lightIndex]);
            }
            if (lightIndex < diagnostics.shadowPredictedAfterMasks.size() &&
                !diagnostics.shadowPredictedAfterMasks[lightIndex].empty()) {
                writeImageChecked(
                    shadowDirectory / ("light_" + number.str() + "_predicted_after.png"),
                    diagnostics.shadowPredictedAfterMasks[lightIndex]);
            }
            if (lightIndex < diagnostics.shadowPredictedBeforeProbability.size() &&
                !diagnostics.shadowPredictedBeforeProbability[lightIndex].empty()) {
                cv::Mat probability8;
                diagnostics.shadowPredictedBeforeProbability[lightIndex].convertTo(
                    probability8, CV_8U, 255.0);
                writeImageChecked(
                    shadowDirectory / ("light_" + number.str() + "_probability_before.png"),
                    probability8);
            }
            if (lightIndex < diagnostics.shadowPredictedAfterProbability.size() &&
                !diagnostics.shadowPredictedAfterProbability[lightIndex].empty()) {
                cv::Mat probability8;
                diagnostics.shadowPredictedAfterProbability[lightIndex].convertTo(
                    probability8, CV_8U, 255.0);
                writeImageChecked(
                    shadowDirectory / ("light_" + number.str() + "_probability_after.png"),
                    probability8);
            }
        }
    }
    if (!height.empty()) {
        reportProgress(progress, "Writing height outputs...");
        const cv::Mat& geometryMask = heightMask.empty() ? validMask : heightMask;
        writeImageChecked(outDir / "height_mask.png", geometryMask);
        writeImageChecked(outDir / "height.png", normalizeFloatTo8U(height, geometryMask, false));
        writePfm(outDir / "height.pfm", height, geometryMask);
        if (!opt.meshPath.empty() || !opt.printableMeshPath.empty()) {
            if (opt.printableFillHoles && !opt.printableMeshPath.empty()) {
                if (!opt.meshPath.empty()) {
                    const PlyGridTopology openTopology = buildPlyGridTopology(
                        height,
                        geometryMask,
                        opt.meshStep,
                        false,
                        progress);
                    writePlyMesh(
                        opt.meshPath,
                        height,
                        albedo8,
                        opt.heightScale,
                        openTopology,
                        progress);
                }

                const PrintableSurfaceFill printable = fillPrintableSurfaceHoles(
                    height,
                    geometryMask,
                    normalMap,
                    albedo8,
                    opt.heightSlopeCap,
                    progress);
                writeImageChecked(outDir / "printable_fill_mask.png", printable.fillMask);
                const PlyGridTopology printableTopology = buildPlyGridTopology(
                    printable.height,
                    printable.mask,
                    opt.meshStep,
                    true,
                    progress);
                writePrintablePlyMesh(
                    opt.printableMeshPath,
                    printable.height,
                    printable.vertexColor,
                    opt.heightScale,
                    opt.pixelScaleMm,
                    opt.printableThicknessMm,
                    printableTopology,
                    progress);
            } else {
                const PlyGridTopology topology = buildPlyGridTopology(
                    height,
                    geometryMask,
                    opt.meshStep,
                    !opt.printableMeshPath.empty(),
                    progress);
                if (!opt.meshPath.empty()) {
                    writePlyMesh(
                        opt.meshPath,
                        height,
                        albedo8,
                        opt.heightScale,
                        topology,
                        progress);
                }
                if (!opt.printableMeshPath.empty()) {
                    writePrintablePlyMesh(
                        opt.printableMeshPath,
                        height,
                        albedo8,
                        opt.heightScale,
                        opt.pixelScaleMm,
                        opt.printableThicknessMm,
                        topology,
                        progress);
                }
            }
        }
    }
}
