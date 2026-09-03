#include "checked_io.hpp"
#include "image_io.hpp"
#include "run_manifest.hpp"
#include "rti_export.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;

struct TestContext {
    int failures = 0;

    void check(bool condition, const std::string& message) {
        if (!condition) {
            ++failures;
            std::cerr << "FAIL: " << message << '\n';
        }
    }
};

std::string readText(const fs::path& path);
std::vector<double> readJsonNumberArray(const std::string& json, const std::string& key);

std::vector<cv::Vec3f> makeLights() {
    std::vector<cv::Vec3f> lights;
    for (int i = 0; i < 4; ++i) {
        const double angle = 2.0 * 3.14159265358979323846 * static_cast<double>(i) / 4.0;
        cv::Vec3f light(
            0.65f * static_cast<float>(std::cos(angle)),
            0.65f * static_cast<float>(std::sin(angle)),
            std::sqrt(1.0f - 0.65f * 0.65f));
        lights.push_back(light);
    }
    return lights;
}

cv::Mat makePattern(int rows, int cols, int lightIndex) {
    cv::Mat image(rows, cols, CV_8UC3);
    for (int y = 0; y < rows; ++y) {
        cv::Vec3b* row = image.ptr<cv::Vec3b>(y);
        for (int x = 0; x < cols; ++x) {
            row[x] = cv::Vec3b(
                static_cast<uchar>((3 * x + y + 17 * lightIndex) % 256),
                static_cast<uchar>((x + 2 * y + 29 * lightIndex) % 256),
                static_cast<uchar>((2 * x + 3 * y + 11 * lightIndex) % 256));
        }
    }
    return image;
}

std::vector<cv::Mat> makePtmCompatibleStack(
    int rows,
    int cols,
    const std::vector<cv::Vec3f>& lights) {
    std::vector<cv::Mat> images;
    images.reserve(lights.size());
    for (const cv::Vec3f& light : lights) {
        cv::Mat image(rows, cols, CV_8UC3);
        for (int y = 0; y < rows; ++y) {
            cv::Vec3b* row = image.ptr<cv::Vec3b>(y);
            const double fy = static_cast<double>(y) / static_cast<double>(rows - 1);
            for (int x = 0; x < cols; ++x) {
                const double fx = static_cast<double>(x) / static_cast<double>(cols - 1);
                const double coefficientX = 0.16 + 0.04 * std::sin(6.283185307179586 * fy);
                const double coefficientY = -0.11 + 0.035 * std::cos(6.283185307179586 * fx);
                const double factor = 0.92 + coefficientX * light[0] + coefficientY * light[1];
                const cv::Vec3d base(
                    0.28 + 0.12 * fx,
                    0.42 + 0.10 * fy,
                    0.58 + 0.08 * fx * fy);
                for (int c = 0; c < 3; ++c) {
                    row[x][c] = static_cast<uchar>(std::clamp(
                        std::round(255.0 * base[c] * factor),
                        0.0,
                        255.0));
                }
            }
        }
        images.push_back(image);
    }
    return images;
}

std::vector<cv::Mat> encodeSrgbStack(const std::vector<cv::Mat>& linearImages) {
    std::vector<cv::Mat> encoded;
    encoded.reserve(linearImages.size());
    for (const cv::Mat& linearImage : linearImages) {
        cv::Mat image = linearImage.clone();
        for (int y = 0; y < image.rows; ++y) {
            cv::Vec3b* row = image.ptr<cv::Vec3b>(y);
            for (int x = 0; x < image.cols; ++x) {
                for (int c = 0; c < 3; ++c) {
                    const double linear = row[x][c] / 255.0;
                    const double srgb = linear <= 0.0031308
                        ? 12.92 * linear
                        : 1.055 * std::pow(linear, 1.0 / 2.4) - 0.055;
                    row[x][c] = static_cast<uchar>(std::clamp(std::lround(255.0 * srgb), 0L, 255L));
                }
            }
        }
        encoded.push_back(std::move(image));
    }
    return encoded;
}

double decodeSrgbCode(uchar code) {
    const double value = code / 255.0;
    return value <= 0.04045
        ? value / 12.92
        : std::pow((value + 0.055) / 1.055, 2.4);
}

double rgbRtiReconstructionError(
    const fs::path& rtiPath,
    const std::vector<cv::Mat>& sourceImages,
    const std::vector<cv::Vec3f>& lights,
    bool sourceIsSrgb = false) {
    const std::string info = readText(rtiPath / "info.json");
    const std::vector<double> scales = readJsonNumberArray(info, "scale");
    const std::vector<double> biases = readJsonNumberArray(info, "bias");
    std::vector<cv::Mat> planes;
    for (int p = 0; p < 3; ++p) {
        planes.push_back(cv::imread((rtiPath / ("plane_" + std::to_string(p) + ".jpg")).string()));
    }

    double error = 0.0;
    std::uint64_t samples = 0;
    for (size_t i = 0; i < sourceImages.size(); ++i) {
        const double basis[3] = {1.0, lights[i][0], lights[i][1]};
        for (int y = 0; y < sourceImages[i].rows; ++y) {
            const cv::Vec3b* sourceRow = sourceImages[i].ptr<cv::Vec3b>(y);
            for (int x = 0; x < sourceImages[i].cols; ++x) {
                for (int bgr = 0; bgr < 3; ++bgr) {
                    const int rgb = 2 - bgr;
                    double predicted = 0.0;
                    for (int p = 0; p < 3; ++p) {
                        const double unit = planes[p].at<cv::Vec3b>(y, x)[bgr] / 255.0;
                        const size_t index = static_cast<size_t>(p * 3 + rgb);
                        const double coefficient = (unit - biases[index]) * scales[index];
                        predicted += coefficient * basis[p];
                    }
                    const double source = sourceIsSrgb
                        ? decodeSrgbCode(sourceRow[x][bgr])
                        : sourceRow[x][bgr] / 255.0;
                    error += std::abs(predicted - source);
                    ++samples;
                }
            }
        }
    }
    return error / static_cast<double>(samples);
}

double lrgbRtiReconstructionError(
    const fs::path& rtiPath,
    const std::vector<cv::Mat>& sourceImages,
    const std::vector<cv::Vec3f>& lights) {
    const std::string info = readText(rtiPath / "info.json");
    const std::vector<double> scales = readJsonNumberArray(info, "scale");
    const std::vector<double> biases = readJsonNumberArray(info, "bias");
    const cv::Mat base = cv::imread((rtiPath / "plane_0.jpg").string());
    const cv::Mat coefficients = cv::imread((rtiPath / "plane_1.jpg").string());

    double error = 0.0;
    std::uint64_t samples = 0;
    for (size_t i = 0; i < sourceImages.size(); ++i) {
        const double basis[3] = {1.0, lights[i][0], lights[i][1]};
        for (int y = 0; y < sourceImages[i].rows; ++y) {
            const cv::Vec3b* sourceRow = sourceImages[i].ptr<cv::Vec3b>(y);
            for (int x = 0; x < sourceImages[i].cols; ++x) {
                double factor = 0.0;
                for (int p = 0; p < 3; ++p) {
                    const int bgr = 2 - p;
                    const double unit = coefficients.at<cv::Vec3b>(y, x)[bgr] / 255.0;
                    const double coefficient = (unit - biases[static_cast<size_t>(3 + p)]) *
                        scales[static_cast<size_t>(3 + p)];
                    factor += coefficient * basis[p];
                }
                for (int bgr = 0; bgr < 3; ++bgr) {
                    const double predicted = base.at<cv::Vec3b>(y, x)[bgr] / 255.0 * factor;
                    error += std::abs(predicted - sourceRow[x][bgr] / 255.0);
                    ++samples;
                }
            }
        }
    }
    return error / static_cast<double>(samples);
}

std::string readText(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

std::vector<double> readJsonNumberArray(const std::string& json, const std::string& key) {
    const size_t keyPosition = json.find("\"" + key + "\"");
    const size_t begin = keyPosition == std::string::npos ? std::string::npos : json.find('[', keyPosition);
    const size_t end = begin == std::string::npos ? std::string::npos : json.find(']', begin);
    if (begin == std::string::npos || end == std::string::npos) {
        throw std::runtime_error("Missing JSON array: " + key);
    }
    std::vector<double> values;
    std::string token;
    std::istringstream in(json.substr(begin + 1, end - begin - 1));
    while (std::getline(in, token, ',')) {
        values.push_back(std::stod(token));
    }
    return values;
}

struct PlyData {
    std::vector<cv::Vec3f> vertices;
    std::vector<std::vector<std::int32_t>> faces;
};

PlyData readBinaryPly(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Could not open PLY fixture output");
    }
    size_t vertexCount = 0;
    size_t faceCount = 0;
    std::string line;
    bool binaryLittleEndian = false;
    while (std::getline(in, line)) {
        if (line == "format binary_little_endian 1.0") {
            binaryLittleEndian = true;
        } else if (line.rfind("element vertex ", 0) == 0) {
            vertexCount = static_cast<size_t>(std::stoull(line.substr(15)));
        } else if (line.rfind("element face ", 0) == 0) {
            faceCount = static_cast<size_t>(std::stoull(line.substr(13)));
        } else if (line == "end_header") {
            break;
        }
    }
    if (!binaryLittleEndian) {
        throw std::runtime_error("Unexpected PLY encoding");
    }

    PlyData data;
    data.vertices.reserve(vertexCount);
    for (size_t i = 0; i < vertexCount; ++i) {
        cv::Vec3f vertex;
        std::uint8_t color[3] = {};
        in.read(reinterpret_cast<char*>(&vertex[0]), sizeof(float));
        in.read(reinterpret_cast<char*>(&vertex[1]), sizeof(float));
        in.read(reinterpret_cast<char*>(&vertex[2]), sizeof(float));
        in.read(reinterpret_cast<char*>(color), sizeof(color));
        data.vertices.push_back(vertex);
    }
    data.faces.reserve(faceCount);
    for (size_t i = 0; i < faceCount; ++i) {
        std::uint8_t count = 0;
        in.read(reinterpret_cast<char*>(&count), sizeof(count));
        std::vector<std::int32_t> face(count);
        in.read(reinterpret_cast<char*>(face.data()), static_cast<std::streamsize>(count * sizeof(std::int32_t)));
        data.faces.push_back(std::move(face));
    }
    if (!in) {
        throw std::runtime_error("Truncated binary PLY output");
    }
    return data;
}

void appendU16(std::vector<std::uint8_t>& bytes, std::uint16_t value) {
    bytes.push_back(static_cast<std::uint8_t>(value & 0xff));
    bytes.push_back(static_cast<std::uint8_t>((value >> 8) & 0xff));
}

void appendU32(std::vector<std::uint8_t>& bytes, std::uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
        bytes.push_back(static_cast<std::uint8_t>((value >> shift) & 0xff));
    }
}

void appendDouble(std::vector<std::uint8_t>& bytes, double value) {
    std::uint64_t raw = 0;
    static_assert(sizeof(raw) == sizeof(value), "Unexpected double size");
    std::memcpy(&raw, &value, sizeof(value));
    for (int shift = 0; shift < 64; shift += 8) {
        bytes.push_back(static_cast<std::uint8_t>((raw >> shift) & 0xff));
    }
}

void appendIfdEntry(
    std::vector<std::uint8_t>& bytes,
    std::uint16_t tag,
    std::uint16_t type,
    std::uint32_t count,
    std::uint32_t valueOffset) {
    appendU16(bytes, tag);
    appendU16(bytes, type);
    appendU32(bytes, count);
    appendU32(bytes, valueOffset);
}

void writeMinimalGeoTiff(
    const fs::path& path,
    double modelScale,
    bool includeLinearUnit,
    std::uint16_t linearUnitCode = 9001) {
    const std::uint16_t entryCount = includeLinearUnit ? 2 : 1;
    const std::uint32_t dataOffset = 8 + 2 + static_cast<std::uint32_t>(entryCount) * 12 + 4;
    const std::uint32_t geoKeyOffset = dataOffset + 3 * 8;

    std::vector<std::uint8_t> bytes;
    bytes.push_back('I');
    bytes.push_back('I');
    appendU16(bytes, 42);
    appendU32(bytes, 8);
    appendU16(bytes, entryCount);
    appendIfdEntry(bytes, 33550, 12, 3, dataOffset);
    if (includeLinearUnit) {
        appendIfdEntry(bytes, 34735, 3, 8, geoKeyOffset);
    }
    appendU32(bytes, 0);
    appendDouble(bytes, modelScale);
    appendDouble(bytes, modelScale);
    appendDouble(bytes, 0.0);
    if (includeLinearUnit) {
        appendU16(bytes, 1);
        appendU16(bytes, 1);
        appendU16(bytes, 0);
        appendU16(bytes, 1);
        appendU16(bytes, 3076);
        appendU16(bytes, 0);
        appendU16(bytes, 1);
        appendU16(bytes, linearUnitCode);
    }

    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!out) {
        throw std::runtime_error("Could not write synthetic TIFF metadata fixture");
    }
}

void testGeoTiffUnits(TestContext& context) {
    const fs::path root = "io_geotiff_test";
    fs::remove_all(root);
    fs::create_directories(root);

    const fs::path metrePath = root / "metre_scale.tif";
    writeMinimalGeoTiff(metrePath, 0.00001, true, 9001);
    context.check(
        std::abs(readPixelScaleMmFromImage(metrePath.string()) - 0.01) < 1.0e-12,
        "GeoTIFF metre model units must be converted to millimetres");

    const fs::path footPath = root / "foot_scale.tif";
    writeMinimalGeoTiff(footPath, 0.001, true, 9002);
    context.check(
        std::abs(readPixelScaleMmFromImage(footPath.string()) - 0.3048) < 1.0e-12,
        "GeoTIFF international-foot model units must be converted to millimetres");

    const fs::path ambiguousPath = root / "ambiguous_scale.tif";
    writeMinimalGeoTiff(ambiguousPath, 0.01, false);
    context.check(
        readPixelScaleMmFromImage(ambiguousPath.string()) == 0.0,
        "ModelPixelScale without declared linear units must not be assumed to be millimetres");

    const fs::path unknownPath = root / "unknown_scale.tif";
    writeMinimalGeoTiff(unknownPath, 0.01, true, 9102);
    context.check(
        readPixelScaleMmFromImage(unknownPath.string()) == 0.0,
        "angular or unsupported GeoTIFF model units must not be converted to physical pixel scale");
    fs::remove_all(root);
}

void testPtmReconstruction(TestContext& context) {
    const fs::path root = "io_rti_reconstruction_test";
    fs::remove_all(root);
    fs::create_directories(root / "inputs");
    constexpr int rows = 48;
    constexpr int cols = 52;
    const std::vector<cv::Vec3f> lights = makeLights();
    const std::vector<cv::Mat> sources = makePtmCompatibleStack(rows, cols, lights);

    Options opt;
    opt.exportRti = true;
    opt.srgb = false;
    opt.rtiLayoutMode = RtiLayoutMode::Image;
    opt.outputDir = (root / "output").string();
    opt.rtiPath = (root / "output" / "rti").string();
    for (size_t i = 0; i < sources.size(); ++i) {
        const fs::path path = root / "inputs" / ("ptm_" + std::to_string(i) + ".png");
        writeImageChecked(path, sources[i]);
        opt.imagePaths.push_back(path.string());
    }

    opt.rtiColorMode = RtiColorMode::Rgb;
    exportRtiPackage(opt, lights, cv::Size(cols, rows));
    const double rgbError = rgbRtiReconstructionError(opt.rtiPath, sources, lights);
    std::cout << "rti_rgb_mean_absolute_linear_code_error=" << rgbError << '\n';
    context.check(
        rgbError < 0.025,
        "RGB PTM export failed to reconstruct its synthetic source stack; MAE=" + std::to_string(rgbError));

    opt.rtiColorMode = RtiColorMode::Lrgb;
    exportRtiPackage(opt, lights, cv::Size(cols, rows));
    const double lrgbError = lrgbRtiReconstructionError(opt.rtiPath, sources, lights);
    std::cout << "rti_lrgb_mean_absolute_linear_code_error=" << lrgbError << '\n';
    context.check(
        lrgbError < 0.030,
        "LRGB PTM export failed to reconstruct its synthetic source stack; MAE=" + std::to_string(lrgbError));
    const cv::Mat base = cv::imread((fs::path(opt.rtiPath) / "plane_0.jpg").string());
    const cv::Scalar channelMean = cv::mean(base);
    context.check(
        std::abs(channelMean[0] - channelMean[2]) > 20.0,
        "LRGB base plane must preserve color rather than collapse to grayscale");

    const std::vector<cv::Mat> srgbSources = encodeSrgbStack(sources);
    for (size_t i = 0; i < srgbSources.size(); ++i) {
        writeImageChecked(opt.imagePaths[i], srgbSources[i]);
    }
    opt.srgb = true;
    opt.rtiColorMode = RtiColorMode::Rgb;
    exportRtiPackage(opt, lights, cv::Size(cols, rows));
    const double srgbError = rgbRtiReconstructionError(opt.rtiPath, srgbSources, lights, true);
    std::cout << "rti_srgb_mean_absolute_linear_error=" << srgbError << '\n';
    context.check(
        srgbError < 0.030,
        "sRGB RTI export failed to reconstruct the decoded linear stack; MAE=" + std::to_string(srgbError));
    fs::remove_all(root);
}

void testDeepZoomLayout(TestContext& context) {
    const fs::path root = "io_export_test";
    fs::remove_all(root);
    fs::create_directories(root / "inputs");
    constexpr int rows = 300;
    constexpr int cols = 513;
    Options opt;
    opt.exportRti = true;
    opt.srgb = false;
    opt.rtiColorMode = RtiColorMode::Rgb;
    opt.rtiLayoutMode = RtiLayoutMode::DeepZoom;
    opt.outputDir = (root / "output").string();
    opt.rtiPath = (root / "output" / "rti").string();
    const std::vector<cv::Vec3f> lights = makeLights();
    for (size_t i = 0; i < lights.size(); ++i) {
        const fs::path path = root / "inputs" / ("image_" + std::to_string(i) + ".png");
        writeImageChecked(path, makePattern(rows, cols, static_cast<int>(i)));
        opt.imagePaths.push_back(path.string());
    }

    exportRtiPackage(opt, lights, cv::Size(cols, rows));
    const fs::path rti = opt.rtiPath;
    const std::string descriptor = readText(rti / "plane_0.dzi");
    context.check(fs::is_regular_file(rti / "rti_manifest.json"), "RTI package manifest is missing");
    context.check(
        descriptor.find("Width=\"513\" Height=\"300\"") != std::string::npos,
        "Deep Zoom descriptor must preserve full image dimensions");
    for (int level = 0; level <= 10; ++level) {
        context.check(
            fs::is_directory(rti / "plane_0_files" / std::to_string(level)),
            "Deep Zoom pyramid is missing level " + std::to_string(level));
    }
    context.check(
        cv::imread((rti / "plane_0_files" / "0" / "0_0.jpg").string()).size() == cv::Size(1, 1),
        "Deep Zoom level zero must be a 1x1 image");
    context.check(
        cv::imread((rti / "plane_0_files" / "10" / "0_0.jpg").string()).size() == cv::Size(256, 256),
        "Deep Zoom top-left full-resolution tile has wrong dimensions");
    context.check(
        cv::imread((rti / "plane_0_files" / "10" / "2_1.jpg").string()).size() == cv::Size(1, 44),
        "Deep Zoom bottom-right edge tile is misplaced or has wrong dimensions");

    cv::Mat stitched(rows, cols, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int ty = 0; ty < 2; ++ty) {
        for (int tx = 0; tx < 3; ++tx) {
            const cv::Mat tile = cv::imread(
                (rti / "plane_0_files" / "10" /
                 (std::to_string(tx) + "_" + std::to_string(ty) + ".jpg")).string());
            context.check(!tile.empty(), "Deep Zoom full-resolution tile could not be decoded");
            if (!tile.empty()) {
                tile.copyTo(stitched(cv::Rect(tx * 256, ty * 256, tile.cols, tile.rows)));
            }
        }
    }

    opt.rtiLayoutMode = RtiLayoutMode::Image;
    exportRtiPackage(opt, lights, cv::Size(cols, rows));
    const cv::Mat fullPlane = cv::imread((rti / "plane_0.jpg").string());
    context.check(!fullPlane.empty(), "full-image RTI coefficient plane could not be decoded");
    if (!fullPlane.empty()) {
        cv::Mat difference;
        cv::absdiff(stitched, fullPlane, difference);
        const cv::Scalar meanDifference = cv::mean(difference);
        const double average = (meanDifference[0] + meanDifference[1] + meanDifference[2]) / 3.0;
        std::cout << "deep_zoom_stitched_mean_absolute_code_error=" << average << '\n';
        context.check(average < 3.0, "Deep Zoom tile placement does not reconstruct the full coefficient plane");
    }
    context.check(!fs::exists(rti / "plane_0.dzi"), "switching RTI layouts must remove stale DZI descriptors");
    context.check(!fs::exists(rti / "plane_0_files"), "switching RTI layouts must remove stale DZI tiles");

    {
        CheckedOutputFile marker(rti / "previous-package-marker.txt");
        marker.stream() << "preserve the last valid package";
        marker.commit();
    }
    const std::string validInfo = readText(rti / "info.json");
    const std::string originalFirstPath = opt.imagePaths.front();
    opt.imagePaths.front() = (root / "inputs" / "missing.png").string();
    bool failedAsExpected = false;
    try {
        exportRtiPackage(opt, lights, cv::Size(cols, rows));
    } catch (const std::exception&) {
        failedAsExpected = true;
    }
    context.check(failedAsExpected, "an unreadable RTI input must fail the requested export");
    context.check(
        readText(rti / "info.json") == validInfo && fs::is_regular_file(rti / "previous-package-marker.txt"),
        "a failed RTI replacement must preserve the complete previous package");
    opt.imagePaths.front() = originalFirstPath;
    const fs::path unrelated = root / "unrelated";
    fs::create_directories(unrelated);
    {
        CheckedOutputFile marker(unrelated / "keep.txt");
        marker.stream() << "not an RTI package";
        marker.commit();
    }
    opt.rtiPath = unrelated.string();
    bool rejectedUnrelatedFolder = false;
    try {
        exportRtiPackage(opt, lights, cv::Size(cols, rows));
    } catch (const std::exception&) {
        rejectedUnrelatedFolder = true;
    }
    context.check(rejectedUnrelatedFolder && readText(unrelated / "keep.txt") == "not an RTI package",
        "RTI export must not replace an unrelated occupied folder");
    fs::remove_all(root);
}

void testRunManifestAndCheckedWrites(TestContext& context) {
    const fs::path root = "io_manifest_test";
    fs::remove_all(root);
    fs::create_directories(root / "inputs");
    fs::create_directories(root / "output");

    Options opt;
    opt.outputDir = (root / "output").string();
    opt.calculateHeight = false;
    opt.solverMode = NormalSolverMode::Standard;
    const std::vector<cv::Vec3f> lights = makeLights();
    for (size_t i = 0; i < lights.size(); ++i) {
        const fs::path path = root / "inputs" / ("manifest_" + std::to_string(i) + ".png");
        writeImageChecked(path, makePattern(12, 14, static_cast<int>(i)));
        opt.imagePaths.push_back(path.string());
    }
    writeImageChecked(root / "output" / "height.png", makePattern(3, 3, 0));
    fs::create_directories(root / "output" / "rti");
    {
        std::ofstream staleRti(root / "output" / "rti" / "rti_manifest.json");
        staleRti << "{\"status\":\"complete\"}\n";
    }

    const RunManifestContext run = beginRunManifest(opt);
    context.check(
        readText(root / "output" / "run_manifest.json").find("\"status\": \"in_progress\"") != std::string::npos,
        "run manifest must advertise in-progress state before outputs are written");
    context.check(
        !fs::exists(root / "output" / "height.png"),
        "a new run must remove stale known output files before writing");
    context.check(
        !fs::exists(root / "output" / "rti"),
        "a run with RTI disabled must remove a stale app-owned default RTI package");

    const cv::Mat normals(12, 14, CV_32FC3, cv::Scalar(0.0f, 0.0f, 1.0f));
    const cv::Mat albedo(12, 14, CV_32F, cv::Scalar(0.6f));
    const cv::Mat residual(12, 14, CV_32F, cv::Scalar(0.01f));
    const cv::Mat mask(12, 14, CV_8U, cv::Scalar(255));
    PhotometricDiagnostics diagnostics;
    diagnostics.lightingConditionNumber = 1.7;
    diagnostics.solvedFraction = 1.0;
    saveOutputs(
        opt,
        lights,
        {},
        normals,
        albedo,
        residual,
        mask,
        diagnostics,
        {},
        {});
    completeRunManifest(opt, run, lights, diagnostics);
    const std::string complete = readText(root / "output" / "run_manifest.json");
    context.check(
        complete.find("\"status\": \"complete\"") != std::string::npos,
        "run manifest must become complete only after verified outputs");
    context.check(
        complete.find("normal_rgb.png") != std::string::npos &&
            complete.find("\"solved_fraction\": 1") != std::string::npos,
        "complete run manifest must enumerate outputs and scientific diagnostics");
    context.check(
        complete.find("\"integration_iterations\": 800") != std::string::npos &&
            complete.find("\"printable_base_thickness_mm\": 2") != std::string::npos &&
            complete.find("\"crop\": null") != std::string::npos,
        "run manifest must retain geometry parameters and explicit absent selections");

    opt.lightsFile = (root / "output" / "lights.csv").string();
    opt.heightMaskPath = (root / "output" / "height_mask.png").string();
    writeImageChecked(opt.heightMaskPath, mask);
    const std::string savedCalibration = readText(opt.lightsFile);
    beginRunManifest(opt);
    context.check(readText(opt.lightsFile) == savedCalibration,
        "repeat-run cleanup must preserve the calibration selected as input");
    context.check(fs::is_regular_file(opt.heightMaskPath),
        "repeat-run cleanup must preserve a selected height mask until it has been read");

    fs::create_directories(root / "output" / "height.png" / "occupied");
    bool rejectedCleanupFailure = false;
    try {
        beginRunManifest(opt);
    } catch (const std::exception&) {
        rejectedCleanupFailure = true;
    }
    context.check(rejectedCleanupFailure, "failed stale-output removal must stop the run");
    context.check(
        readText(root / "output" / "run_manifest.json").find("\"status\": \"in_progress\"") != std::string::npos,
        "cleanup failures must not leave the previous manifest marked complete");
    fs::remove_all(root / "output" / "height.png");

    const fs::path collidingInput = root / "output" / "normal_rgb.png";
    writeImageChecked(collidingInput, makePattern(3, 3, 0));
    opt.imagePaths.push_back(collidingInput.string());
    bool rejectedInputCollision = false;
    try {
        beginRunManifest(opt);
    } catch (const std::exception&) {
        rejectedInputCollision = true;
    }
    context.check(rejectedInputCollision && fs::is_regular_file(collidingInput),
        "input images must not be deleted or overwritten by output cleanup");

    bool rejectedUnsupportedImage = false;
    try {
        writeImageChecked(root / "output" / "unsupported.no_such_codec", makePattern(3, 3, 0));
    } catch (const std::exception&) {
        rejectedUnsupportedImage = true;
    }
    context.check(rejectedUnsupportedImage, "checked image writes must reject unsupported output formats");
    context.check(
        !fs::exists(root / "output" / "unsupported.no_such_codec"),
        "failed checked image writes must not leave a final artifact");
    fs::remove_all(root);
}

void testPrintableMeshTopology(TestContext& context) {
    const fs::path root = "io_mesh_test";
    fs::remove_all(root);
    constexpr int rows = 7;
    constexpr int cols = 9;
    cv::Mat height(rows, cols, CV_32F);
    cv::Mat normals(rows, cols, CV_32FC3);
    for (int y = 0; y < rows; ++y) {
        float* heightRow = height.ptr<float>(y);
        cv::Vec3f* normalRow = normals.ptr<cv::Vec3f>(y);
        for (int x = 0; x < cols; ++x) {
            heightRow[x] = static_cast<float>(0.03 * x + 0.02 * y + 0.08 * std::sin(0.5 * x));
            normalRow[x] = cv::Vec3f(0.0f, 0.0f, 1.0f);
        }
    }
    const cv::Mat mask(rows, cols, CV_8U, cv::Scalar(255));
    const cv::Mat albedo(rows, cols, CV_32F, cv::Scalar(0.65f));
    const cv::Mat residual(rows, cols, CV_32F, cv::Scalar(0.0f));

    Options opt;
    opt.outputDir = (root / "output").string();
    opt.calculateHeight = true;
    opt.solverMode = NormalSolverMode::Standard;
    opt.meshPath = (root / "output" / "surface.ply").string();
    opt.printableMeshPath = (root / "output" / "printable_surface.ply").string();
    opt.pixelScaleMm = 0.2;
    opt.printableThicknessMm = 1.5;
    const std::vector<cv::Vec3f> lights = makeLights();
    for (size_t i = 0; i < lights.size(); ++i) {
        opt.imagePaths.push_back("synthetic_mesh_input_" + std::to_string(i) + ".png");
    }
    saveOutputs(
        opt,
        lights,
        {},
        normals,
        albedo,
        residual,
        mask,
        {},
        height,
        mask);

    const PlyData openMesh = readBinaryPly(opt.meshPath);
    const PlyData solid = readBinaryPly(opt.printableMeshPath);
    context.check(
        openMesh.vertices.size() == static_cast<size_t>(rows * cols),
        "open PLY must contain one vertex per sampled valid height pixel");
    context.check(
        solid.vertices.size() == 2 * openMesh.vertices.size(),
        "printable PLY must contain paired top and bottom vertices");

    std::map<std::pair<std::int32_t, std::int32_t>, int> edgeIncidence;
    bool validIndices = true;
    for (const std::vector<std::int32_t>& face : solid.faces) {
        if (face.size() < 3) {
            validIndices = false;
            continue;
        }
        for (size_t i = 0; i < face.size(); ++i) {
            const std::int32_t a = face[i];
            const std::int32_t b = face[(i + 1) % face.size()];
            if (a < 0 || b < 0 || static_cast<size_t>(a) >= solid.vertices.size() ||
                static_cast<size_t>(b) >= solid.vertices.size() || a == b) {
                validIndices = false;
                continue;
            }
            ++edgeIncidence[{std::min(a, b), std::max(a, b)}];
        }
    }
    context.check(validIndices, "printable PLY contains invalid face indices");
    const bool everyEdgeClosed = std::all_of(
        edgeIncidence.begin(),
        edgeIncidence.end(),
        [](const auto& edge) { return edge.second == 2; });
    context.check(everyEdgeClosed, "every printable PLY edge must have exactly two incident faces");
    const long long eulerCharacteristic =
        static_cast<long long>(solid.vertices.size()) -
        static_cast<long long>(edgeIncidence.size()) +
        static_cast<long long>(solid.faces.size());
    context.check(eulerCharacteristic == 2, "rectangular printable PLY must be a closed genus-zero solid");

    const size_t topCount = openMesh.vertices.size();
    float minimumTop = std::numeric_limits<float>::infinity();
    float bottom = solid.vertices[topCount][2];
    bool flatBottom = true;
    for (size_t i = 0; i < topCount; ++i) {
        minimumTop = std::min(minimumTop, solid.vertices[i][2]);
        flatBottom = flatBottom && std::abs(solid.vertices[topCount + i][2] - bottom) < 1.0e-6f;
    }
    context.check(flatBottom, "printable PLY bottom must be planar");
    context.check(
        std::abs((minimumTop - bottom) - 1.5f) < 1.0e-5f,
        "printable PLY base thickness must be expressed in millimetres");
    fs::remove_all(root);
}

} // namespace

int main() {
    TestContext context;
    testGeoTiffUnits(context);
    testPtmReconstruction(context);
    testDeepZoomLayout(context);
    testRunManifestAndCheckedWrites(context);
    testPrintableMeshTopology(context);
    if (context.failures != 0) {
        std::cerr << context.failures << " I/O/export regression check(s) failed.\n";
        return 1;
    }
    std::cout << "All I/O and export regression checks passed.\n";
    return 0;
}
