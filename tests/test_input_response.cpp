#include "input_response.hpp"
#include "image_io.hpp"
#include "photometric.hpp"
#include "args.hpp"
#include "rti_export.hpp"
#include "run_manifest.hpp"

#include <opencv2/imgcodecs.hpp>
#include <zlib.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

namespace {
using Bytes = std::vector<unsigned char>;
namespace fs = std::filesystem;
void check(bool value, const char* text) { if (!value) throw std::runtime_error(text); }

void put(Bytes& b, size_t p, std::uint32_t value, size_t n = 4, bool little = false) {
    if (b.size() < p + n) b.resize(p + n);
    for (size_t i = 0; i < n; ++i) b[p + (little ? i : n - i - 1)] = static_cast<unsigned char>(value >> (8 * i));
}
void name(Bytes& b, size_t p, const char* text) { std::copy(text, text + 4, b.begin() + p); }
void save(const fs::path& path, const Bytes& bytes) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    check(static_cast<bool>(out), "fixture write failed");
}
Bytes exif(bool little, unsigned color) {
    Bytes b(44, 0);
    b[0] = b[1] = little ? 'I' : 'M';
    put(b, 2, 42, 2, little); put(b, 4, 8, 4, little); put(b, 8, 1, 2, little);
    put(b, 10, 34665, 2, little); put(b, 12, 4, 2, little); put(b, 14, 1, 4, little);
    put(b, 18, 26, 4, little); put(b, 26, 1, 2, little);
    put(b, 28, 40961, 2, little); put(b, 30, 3, 2, little); put(b, 32, 1, 4, little);
    put(b, 36, color, 2, little);
    return b;
}
Bytes jpegSegment(const Bytes& original, int marker, Bytes payload) {
    check(payload.size() < 65534, "test JPEG marker too large");
    Bytes segment{255, static_cast<unsigned char>(marker), 0, 0};
    put(segment, 2, static_cast<unsigned>(payload.size() + 2), 2);
    segment.insert(segment.end(), payload.begin(), payload.end());
    Bytes result = original;
    result.insert(result.begin() + 2, segment.begin(), segment.end());
    return result;
}
Bytes pngChunk(const Bytes& original, const char* type, const Bytes& payload) {
    Bytes chunk(8, 0);
    put(chunk, 0, static_cast<unsigned>(payload.size())); name(chunk, 4, type);
    chunk.insert(chunk.end(), payload.begin(), payload.end());
    const auto crc = crc32(0, chunk.data() + 4, static_cast<uInt>(chunk.size() - 4));
    put(chunk, chunk.size(), crc);
    Bytes result = original;
    result.insert(result.begin() + 33, chunk.begin(), chunk.end());
    return result;
}
Bytes profile(bool srgb, bool wide = false) {
    Bytes b(132 + 6 * 12, 0);
    name(b, 12, "mntr"); name(b, 16, "RGB "); name(b, 20, "XYZ "); name(b, 36, "acsp");
    put(b, 128, 6);
    const char* names[] = {"rXYZ", "gXYZ", "bXYZ", "rTRC", "gTRC", "bTRC"};
    const double xyz[3][3] = {{0.43607, 0.22249, 0.01392}, {0.38515, 0.71687, 0.09708}, {0.14307, 0.06061, 0.71410}};
    for (int i = 0; i < 6; ++i) {
        const auto offset = b.size();
        const size_t size = i < 3 ? 20 : (srgb ? 32 : 12);
        name(b, 132 + i * 12, names[i]); put(b, 136 + i * 12, static_cast<unsigned>(offset));
        put(b, 140 + i * 12, static_cast<unsigned>(size)); b.resize(offset + size);
        if (i < 3) {
            name(b, offset, "XYZ ");
            for (int j = 0; j < 3; ++j) put(b, offset + 8 + 4 * j,
                static_cast<unsigned>(std::lround(65536 * (xyz[i][j] + (wide && i == 0 ? 0.1 : 0)))));
        } else if (srgb) {
            name(b, offset, "para"); put(b, offset + 8, 3, 2);
            const double p[] = {2.4, 1.0 / 1.055, 0.055 / 1.055, 1.0 / 12.92, 0.04045};
            for (int j = 0; j < 5; ++j) put(b, offset + 12 + 4 * j, static_cast<unsigned>(std::lround(p[j] * 65536)));
        } else name(b, offset, "curv");
    }
    put(b, 0, static_cast<unsigned>(b.size()));
    return b;
}
Bytes iccp(const Bytes& profileBytes) {
    Bytes compressed(compressBound(static_cast<uLong>(profileBytes.size())));
    uLongf count = static_cast<uLongf>(compressed.size());
    check(compress(compressed.data(), &count, profileBytes.data(), static_cast<uLong>(profileBytes.size())) == Z_OK, "ICC compression failed");
    compressed.resize(count);
    Bytes payload{'t', 'e', 's', 't', 0, 0};
    payload.insert(payload.end(), compressed.begin(), compressed.end());
    return payload;
}
void expectInvalid(const fs::path& path, const Bytes& bytes) {
    save(path, bytes);
    bool threw = false;
    try { (void)inspectInputResponse(path.string()); } catch (const std::runtime_error&) { threw = true; }
    check(threw, "unsupported/malformed metadata was silently accepted");
}
Bytes read(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    check(static_cast<bool>(in), "cannot read test output");
    return Bytes(std::istreambuf_iterator<char>(in), {});
}
Options arguments(std::vector<std::string> strings) {
    std::vector<char*> argv;
    for (auto& s : strings) argv.push_back(s.data());
    return parseArgs(static_cast<int>(argv.size()), argv.data());
}
}

int main(int argc, char** argv) {
    try {
        if (argc > 1) {
            for (int i = 1; i < argc; ++i) {
                const auto response = inspectInputResponse(argv[i]);
                std::cout << argv[i] << ": " << response.source << '\n';
            }
            return 0;
        }
        const fs::path dir = fs::current_path() / "input-response-fixture";
        fs::create_directories(dir);
        const auto path = dir / "metadata-image";
        cv::Mat raw(12, 16, CV_8UC3, cv::Scalar(75, 140, 190));
        Bytes jpeg, png;
        cv::imencode(".jpg", raw, jpeg); cv::imencode(".png", raw, png);
        save(path, jpeg);
        check(inspectInputResponse(path.string()).srgb && inspectInputResponse(path.string()).assumed,
            "untagged JPEG assumption must be explicit and based on signature");
        save(path, png);
        check(!inspectInputResponse(path.string()).srgb && inspectInputResponse(path.string()).assumed,
            "untagged scientific PNG must not be silently decoded");
        for (bool little : {false, true}) {
            auto metadata = exif(little, 1);
            save(path, metadata);
            check(inspectInputResponse(path.string()).srgb, "TIFF Exif sRGB not recognized");
            metadata.insert(metadata.begin(), {'E', 'x', 'i', 'f', 0, 0});
            save(path, jpegSegment(jpeg, 0xe1, metadata));
            const auto result = inspectInputResponse(path.string());
            check(result.srgb && !result.assumed, "JPEG Exif sRGB not recognized");
        }
        auto tagPng = pngChunk(png, "sRGB", {0});
        save(path, tagPng);
        check(inspectInputResponse(path.string()).srgb, "PNG sRGB not recognized");
        save(path, pngChunk(png, "gAMA", {0, 1, 0x86, 0xa0}));
        check(!inspectInputResponse(path.string()).srgb && !inspectInputResponse(path.string()).assumed, "linear PNG gamma not recognized");
        save(path, pngChunk(pngChunk(png, "eXIf", exif(true, 1)), "gAMA", {0, 1, 0x86, 0xa0}));
        check(!inspectInputResponse(path.string()).srgb, "PNG gamma must outrank historical Exif metadata");
        expectInvalid(path, pngChunk(png, "gAMA", {0, 0, 0xb1, 0x8f}));
        save(path, pngChunk(tagPng, "cICP", {1, 8, 0, 1}));
        check(!inspectInputResponse(path.string()).srgb, "cICP must outrank sRGB");
        save(path, pngChunk(pngChunk(png, "iCCP", iccp(profile(false))), "cICP", {1, 13, 0, 1}));
        check(inspectInputResponse(path.string()).srgb, "cICP must outrank ICC");
        expectInvalid(path, pngChunk(tagPng, "cICP", {9, 16, 0, 1}));
        expectInvalid(path, pngChunk(tagPng, "cICP", {1, 13, 0, 0}));
        expectInvalid(path, pngChunk(tagPng, "sRGB", {0}));
        Bytes chrm(32);
        const unsigned primaries[] = {31270, 32900, 64000, 33000, 30000, 60000, 15000, 6000};
        for (int i = 0; i < 8; ++i) put(chrm, 4 * i, primaries[i]);
        save(path, pngChunk(pngChunk(png, "gAMA", {0, 1, 0x86, 0xa0}), "cHRM", chrm));
        check(!inspectInputResponse(path.string()).srgb, "linear sRGB primaries not recognized");
        put(chrm, 8, 68000);
        expectInvalid(path, pngChunk(png, "cHRM", chrm));
        for (bool srgb : {false, true}) {
            const auto icc = profile(srgb);
            save(path, pngChunk(tagPng, "iCCP", iccp(icc)));
            check(inspectInputResponse(path.string()).srgb == srgb, "ICC must outrank PNG sRGB");
            Bytes tagged = jpeg;
            for (int part = 1; part <= 2; ++part) {
                Bytes payload{'I','C','C','_','P','R','O','F','I','L','E',0,static_cast<unsigned char>(part),2};
                const auto begin = icc.begin() + (part - 1) * (icc.size() / 2);
                const auto end = part == 2 ? icc.end() : icc.begin() + icc.size() / 2;
                payload.insert(payload.end(), begin, end);
                tagged = jpegSegment(tagged, 0xe2, payload);
            }
            save(path, tagged);
            check(inspectInputResponse(path.string()).srgb == srgb, "out-of-order JPEG ICC parts not recognized");
        }
        expectInvalid(path, pngChunk(tagPng, "iCCP", iccp(profile(true, true))));
        auto brokenProfile = profile(true); put(brokenProfile, 136, 0xfffffff0);
        expectInvalid(path, pngChunk(tagPng, "iCCP", iccp(brokenProfile)));
        auto brokenPng = tagPng; brokenPng[42] ^= 1;
        expectInvalid(path, brokenPng);
        auto cycle = exif(true, 1); put(cycle, 18, 8, 4, true);
        expectInvalid(path, cycle);
        expectInvalid(path, exif(false, 65535));
        expectInvalid(path, Bytes{'I','I',43,0,8,0,0,0});
        expectInvalid(path, Bytes{255,216,255,225,0,50});
        auto lut = profile(true); name(lut, 132, "A2B0");
        expectInvalid(path, pngChunk(tagPng, "iCCP", iccp(lut)));
        auto wrongGamma = profile(true);
        for (int i = 3; i < 6; ++i) {
            const size_t entry = 136 + i * 12;
            const size_t offset = (wrongGamma[entry] << 24) | (wrongGamma[entry + 1] << 16) |
                (wrongGamma[entry + 2] << 8) | wrongGamma[entry + 3];
            put(wrongGamma, offset + 8, 0, 2);
            put(wrongGamma, offset + 12, static_cast<unsigned>(2.2 * 65536));
        }
        expectInvalid(path, pngChunk(tagPng, "iCCP", iccp(wrongGamma)));
        Options override;
        override.imagePaths = {path.string()};
        override.inputResponseMode = InputResponseMode::Srgb;
        resolveInputResponses(override);
        check(override.inputResponses[0].srgb, "manual sRGB must override unsupported metadata");
        override.inputResponseMode = InputResponseMode::Linear;
        resolveInputResponses(override);
        check(!override.inputResponses[0].srgb, "manual linear must override unsupported metadata");
        check(arguments({"test", "--gui"}).inputResponseMode == InputResponseMode::Auto, "Auto must be the default");
        check(arguments({"test", "--gui", "--srgb"}).inputResponseMode == InputResponseMode::Srgb, "legacy sRGB flag changed");
        check(arguments({"test", "--gui", "--linear"}).inputResponseMode == InputResponseMode::Linear, "linear override not parsed");
        check(arguments({"test", "--gui", "--srgb", "--input-response", "auto"}).inputResponseMode == InputResponseMode::Auto,
            "explicit Auto must override an earlier flag");
        bool invalidOption = false;
        try { (void)arguments({"test", "--gui", "--input-response", "guess"}); }
        catch (const std::exception&) { invalidOption = true; }
        check(invalidOption, "unknown input response accepted");

        Options mixed;
        std::vector<cv::Vec3f> lights;
        const cv::Vec3f truth = cv::normalize(cv::Vec3f(0.12f, -0.08f, 1.0f));
        for (int i = 0; i < 8; ++i) {
            const double theta = i * 3.141592653589793 / 4;
            lights.emplace_back(static_cast<float>(0.6 * std::cos(theta)), static_cast<float>(0.6 * std::sin(theta)), 0.8f);
            const double linear = 0.55 * truth.dot(lights.back());
            const bool srgb = i % 2 == 0;
            const double value = srgb ? 1.055 * std::pow(linear, 1.0 / 2.4) - 0.055 : linear;
            cv::Mat source(12, 16, CV_16U, cv::Scalar(std::lround(value * 65535)));
            Bytes encoded; cv::imencode(".png", source, encoded);
            if (srgb) encoded = pngChunk(encoded, "sRGB", {0});
            else encoded = pngChunk(encoded, "gAMA", {0, 1, 0x86, 0xa0});
            const auto input = dir / ("source-" + std::to_string(i) + ".png");
            save(input, encoded); mixed.imagePaths.push_back(input.string());
        }
        resolveInputResponses(mixed);
        const auto images = loadLuminanceImages(mixed);
        cv::Mat normals, albedo, residual, valid;
        PhotometricDiagnostics diagnostics;
        solvePhotometricStereo(images, lights, cv::Mat(raw.size(), CV_8U, cv::Scalar(255)), 0.02f,
            NormalSolverMode::Robust, 0.98f, LightingModel::Directional, 10, 10, 0.01, cv::Point2d(8, 6),
            cv::Vec3f(0, 0, 1), normals, albedo, residual, valid, diagnostics);
        check(cv::countNonZero(valid) == raw.rows * raw.cols, "automatic response lost mixed-stack pixels");
        const double angle = std::acos(std::clamp(static_cast<double>(truth.dot(normals.at<cv::Vec3f>(0, 0))), -1.0, 1.0)) * 180 / 3.141592653589793;
        check(angle < 0.1, "automatic mixed-response normal error exceeds 0.1 degrees");
        mixed.outputDir = (dir / "manifest").string();
        mixed.calculateHeight = false;
        beginRunManifest(mixed);
        const auto manifestBytes = read(fs::path(mixed.outputDir) / "run_manifest.json");
        const std::string manifest(manifestBytes.begin(), manifestBytes.end());
        check(manifest.find("\"input_response_mode\": \"auto\"") != std::string::npos &&
            manifest.find("\"srgb_decode\": null") != std::string::npos &&
            manifest.find("PNG sRGB chunk") != std::string::npos &&
            manifest.find("PNG gAMA = 1 (linear)") != std::string::npos,
            "manifest must retain per-image mixed response provenance");
        Options tagged;
        for (int i = 0; i < 8; ++i) {
            const double linear = 0.55 * truth.dot(lights[i]);
            const double value = 1.055 * std::pow(linear, 1.0 / 2.4) - 0.055;
            cv::Mat source(12, 16, CV_16UC3, cv::Scalar::all(std::lround(value * 65535)));
            Bytes encoded; cv::imencode(".png", source, encoded);
            const auto input = dir / ("tagged-" + std::to_string(i) + ".png");
            save(input, pngChunk(encoded, "sRGB", {0})); tagged.imagePaths.push_back(input.string());
        }
        resolveInputResponses(tagged);
        tagged.exportRti = true;
        tagged.rtiPath = (dir / "rti-auto").string();
        exportRtiPackage(tagged, lights, raw.size());
        Options explicitSrgb = tagged;
        explicitSrgb.inputResponseMode = InputResponseMode::Srgb;
        explicitSrgb.rtiPath = (dir / "rti-explicit").string();
        exportRtiPackage(explicitSrgb, lights, raw.size());
        int planes = 0;
        for (const auto& entry : fs::directory_iterator(tagged.rtiPath)) {
            if (entry.path().extension() != ".jpg") continue;
            check(read(entry.path()) == read(fs::path(explicitSrgb.rtiPath) / entry.path().filename()),
                "automatic and explicit sRGB RTI planes differ");
            ++planes;
        }
        check(planes > 0, "RTI comparison produced no planes");
        for (int depth : {CV_8U, CV_16U}) {
            const int white = depth == CV_8U ? 255 : 65535;
            cv::Mat rgba(raw.size(), CV_MAKETYPE(depth, 4), cv::Scalar(white / 4, white / 2, white / 3, white));
            const auto rgbaPath = dir / ("alpha-" + std::to_string(depth) + ".png");
            check(cv::imwrite(rgbaPath.string(), rgba), "RGBA fixture write failed");
            Options alpha;
            alpha.imagePaths = {rgbaPath.string()};
            std::vector<cv::Mat> clipping;
            (void)loadLuminanceImages(alpha, &clipping);
            check(cv::countNonZero(clipping[0]) == 0, "opaque alpha must not mark RGB observations clipped");
            rgba(cv::Rect(0, 0, 1, 1)).setTo(cv::Scalar(white, white / 2, white / 3, white));
            check(cv::imwrite(rgbaPath.string(), rgba), "clipped RGBA fixture write failed");
            (void)loadLuminanceImages(alpha, &clipping);
            check(cv::countNonZero(clipping[0]) == 1, "RGB clipping must still be detected with alpha");
        }
        std::cout << "auto_response_mixed_stack_angle_degrees=" << angle << '\n';
        std::cout << "All input-response metadata and scientific checks passed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
