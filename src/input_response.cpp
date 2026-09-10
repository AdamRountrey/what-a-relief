#include "input_response.hpp"

#include <zlib.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <vector>

namespace {
using Bytes = std::vector<unsigned char>;
constexpr size_t kMetadataLimit = 4 * 1024 * 1024;

[[noreturn]] void invalidMetadata() {
    throw std::runtime_error("Invalid or oversized image color metadata. Convert to a known color space or select an explicit input response.");
}

std::uint32_t number(const Bytes& data, size_t offset, size_t size, bool little = false) {
    if (size > 4 || offset > data.size() || size > data.size() - offset) invalidMetadata();
    std::uint32_t result = 0;
    for (size_t i = 0; i < size; ++i) result = (result << 8) | data[offset + (little ? size - i - 1 : i)];
    return result;
}

bool signature(const Bytes& data, size_t offset, const char* text, size_t size) {
    return offset <= data.size() && size <= data.size() - offset &&
        std::equal(text, text + size, data.begin() + static_cast<std::ptrdiff_t>(offset),
            [](char a, unsigned char b) { return static_cast<unsigned char>(a) == b; });
}

class MetadataFile {
public:
    explicit MetadataFile(const std::string& path) : stream(path, std::ios::binary | std::ios::ate) {
        if (!stream) throw std::runtime_error("Cannot read input image metadata: " + path);
        const auto end = stream.tellg();
        if (end < 0) invalidMetadata();
        size = static_cast<std::uint64_t>(end);
    }
    Bytes read(std::uint64_t offset, size_t count) {
        if (count > kMetadataLimit || offset > size || count > size - offset ||
            count > 4 * kMetadataLimit - bytesRead) invalidMetadata();
        bytesRead += count;
        Bytes bytes(count);
        stream.seekg(static_cast<std::streamoff>(offset));
        if (count && !stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(count))) invalidMetadata();
        return bytes;
    }
    std::uint64_t size = 0;
private:
    std::ifstream stream;
    size_t bytesRead = 0;
};

struct Tags {
    std::optional<unsigned> exifColor;
    bool pngSrgb = false;
    std::optional<double> gamma;
    std::optional<std::array<unsigned char, 4>> cicp;
    std::optional<std::array<double, 8>> chromaticities;
    Bytes icc;
};

template<class Reader>
void readTiffTags(Reader read, Tags& tags) {
    const auto header = read(0, 8);
    const bool little = signature(header, 0, "II", 2);
    if ((!little && !signature(header, 0, "MM", 2)) || number(header, 2, 2, little) != 42) invalidMetadata();
    std::uint32_t offset = number(header, 4, 4, little);
    // Only the primary IFD and its Exif sub-IFD are relevant; never follow cycles.
    for (int depth = 0; offset != 0 && depth < 2; ++depth) {
        const auto countBytes = read(offset, 2);
        const auto count = number(countBytes, 0, 2, little);
        const auto entries = read(static_cast<std::uint64_t>(offset) + 2, static_cast<size_t>(count) * 12);
        std::uint32_t exif = 0;
        for (size_t i = 0; i < count; ++i) {
            const size_t entry = i * 12;
            const auto tag = number(entries, entry, 2, little);
            const auto type = number(entries, entry + 2, 2, little);
            const auto n = number(entries, entry + 4, 4, little);
            const auto value = number(entries, entry + 8, 4, little);
            if (tag == 34665 && type == 4 && n == 1) exif = value;
            if (tag == 40961) {
                if (type != 3 || n != 1) invalidMetadata();
                const auto color = number(entries, entry + 8, 2, little);
                if (tags.exifColor && *tags.exifColor != color) invalidMetadata();
                tags.exifColor = color;
            }
            if (tag == 34675) {
                if ((type != 7 && type != 1) || n <= 4 || !tags.icc.empty()) invalidMetadata();
                tags.icc = read(value, n);
            }
        }
        if (exif == offset || (depth == 1 && exif != 0)) invalidMetadata();
        offset = exif;
    }
}

void readExif(const Bytes& bytes, Tags& tags) {
    const size_t base = signature(bytes, 0, "Exif\0\0", 6) ? 6 : 0;
    readTiffTags([&](std::uint64_t offset, size_t count) {
        if (offset > bytes.size() - base || count > bytes.size() - base - offset) invalidMetadata();
        const auto start = bytes.begin() + static_cast<std::ptrdiff_t>(base + offset);
        return Bytes(start, start + static_cast<std::ptrdiff_t>(count));
    }, tags);
}

void readJpeg(MetadataFile& file, Tags& tags) {
    std::uint64_t offset = 2;
    std::map<unsigned, Bytes> parts;
    unsigned partCount = 0;
    size_t iccSize = 0;
    bool reachedImage = false;
    for (int markerCount = 0; markerCount < 4096; ++markerCount) {
        auto marker = file.read(offset++, 1)[0];
        if (marker != 0xff) invalidMetadata();
        int fillBytes = 0;
        do {
            if (++fillBytes > 4096) invalidMetadata();
            marker = file.read(offset++, 1)[0];
        } while (marker == 0xff && offset < file.size);
        if (marker == 0xda || marker == 0xd9) { reachedImage = true; break; }
        if (marker == 0x01 || (marker >= 0xd0 && marker <= 0xd7)) continue;
        const auto length = number(file.read(offset, 2), 0, 2);
        if (length < 2 || offset > file.size || length > file.size - offset) invalidMetadata();
        if (marker == 0xe1 || marker == 0xe2) {
            const auto bytes = file.read(offset + 2, length - 2);
            if (marker == 0xe1 && signature(bytes, 0, "Exif\0\0", 6)) readExif(bytes, tags);
            if (marker == 0xe2 && signature(bytes, 0, "ICC_PROFILE\0", 12)) {
                if (bytes.size() < 14 || bytes[12] == 0 || bytes[13] == 0 || bytes[12] > bytes[13] ||
                    (partCount != 0 && partCount != bytes[13]) || parts.count(bytes[12])) invalidMetadata();
                partCount = bytes[13];
                iccSize += bytes.size() - 14;
                if (iccSize > kMetadataLimit) invalidMetadata();
                parts.emplace(bytes[12], Bytes(bytes.begin() + 14, bytes.end()));
            }
        }
        offset += length;
    }
    if (!reachedImage || parts.size() != partCount || (!parts.empty() && !tags.icc.empty())) invalidMetadata();
    for (const auto& part : parts) tags.icc.insert(tags.icc.end(), part.second.begin(), part.second.end());
}

void readPng(MetadataFile& file, Tags& tags) {
    std::uint64_t offset = 8;
    for (int i = 0; i < 4096 && offset < file.size; ++i) {
        const auto header = file.read(offset, 8);
        const auto length = number(header, 0, 4);
        if (offset > file.size || file.size - offset < 12 || length > file.size - offset - 12) invalidMetadata();
        if (signature(header, 4, "IDAT", 4) || signature(header, 4, "IEND", 4)) return;
        const bool srgb = signature(header, 4, "sRGB", 4);
        const bool gamma = signature(header, 4, "gAMA", 4);
        const bool icc = signature(header, 4, "iCCP", 4);
        const bool exif = signature(header, 4, "eXIf", 4);
        const bool cicp = signature(header, 4, "cICP", 4);
        const bool chrm = signature(header, 4, "cHRM", 4);
        if (srgb || gamma || icc || exif || cicp || chrm) {
            const auto bytes = file.read(offset + 8, length);
            auto crc = crc32(0, header.data() + 4, 4);
            crc = crc32(crc, bytes.data(), static_cast<uInt>(bytes.size()));
            if (crc != number(file.read(offset + 8 + length, 4), 0, 4)) invalidMetadata();
            if (srgb) {
                if (length != 1 || bytes[0] > 3 || tags.pngSrgb) invalidMetadata();
                tags.pngSrgb = true;
            }
            if (gamma) {
                if (length != 4 || number(bytes, 0, 4) == 0 || tags.gamma) invalidMetadata();
                tags.gamma = number(bytes, 0, 4) / 100000.0;
            }
            if (exif) readExif(bytes, tags);
            if (cicp) {
                if (length != 4 || tags.cicp) invalidMetadata();
                tags.cicp = std::array<unsigned char, 4>{bytes[0], bytes[1], bytes[2], bytes[3]};
            }
            if (chrm) {
                if (length != 32 || tags.chromaticities) invalidMetadata();
                std::array<double, 8> values{};
                for (size_t j = 0; j < values.size(); ++j) values[j] = number(bytes, 4 * j, 4) / 100000.0;
                tags.chromaticities = values;
            }
            if (icc) {
                const auto end = std::find(bytes.begin(), bytes.end(), 0);
                const size_t nameLength = static_cast<size_t>(end - bytes.begin());
                if (nameLength == 0 || nameLength > 79 || bytes.size() < nameLength + 3 ||
                    bytes[nameLength + 1] != 0 || !tags.icc.empty()) invalidMetadata();
                tags.icc.resize(kMetadataLimit);
                uLongf size = static_cast<uLongf>(tags.icc.size());
                const size_t start = nameLength + 2;
                if (uncompress(tags.icc.data(), &size, bytes.data() + start,
                        static_cast<uLong>(bytes.size() - start)) != Z_OK) invalidMetadata();
                tags.icc.resize(size);
                if (tags.icc.empty()) invalidMetadata();
            }
        }
        offset += static_cast<std::uint64_t>(length) + 12;
    }
    invalidMetadata();
}

double fixed(const Bytes& bytes, size_t offset) {
    const auto raw = number(bytes, offset, 4);
    const std::int64_t signedValue = raw < 0x80000000U ? raw : static_cast<std::int64_t>(raw) - 0x100000000LL;
    return signedValue / 65536.0;
}

double curveValue(const Bytes& bytes, size_t offset, size_t size, double x) {
    if (size < 12) invalidMetadata();
    if (signature(bytes, offset, "curv", 4)) {
        const auto count = number(bytes, offset + 8, 4);
        if (count > (size - 12) / 2) invalidMetadata();
        if (count == 0) return x;
        if (count == 1) return std::pow(x, number(bytes, offset + 12, 2) / 256.0);
        const double p = x * (count - 1);
        const auto lower = static_cast<size_t>(p);
        const auto upper = std::min(static_cast<size_t>(count - 1), lower + 1);
        return ((1.0 - (p - lower)) * number(bytes, offset + 12 + 2 * lower, 2) +
            (p - lower) * number(bytes, offset + 12 + 2 * upper, 2)) / 65535.0;
    }
    if (signature(bytes, offset, "para", 4)) {
        const unsigned type = number(bytes, offset + 8, 2);
        constexpr size_t counts[] = {1, 3, 4, 5, 7};
        if (type > 4 || counts[type] > (size - 12) / 4) invalidMetadata();
        std::array<double, 7> p{};
        for (size_t i = 0; i < counts[type]; ++i) p[i] = fixed(bytes, offset + 12 + 4 * i);
        if (p[0] <= 0.0 || (type != 0 && p[1] <= 0.0)) invalidMetadata();
        if (type == 0) return std::pow(x, p[0]);
        if (type == 1) return x >= -p[2] / p[1] ? std::pow(p[1] * x + p[2], p[0]) : 0.0;
        if (type == 2) return (x >= -p[2] / p[1] ? std::pow(p[1] * x + p[2], p[0]) : 0.0) + p[3];
        if (type == 3) return x >= p[4] ? std::pow(p[1] * x + p[2], p[0]) : p[3] * x;
        return x >= p[4] ? std::pow(p[1] * x + p[2], p[0]) + p[5] : p[3] * x + p[6];
    }
    return std::numeric_limits<double>::quiet_NaN();
}

std::optional<bool> iccResponse(const Bytes& bytes) {
    if (bytes.size() < 132 || number(bytes, 0, 4) != bytes.size() || !signature(bytes, 36, "acsp", 4)) invalidMetadata();
    if (!signature(bytes, 16, "RGB ", 4) || !signature(bytes, 20, "XYZ ", 4)) return {};
    const auto count = number(bytes, 128, 4);
    if (count > (bytes.size() - 132) / 12) invalidMetadata();
    struct Tag { size_t offset = 0; size_t size = 0; };
    std::map<std::string, Tag> tags;
    for (size_t i = 0; i < count; ++i) {
        const size_t entry = 132 + i * 12;
        const auto offset = number(bytes, entry + 4, 4);
        const auto size = number(bytes, entry + 8, 4);
        if (offset < 132 + count * 12 || offset > bytes.size() || size > bytes.size() - offset) invalidMetadata();
        const std::string name(bytes.begin() + entry, bytes.begin() + entry + 4);
        if (name.substr(0, 3) == "A2B" || name.substr(0, 3) == "B2A" ||
            name.substr(0, 3) == "D2B" || name.substr(0, 3) == "B2D") return {};
        if (!tags.emplace(name, Tag{offset, size}).second) invalidMetadata();
    }
    // Only recognize sRGB primaries and transfer curves. Profile names are not
    // reliable evidence; other ICC spaces require conversion outside this app.
    constexpr double xyz[3][3] = {{0.43607, 0.22249, 0.01392}, {0.38515, 0.71687, 0.09708}, {0.14307, 0.06061, 0.71410}};
    bool srgb = true, linear = true;
    const char* colors[] = {"r", "g", "b"};
    for (int c = 0; c < 3; ++c) {
        const auto primary = tags.find(std::string(colors[c]) + "XYZ");
        const auto curve = tags.find(std::string(colors[c]) + "TRC");
        if (primary == tags.end() || curve == tags.end() || primary->second.size < 20 ||
            !signature(bytes, primary->second.offset, "XYZ ", 4)) return {};
        for (int j = 0; j < 3; ++j) if (std::abs(fixed(bytes, primary->second.offset + 8 + 4 * j) - xyz[c][j]) > 0.002) return {};
        for (int i = 0; i <= 128; ++i) {
            const double x = i / 128.0;
            const double value = curveValue(bytes, curve->second.offset, curve->second.size, x);
            const double expected = x <= 0.04045 ? x / 12.92 : std::pow((x + 0.055) / 1.055, 2.4);
            if (!std::isfinite(value)) return {};
            srgb = srgb && std::abs(value - expected) <= 0.001;
            linear = linear && std::abs(value - x) <= 0.001;
        }
    }
    if (srgb) return true;
    if (linear) return false;
    return {};
}
} // namespace

InputResponse inspectInputResponse(const std::string& path) {
    MetadataFile file(path);
    const auto header = file.read(0, static_cast<size_t>(std::min<std::uint64_t>(file.size, 8)));
    const bool jpeg = header.size() >= 2 && header[0] == 0xff && header[1] == 0xd8;
    Tags tags;
    if (jpeg) readJpeg(file, tags);
    else if (signature(header, 0, "\211PNG\r\n\032\n", 8)) readPng(file, tags);
    else if (signature(header, 0, "II\052\0", 4) || signature(header, 0, "MM\0\052", 4)) {
        readTiffTags([&](std::uint64_t offset, size_t count) { return file.read(offset, count); }, tags);
    }
    else if (signature(header, 0, "II\053\0", 4) || signature(header, 0, "MM\0\053", 4)) {
        throw std::runtime_error("Automatic color metadata detection does not support BigTIFF: " + path +
            ". Choose an explicit input response after checking the image provenance.");
    }
    if (tags.cicp) {
        const auto& c = *tags.cicp;
        if (c[0] != 1 || (c[1] != 8 && c[1] != 13) || c[2] != 0 || c[3] != 1) {
            throw std::runtime_error("Unsupported PNG cICP color space in " + path +
                ". Convert to sRGB or linear sRGB before processing.");
        }
        return {c[1] == 13, false, c[1] == 13 ? "PNG cICP sRGB" : "PNG cICP linear sRGB"};
    }
    if (!tags.icc.empty()) {
        const auto response = iccResponse(tags.icc);
        if (!response) throw std::runtime_error("Unsupported ICC color space in " + path +
            ". Convert to sRGB or linear sRGB before processing; an explicit override ignores this profile.");
        return {*response, false, *response ? "ICC sRGB matrix/transfer curves" : "ICC linear sRGB matrix/transfer curves"};
    }
    if (tags.pngSrgb) return {true, false, "PNG sRGB chunk"};
    if (tags.chromaticities) {
        constexpr std::array<double, 8> srgb{0.3127, 0.3290, 0.6400, 0.3300, 0.3000, 0.6000, 0.1500, 0.0600};
        for (size_t i = 0; i < srgb.size(); ++i) {
            if (std::abs((*tags.chromaticities)[i] - srgb[i]) > 0.00002) {
                throw std::runtime_error("Unsupported PNG chromaticities in " + path +
                    ". Convert to sRGB or linear sRGB before processing.");
            }
        }
    }
    if (tags.gamma) {
        if (std::abs(*tags.gamma - 1.0) < 1.0e-5) return {false, false, "PNG gAMA = 1 (linear)"};
        throw std::runtime_error("PNG gamma alone does not declare the sRGB transfer curve in " + path +
            ". Convert to a known color space or choose an explicit input response.");
    }
    if (tags.exifColor) {
        if (*tags.exifColor == 1) return {true, false, "Exif ColorSpace = sRGB"};
        throw std::runtime_error("Exif declares a non-sRGB or uncalibrated color space in " + path +
            ". Convert to a known color space or choose an explicit input response.");
    }
    return {jpeg, true, jpeg ? "No color tag: JPEG assumed sRGB" : "No supported color tag: assumed linear"};
}

InputResponse inputResponseForImage(const Options& opt, size_t index) {
    if (opt.inputResponseMode == InputResponseMode::Srgb) return {true, false, "Manual sRGB override"};
    if (opt.inputResponseMode == InputResponseMode::Linear) return {false, false, "Manual linear override"};
    if (opt.inputResponses.size() == opt.imagePaths.size()) return opt.inputResponses.at(index);
    return inspectInputResponse(opt.imagePaths.at(index));
}

void resolveInputResponses(Options& opt) {
    opt.inputResponses.clear();
    std::vector<InputResponse> resolved;
    for (size_t i = 0; i < opt.imagePaths.size(); ++i) {
        try {
            resolved.push_back(inputResponseForImage(opt, i));
        } catch (const std::exception& e) {
            throw std::runtime_error("Input response for " + opt.imagePaths[i] + ": " + e.what());
        }
    }
    opt.inputResponses = std::move(resolved);
}

std::string inputResponseSummary(const Options& opt) {
    size_t srgb = 0, assumed = 0;
    for (const auto& response : opt.inputResponses) {
        srgb += response.srgb ? 1 : 0;
        assumed += response.assumed ? 1 : 0;
    }
    if (opt.inputResponses.empty()) return "Select images to inspect color metadata. Untagged JPEG: sRGB; other untagged inputs: linear.";
    return std::to_string(srgb) + " sRGB; " + std::to_string(opt.inputResponses.size() - srgb) + " linear. " +
        (assumed ? std::to_string(assumed) + " assumed (untagged JPEG: sRGB; others: linear). Check input provenance." : "No untagged assumptions.");
}

const char* inputResponseModeName(InputResponseMode mode) {
    switch (mode) {
    case InputResponseMode::Linear: return "linear";
    case InputResponseMode::Srgb: return "srgb";
    default: return "auto";
    }
}
