#include "checked_io.hpp"

#include <opencv2/imgcodecs.hpp>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <system_error>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace fs = std::filesystem;

namespace {

std::atomic<unsigned long long> uniqueCounter{0};

void ensureParentDirectory(const fs::path& path) {
    const fs::path parent = path.parent_path();
    if (!parent.empty()) {
        fs::create_directories(parent);
    }
}

} // namespace

fs::path makeUniqueSiblingPath(
    const fs::path& target,
    const std::string& marker,
    bool preserveExtension) {
    const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto sequence = uniqueCounter.fetch_add(1, std::memory_order_relaxed);
    const std::string token = marker + "-" + std::to_string(ticks) + "-" + std::to_string(sequence);
    if (preserveExtension && target.has_extension()) {
        return target.parent_path() /
            (target.stem().string() + "." + token + target.extension().string());
    }
    return target.parent_path() / (target.filename().string() + "." + token);
}

void replaceFileAtomically(const fs::path& temporaryPath, const fs::path& finalPath) {
    ensureParentDirectory(finalPath);
#ifdef _WIN32
    if (!MoveFileExW(
            temporaryPath.c_str(),
            finalPath.c_str(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        throw std::system_error(
            static_cast<int>(GetLastError()),
            std::system_category(),
            "Could not promote temporary output to " + finalPath.string());
    }
#else
    std::error_code error;
    fs::rename(temporaryPath, finalPath, error);
    if (error) {
        throw std::system_error(error, "Could not promote temporary output to " + finalPath.string());
    }
#endif
}

void replaceDirectoryTransactionally(const fs::path& stagingPath, const fs::path& finalPath) {
    if (!fs::is_directory(stagingPath)) {
        throw std::runtime_error("RTI staging directory does not exist: " + stagingPath.string());
    }
    ensureParentDirectory(finalPath);
    const fs::path backupPath = makeUniqueSiblingPath(finalPath, "previous");
    const bool hadPrevious = fs::exists(finalPath);
    if (hadPrevious) {
        fs::rename(finalPath, backupPath);
    }
    try {
        fs::rename(stagingPath, finalPath);
    } catch (...) {
        if (hadPrevious && !fs::exists(finalPath) && fs::exists(backupPath)) {
            std::error_code ignored;
            fs::rename(backupPath, finalPath, ignored);
        }
        throw;
    }
    if (hadPrevious) {
        std::error_code error;
        fs::remove_all(backupPath, error);
        if (error) {
            throw std::system_error(error, "Could not remove replaced output directory " + backupPath.string());
        }
    }
}

CheckedOutputFile::CheckedOutputFile(const fs::path& finalPath, std::ios::openmode mode)
    : finalPath_(finalPath),
      temporaryPath_(makeUniqueSiblingPath(finalPath, "part", true)),
      buffer_(1024 * 1024) {
    ensureParentDirectory(finalPath_);
    output_.rdbuf()->pubsetbuf(buffer_.data(), static_cast<std::streamsize>(buffer_.size()));
    output_.open(temporaryPath_, mode | std::ios::trunc);
    if (!output_) {
        throw std::runtime_error("Could not open temporary output for " + finalPath_.string());
    }
}

CheckedOutputFile::~CheckedOutputFile() {
    if (output_.is_open()) {
        output_.close();
    }
    if (!committed_) {
        std::error_code ignored;
        fs::remove(temporaryPath_, ignored);
    }
}

std::ostream& CheckedOutputFile::stream() {
    return output_;
}

void CheckedOutputFile::commit() {
    if (committed_) {
        return;
    }
    output_.flush();
    if (!output_) {
        throw std::runtime_error("Failed while writing output: " + finalPath_.string());
    }
    output_.close();
    if (output_.fail()) {
        throw std::runtime_error("Failed while closing output: " + finalPath_.string());
    }
    if (!fs::is_regular_file(temporaryPath_) || fs::file_size(temporaryPath_) == 0) {
        throw std::runtime_error("Output is missing or empty: " + finalPath_.string());
    }
    replaceFileAtomically(temporaryPath_, finalPath_);
    committed_ = true;
}

void writeImageChecked(
    const std::filesystem::path& path,
    const cv::Mat& image,
    const std::vector<int>& parameters) {
    if (image.empty()) {
        throw std::runtime_error("Refusing to write an empty image: " + path.string());
    }
    ensureParentDirectory(path);
    const fs::path temporaryPath = makeUniqueSiblingPath(path, "part", true);
    bool written = false;
    try {
        written = cv::imwrite(temporaryPath.string(), image, parameters);
    } catch (const cv::Exception& e) {
        std::error_code ignored;
        fs::remove(temporaryPath, ignored);
        throw std::runtime_error("Failed to write image " + path.string() + ": " + e.what());
    }
    if (!written || !fs::is_regular_file(temporaryPath) || fs::file_size(temporaryPath) == 0) {
        std::error_code ignored;
        fs::remove(temporaryPath, ignored);
        throw std::runtime_error("Failed to write image: " + path.string());
    }
    try {
        replaceFileAtomically(temporaryPath, path);
    } catch (...) {
        std::error_code ignored;
        fs::remove(temporaryPath, ignored);
        throw;
    }
}
