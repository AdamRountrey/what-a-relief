#pragma once

#include <opencv2/core.hpp>

#include <filesystem>
#include <fstream>
#include <ios>
#include <string>
#include <vector>

std::filesystem::path makeUniqueSiblingPath(
    const std::filesystem::path& target,
    const std::string& marker,
    bool preserveExtension = false);

void replaceFileAtomically(
    const std::filesystem::path& temporaryPath,
    const std::filesystem::path& finalPath);

void replaceDirectoryTransactionally(
    const std::filesystem::path& stagingPath,
    const std::filesystem::path& finalPath);

class CheckedOutputFile {
public:
    explicit CheckedOutputFile(
        const std::filesystem::path& finalPath,
        std::ios::openmode mode = std::ios::out | std::ios::binary);
    ~CheckedOutputFile();

    CheckedOutputFile(const CheckedOutputFile&) = delete;
    CheckedOutputFile& operator=(const CheckedOutputFile&) = delete;

    std::ostream& stream();
    void commit();

private:
    std::filesystem::path finalPath_;
    std::filesystem::path temporaryPath_;
    std::ofstream output_;
    bool committed_ = false;
};

void writeImageChecked(
    const std::filesystem::path& path,
    const cv::Mat& image,
    const std::vector<int>& parameters = {});
