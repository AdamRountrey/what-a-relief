#include "mitsuba_backend.hpp"

#include "checked_io.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <csignal>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

#ifndef WHAT_A_RELIEF_VERSION
#define WHAT_A_RELIEF_VERSION "0.2.1"
#endif

namespace {

constexpr int kJobSchemaVersion = 1;

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool looksLikeConda(const fs::path& path) {
    const std::string value = lowercase(path.string());
    return value.find("anaconda") != std::string::npos ||
        value.find("miniconda") != std::string::npos ||
        value.find(".conda") != std::string::npos ||
        value.find("\\conda\\") != std::string::npos ||
        value.find("/conda/") != std::string::npos;
}

fs::path executableDirectory() {
#ifdef _WIN32
    std::vector<wchar_t> buffer(32768, L'\0');
    const DWORD length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (length == 0 || length >= buffer.size()) {
        return fs::current_path();
    }
    return fs::path(std::wstring(buffer.data(), length)).parent_path();
#else
    std::array<char, 4096> buffer{};
    const ssize_t length = readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
    if (length <= 0) {
        return fs::current_path();
    }
    return fs::path(std::string(buffer.data(), static_cast<size_t>(length))).parent_path();
#endif
}

std::string environmentValue(const char* name) {
#ifdef _WIN32
    char* value = nullptr;
    size_t length = 0;
    if (_dupenv_s(&value, &length, name) != 0 || value == nullptr) {
        return {};
    }
    const std::string result(value);
    std::free(value);
    return result;
#else
    const char* value = std::getenv(name);
    return value == nullptr ? std::string() : std::string(value);
#endif
}

fs::path firstExistingFile(const std::vector<fs::path>& candidates) {
    std::error_code error;
    for (const fs::path& candidate : candidates) {
        if (!candidate.empty() && fs::is_regular_file(candidate, error) && !error) {
            return fs::absolute(candidate).lexically_normal();
        }
        error.clear();
    }
    return {};
}

const char* backendModeName(MitsubaBackendMode mode) {
    switch (mode) {
    case MitsubaBackendMode::Cuda:
        return "cuda";
    case MitsubaBackendMode::Cpu:
        return "cpu";
    case MitsubaBackendMode::Auto:
    default:
        return "auto";
    }
}

const char* qualityModeName(MitsubaQualityMode mode) {
    switch (mode) {
    case MitsubaQualityMode::Preview:
        return "preview";
    case MitsubaQualityMode::Research:
        return "research";
    case MitsubaQualityMode::Standard:
    default:
        return "standard";
    }
}

std::string jsonEscape(const std::string& value) {
    std::ostringstream out;
    for (const unsigned char c : value) {
        switch (c) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (c < 0x20) {
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                    << static_cast<int>(c) << std::dec << std::setfill(' ');
            } else {
                out << static_cast<char>(c);
            }
        }
    }
    return out.str();
}

std::string absolutePathString(const fs::path& path) {
    std::error_code error;
    const fs::path absolute = fs::absolute(path, error);
    return (error ? path : absolute).lexically_normal().string();
}

void writePfm(const fs::path& path, const cv::Mat& image, const cv::Mat& mask) {
    if (image.empty() || image.type() != CV_32F || mask.type() != CV_8U || image.size() != mask.size()) {
        throw std::runtime_error("Mitsuba handoff requires matching CV_32F data and CV_8U mask.");
    }
    CheckedOutputFile checked(path);
    std::ostream& out = checked.stream();
    out << "Pf\n" << image.cols << ' ' << image.rows << "\n-1.0\n";
    std::vector<float> row(static_cast<size_t>(image.cols), 0.0f);
    for (int y = image.rows - 1; y >= 0; --y) {
        const float* source = image.ptr<float>(y);
        const uchar* valid = mask.ptr<uchar>(y);
        for (int x = 0; x < image.cols; ++x) {
            row[static_cast<size_t>(x)] = valid[x] != 0 && std::isfinite(source[x]) ? source[x] : 0.0f;
        }
        out.write(reinterpret_cast<const char*>(row.data()),
            static_cast<std::streamsize>(row.size() * sizeof(float)));
    }
    checked.commit();
}

struct ObservationHandoff {
    std::vector<fs::path> paths;
    double normalizationDivisor = 1.0;
};

ObservationHandoff writeObservationHandoff(
    const fs::path& directory,
    const std::vector<cv::Mat>& images,
    const cv::Size& expectedSize) {
    if (images.empty()) {
        throw std::runtime_error("Mitsuba handoff requires a nonempty linear image stack.");
    }

    double peak = 0.0;
    for (const cv::Mat& image : images) {
        if (image.empty() || image.type() != CV_32F || image.size() != expectedSize) {
            throw std::runtime_error(
                "Mitsuba handoff requires matching single-channel CV_32F observations.");
        }
        for (int y = 0; y < image.rows; ++y) {
            const float* row = image.ptr<float>(y);
            for (int x = 0; x < image.cols; ++x) {
                if (std::isfinite(row[x])) {
                    peak = std::max(peak, static_cast<double>(row[x]));
                }
            }
        }
    }
    if (peak <= static_cast<double>(std::numeric_limits<float>::epsilon())) {
        throw std::runtime_error("Mitsuba handoff image stack contains no positive finite samples.");
    }

    ObservationHandoff handoff;
    handoff.normalizationDivisor = std::max(1.0, peak);
    handoff.paths.reserve(images.size());
    fs::create_directories(directory);
    for (size_t index = 0; index < images.size(); ++index) {
        const cv::Mat& image = images[index];
        cv::Mat encoded(image.size(), CV_16U, cv::Scalar(0));
        for (int y = 0; y < image.rows; ++y) {
            const float* source = image.ptr<float>(y);
            uint16_t* destination = encoded.ptr<uint16_t>(y);
            for (int x = 0; x < image.cols; ++x) {
                const double value = std::isfinite(source[x])
                    ? std::clamp(static_cast<double>(source[x]) / handoff.normalizationDivisor, 0.0, 1.0)
                    : 0.0;
                destination[x] = cv::saturate_cast<uint16_t>(value * 65535.0);
            }
        }
        std::ostringstream name;
        name << "observation_" << std::setw(4) << std::setfill('0') << (index + 1) << ".png";
        const fs::path path = directory / name.str();
        writeImageChecked(path, encoded, {cv::IMWRITE_PNG_COMPRESSION, 1});
        handoff.paths.push_back(path);
    }
    return handoff;
}

void writeJob(
    const fs::path& path,
    const fs::path& staging,
    const Options& opt,
    const std::vector<cv::Vec3f>& lights,
    const ObservationHandoff& observations,
    const std::string& selectedBackend,
    const fs::path& heightPath,
    const fs::path& albedoPath,
    const fs::path& maskPath,
    const fs::path& robustWeightPath) {
    CheckedOutputFile checked(path);
    std::ostream& out = checked.stream();
    out << std::setprecision(17);
    out << "{\n";
    out << "  \"schema_version\": " << kJobSchemaVersion << ",\n";
    out << "  \"application\": {\"name\": \"what-a-relief\", \"version\": \""
        << WHAT_A_RELIEF_VERSION << "\"},\n";
    out << "  \"method\": \"mitsuba_heightfield_inverse_v1\",\n";
    out << "  \"inputs\": {\n";
    out << "    \"images\": [\n";
    for (size_t i = 0; i < observations.paths.size(); ++i) {
        out << "      \"" << jsonEscape(absolutePathString(observations.paths[i])) << "\""
            << (i + 1 == observations.paths.size() ? "\n" : ",\n");
    }
    out << "    ],\n";
    out << "    \"image_transfer\": {\"encoding\": \"png16\", "
        << "\"photometry\": \"linear_luminance\", \"normalization_divisor\": "
        << observations.normalizationDivisor << "},\n";
    out << "    \"source_images\": [\n";
    for (size_t i = 0; i < opt.imagePaths.size(); ++i) {
        out << "      \"" << jsonEscape(absolutePathString(opt.imagePaths[i])) << "\""
            << (i + 1 == opt.imagePaths.size() ? "\n" : ",\n");
    }
    out << "    ],\n";
    out << "    \"height_pfm\": \"" << jsonEscape(absolutePathString(heightPath)) << "\",\n";
    out << "    \"albedo_pfm\": \"" << jsonEscape(absolutePathString(albedoPath)) << "\",\n";
    out << "    \"mask_png\": \"" << jsonEscape(absolutePathString(maskPath)) << "\",\n";
    out << "    \"robust_weight_pfm\": ";
    if (robustWeightPath.empty()) {
        out << "null,\n";
    } else {
        out << "\"" << jsonEscape(absolutePathString(robustWeightPath)) << "\",\n";
    }
    out << "    \"lights\": [\n";
    for (size_t i = 0; i < lights.size(); ++i) {
        out << "      [" << lights[i][0] << ", " << lights[i][1] << ", " << lights[i][2] << "]"
            << (i + 1 == lights.size() ? "\n" : ",\n");
    }
    out << "    ]\n";
    out << "  },\n";
    out << "  \"geometry\": {\n";
    out << "    \"lighting_model\": \""
        << (opt.lightingModel == LightingModel::NearFieldRing ? "near_field_ring" : "directional")
        << "\",\n";
    out << "    \"ring_radius_mm\": " << opt.ringLightRadiusMm << ",\n";
    out << "    \"ring_height_mm\": " << opt.ringLightHeightMm << ",\n";
    out << "    \"pixel_scale_mm_per_pixel\": " << opt.pixelScaleMm << ",\n";
    out << "    \"crop\": ";
    if (opt.hasCrop) {
        out << "{\"x\": " << opt.crop.x << ", \"y\": " << opt.crop.y
            << ", \"width\": " << opt.crop.width << ", \"height\": " << opt.crop.height << "}";
    } else {
        out << "null";
    }
    out << "\n  },\n";
    out << "  \"parameters\": {\n";
    out << "    \"backend_requested\": \"" << backendModeName(opt.mitsubaBackendMode) << "\",\n";
    out << "    \"backend_selected\": \"" << selectedBackend << "\",\n";
    out << "    \"quality\": \"" << qualityModeName(opt.mitsubaQualityMode) << "\",\n";
    out << "    \"srgb_decode\": false,\n";
    out << "    \"source_srgb_decode\": " << (opt.srgb ? "true" : "false") << ",\n";
    out << "    \"height_scale\": " << opt.heightScale << ",\n";
    out << "    \"seed\": 1592607270\n";
    out << "  },\n";
    out << "  \"outputs\": {\"directory\": \""
        << jsonEscape(absolutePathString(staging)) << "\"},\n";
    out << "  \"method_references\": [\n";
    out << "    {\"id\": \"zhang2023projective\", \"doi\": \"10.1145/3618385\"},\n";
    out << "    {\"id\": \"jakob2022drjit\", \"doi\": \"10.1145/3528223.3530099\"},\n";
    out << "    {\"id\": \"mitsuba3\", \"url\": \"https://mitsuba-renderer.org/\"}\n";
    out << "  ]\n";
    out << "}\n";
    checked.commit();
}

std::string readText(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Could not read Mitsuba backend result: " + path.string());
    }
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

std::string regexEscape(const std::string& value) {
    static const std::regex special(R"([.^$|()\[\]{}*+?\\])");
    return std::regex_replace(value, special, R"(\$&)");
}

std::string jsonString(const std::string& text, const std::string& key, const std::string& fallback = {}) {
    const std::regex expression("\\\"" + regexEscape(key) + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
    std::smatch match;
    return std::regex_search(text, match, expression) ? match[1].str() : fallback;
}

double jsonNumber(const std::string& text, const std::string& key, double fallback) {
    const std::regex expression("\\\"" + regexEscape(key) +
        "\\\"\\s*:\\s*(-?(?:[0-9]+(?:\\.[0-9]*)?|\\.[0-9]+)(?:[eE][+-]?[0-9]+)?)");
    std::smatch match;
    return std::regex_search(text, match, expression) ? std::stod(match[1].str()) : fallback;
}

bool jsonBool(const std::string& text, const std::string& key, bool fallback) {
    const std::regex expression("\\\"" + regexEscape(key) + "\\\"\\s*:\\s*(true|false)");
    std::smatch match;
    return std::regex_search(text, match, expression) ? match[1].str() == "true" : fallback;
}

void requireNonempty(const fs::path& path) {
    std::error_code error;
    if (!fs::is_regular_file(path, error) || error || fs::file_size(path, error) == 0 || error) {
        throw std::runtime_error("Mitsuba backend did not produce: " + path.string());
    }
}

void readProgressFile(
    const fs::path& path,
    int& previousPercent,
    std::string& previousMessage,
    const std::function<void(const std::string&, int)>& progress) {
    if (!progress) {
        return;
    }
    std::string snapshot;
#ifdef _WIN32
    // Atomic replacement needs FILE_SHARE_DELETE while this polling read is open.
    // Without it, Windows SMB servers can reject the worker's os.replace().
    const HANDLE handle = CreateFileW(
        path.c_str(), GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        return;
    }
    std::array<char, 8192> buffer{};
    DWORD bytesRead = 0;
    const BOOL read = ReadFile(
        handle, buffer.data(), static_cast<DWORD>(buffer.size() - 1), &bytesRead, nullptr);
    CloseHandle(handle);
    if (!read || bytesRead == 0) {
        return;
    }
    snapshot.assign(buffer.data(), static_cast<size_t>(bytesRead));
#else
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return;
    }
    snapshot.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
#endif
    std::istringstream in(snapshot);
    int percent = -1;
    std::string message;
    if (!(in >> percent)) {
        return;
    }
    in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::getline(in, message);
    percent = std::clamp(percent, 0, 100);
    if (percent != previousPercent || message != previousMessage) {
        previousPercent = percent;
        previousMessage = message;
        progress(message.empty() ? "Mitsuba inverse refinement" : message, percent);
    }
}

int runProcess(
    const fs::path& executable,
    const std::vector<std::string>& arguments,
    const fs::path& workingDirectory,
    const fs::path& logPath,
    const fs::path& progressPath,
    const std::function<void(const std::string&, int)>& progress,
    const std::function<bool()>& cancellationRequested) {
    int previousPercent = -1;
    std::string previousMessage;
#ifdef _WIN32
    SECURITY_ATTRIBUTES security{};
    security.nLength = sizeof(security);
    security.bInheritHandle = TRUE;
    HANDLE logHandle = CreateFileW(
        logPath.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE, &security,
        OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (logHandle == INVALID_HANDLE_VALUE) {
        throw std::system_error(static_cast<int>(GetLastError()), std::system_category(),
            "Could not open Mitsuba backend log");
    }

    std::wstring command = L"\"" + executable.wstring() + L"\"";
    for (const std::string& argument : arguments) {
        const std::wstring wide = fs::path(argument).wstring();
        command += L" \"";
        size_t backslashes = 0;
        for (const wchar_t c : wide) {
            if (c == L'\\') {
                ++backslashes;
            } else if (c == L'\"') {
                command.append(backslashes * 2 + 1, L'\\');
                command.push_back(L'\"');
                backslashes = 0;
            } else {
                command.append(backslashes, L'\\');
                backslashes = 0;
                command.push_back(c);
            }
        }
        command.append(backslashes * 2, L'\\');
        command.push_back(L'\"');
    }
    std::vector<wchar_t> mutableCommand(command.begin(), command.end());
    mutableCommand.push_back(L'\0');

    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdOutput = logHandle;
    startup.hStdError = logHandle;
    startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION process{};
    const BOOL created = CreateProcessW(
        executable.c_str(), mutableCommand.data(), nullptr, nullptr, TRUE,
        CREATE_NO_WINDOW, nullptr,
        workingDirectory.empty() ? nullptr : workingDirectory.c_str(), &startup, &process);
    CloseHandle(logHandle);
    if (!created) {
        throw std::system_error(static_cast<int>(GetLastError()), std::system_category(),
            "Could not launch the Mitsuba backend");
    }

    bool canceled = false;
    while (WaitForSingleObject(process.hProcess, 125) == WAIT_TIMEOUT) {
        readProgressFile(progressPath, previousPercent, previousMessage, progress);
        if (cancellationRequested && cancellationRequested()) {
            TerminateProcess(process.hProcess, 2);
            canceled = true;
            break;
        }
    }
    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD exitCode = 1;
    GetExitCodeProcess(process.hProcess, &exitCode);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    readProgressFile(progressPath, previousPercent, previousMessage, progress);
    if (canceled) {
        throw std::runtime_error("Processing canceled by user.");
    }
    return static_cast<int>(exitCode);
#else
    const pid_t child = fork();
    if (child < 0) {
        throw std::runtime_error("Could not fork the Mitsuba backend process.");
    }
    if (child == 0) {
        if (!workingDirectory.empty()) {
            (void)chdir(workingDirectory.c_str());
        }
        const int log = open(logPath.c_str(), O_CREAT | O_WRONLY | O_APPEND, 0644);
        if (log >= 0) {
            dup2(log, STDOUT_FILENO);
            dup2(log, STDERR_FILENO);
            close(log);
        }
        std::vector<std::string> values;
        values.reserve(arguments.size() + 1);
        values.push_back(executable.string());
        values.insert(values.end(), arguments.begin(), arguments.end());
        std::vector<char*> argv;
        argv.reserve(values.size() + 1);
        for (std::string& value : values) {
            argv.push_back(value.data());
        }
        argv.push_back(nullptr);
        execv(executable.c_str(), argv.data());
        _exit(127);
    }
    int status = 0;
    while (waitpid(child, &status, WNOHANG) == 0) {
        readProgressFile(progressPath, previousPercent, previousMessage, progress);
        if (cancellationRequested && cancellationRequested()) {
            kill(child, SIGTERM);
            waitpid(child, &status, 0);
            throw std::runtime_error("Processing canceled by user.");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(125));
    }
    readProgressFile(progressPath, previousPercent, previousMessage, progress);
    return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
#endif
}

std::vector<std::string> backendCandidates(MitsubaBackendMode mode) {
    if (mode == MitsubaBackendMode::Cuda) {
        return {"cuda"};
    }
    if (mode == MitsubaBackendMode::Cpu) {
        return {"llvm"};
    }
    return {"cuda", "llvm"};
}

std::string selectBackend(
    const MitsubaBackendPaths& paths,
    MitsubaBackendMode requested,
    const fs::path& staging,
    const std::function<void(const std::string&, int)>& progress,
    const std::function<bool()>& cancellationRequested) {
    for (const std::string& candidate : backendCandidates(requested)) {
        if (progress) {
            progress("Testing Mitsuba " + candidate + " backend...", 2);
        }
        const fs::path result = staging / ("probe_" + candidate + ".json");
        const int code = runProcess(
            paths.python,
            {paths.worker.string(), "--probe", "--backend", candidate, "--result", result.string()},
            paths.worker.parent_path(),
            staging / "backend.log",
            staging / "probe_progress.txt",
            {},
            cancellationRequested);
        if (code == 0) {
            requireNonempty(result);
            return candidate;
        }
    }
    if (requested == MitsubaBackendMode::Cuda) {
        throw std::runtime_error(
            "The optional Mitsuba backend was found, but its NVIDIA CUDA probe failed. "
            "Choose Auto or CPU, and inspect " + (staging / "backend.log").string() + ".");
    }
    throw std::runtime_error(
        "The optional Mitsuba backend was found, but no requested compute backend passed its live probe. "
        "Inspect " + (staging / "backend.log").string() + ".");
}

void populateDiagnostics(
    MitsubaRefinementDiagnostics& diagnostics,
    const std::string& text,
    const fs::path& resultPath) {
    diagnostics.succeeded = jsonString(text, "status") == "complete";
    diagnostics.accepted = jsonBool(text, "accepted", false);
    diagnostics.status = jsonString(text, "status", "invalid_result");
    diagnostics.decision = jsonString(text, "decision", "missing_decision");
    diagnostics.selectedBackend = jsonString(text, "selected_backend");
    diagnostics.variant = jsonString(text, "variant");
    diagnostics.optimizer = jsonString(text, "optimizer");
    diagnostics.mitsubaVersion = jsonString(text, "mitsuba_version");
    diagnostics.drjitVersion = jsonString(text, "drjit_version");
    diagnostics.numpyVersion = jsonString(text, "numpy_version");
    diagnostics.pythonVersion = jsonString(text, "python_version");
    diagnostics.iterationsCompleted = static_cast<int>(jsonNumber(text, "iterations_completed", 0));
    diagnostics.renderWidth = static_cast<int>(jsonNumber(text, "render_width", 0));
    diagnostics.renderHeight = static_cast<int>(jsonNumber(text, "render_height", 0));
    diagnostics.trainLossBefore = jsonNumber(text, "train_loss_before", -1.0);
    diagnostics.trainLossAfter = jsonNumber(text, "train_loss_after", -1.0);
    diagnostics.holdoutLossBefore = jsonNumber(text, "holdout_loss_before", -1.0);
    diagnostics.holdoutLossAfter = jsonNumber(text, "holdout_loss_after", -1.0);
    diagnostics.correctionRmsPixels = jsonNumber(text, "correction_rms_pixels", 0.0);
    diagnostics.correctionMaximumPixels = jsonNumber(text, "correction_maximum_pixels", 0.0);
    diagnostics.resultPath = absolutePathString(resultPath);
}

} // namespace

MitsubaBackendPaths resolveMitsubaBackend(const Options& opt) {
    MitsubaBackendPaths result;
    const fs::path exeDir = executableDirectory();
    const std::string envPython = environmentValue("WHAT_A_RELIEF_MITSUBA_PYTHON");
    const std::string envWorker = environmentValue("WHAT_A_RELIEF_MITSUBA_WORKER");
    std::vector<fs::path> pythonCandidates;
    if (!opt.mitsubaPythonPath.empty()) {
        pythonCandidates.emplace_back(opt.mitsubaPythonPath);
    }
    if (!envPython.empty()) {
        pythonCandidates.emplace_back(envPython);
    }
#ifdef _WIN32
    pythonCandidates.push_back(exeDir / "mitsuba-backend" / "python.exe");
    pythonCandidates.push_back(exeDir / "mitsuba-backend" / "Scripts" / "python.exe");
    const std::string localAppData = environmentValue("LOCALAPPDATA");
    if (!localAppData.empty()) {
        pythonCandidates.push_back(fs::path(localAppData) / "Programs" / "what-a-relief-mitsuba" / "python.exe");
        pythonCandidates.push_back(fs::path(localAppData) / "Programs" / "what-a-relief-mitsuba" / "Scripts" / "python.exe");
    }
#else
    pythonCandidates.push_back(exeDir / "mitsuba-backend" / "bin" / "python3");
#endif
    result.python = firstExistingFile(pythonCandidates);

    std::vector<fs::path> workerCandidates;
    if (!opt.mitsubaWorkerPath.empty()) {
        workerCandidates.emplace_back(opt.mitsubaWorkerPath);
    }
    if (!envWorker.empty()) {
        workerCandidates.emplace_back(envWorker);
    }
    workerCandidates.push_back(exeDir / "mitsuba_worker.py");
#ifdef _WIN32
    const std::string localAppDataForWorker = environmentValue("LOCALAPPDATA");
    if (!localAppDataForWorker.empty()) {
        workerCandidates.push_back(
            fs::path(localAppDataForWorker) / "Programs" / "what-a-relief-mitsuba" / "worker.py");
    }
#endif
    workerCandidates.push_back(fs::current_path() / "tools" / "mitsuba_backend" / "worker.py");
    result.worker = firstExistingFile(workerCandidates);

    if (result.python.empty()) {
        result.problem =
            "Optional Mitsuba backend not found. Run the separate what-a-relief Mitsuba backend "
            "installer, then restart this window. Advanced users may also locate a compatible private runtime.";
    } else if (looksLikeConda(result.python)) {
        result.problem =
            "The selected Python belongs to Conda/Anaconda. Use the isolated what-a-relief Mitsuba backend instead.";
    } else if (result.worker.empty()) {
        result.problem =
            "Mitsuba worker script not found. Reinstall what-a-relief or specify --mitsuba-worker.";
    }
    return result;
}

std::string describeMitsubaBackend(const Options& opt) {
    const MitsubaBackendPaths paths = resolveMitsubaBackend(opt);
    if (!paths.available()) {
        return paths.problem;
    }
    return "Backend ready: " + paths.python.parent_path().string() + " (live device probe runs at Start)";
}

void runMitsubaInverseRefinement(
    const Options& opt,
    const std::vector<cv::Vec3f>& lights,
    const std::vector<cv::Mat>& images,
    const cv::Mat& albedo,
    const cv::Mat& height,
    const cv::Mat& heightMask,
    PhotometricDiagnostics& diagnostics,
    const std::function<void(const std::string&, int)>& progress,
    const std::function<bool()>& cancellationRequested) {
    MitsubaRefinementDiagnostics& result = diagnostics.mitsuba;
    result.attempted = true;
    result.status = "preparing";
    result.requestedBackend = backendModeName(opt.mitsubaBackendMode);
    if (opt.uncalibratedLighting || opt.solverMode != NormalSolverMode::Robust ||
        !opt.calculateHeight || opt.imagePaths.size() < 6 || lights.size() != opt.imagePaths.size() ||
        images.size() != opt.imagePaths.size()) {
        throw std::runtime_error(
            "Mitsuba inverse refinement requires at least 6 calibrated robust images and a height field.");
    }
    if (height.empty() || albedo.empty() || heightMask.empty()) {
        throw std::runtime_error("Mitsuba inverse refinement did not receive baseline height, albedo, and mask data.");
    }

    const MitsubaBackendPaths paths = resolveMitsubaBackend(opt);
    if (!paths.available()) {
        result.status = "backend_unavailable";
        result.decision = paths.problem;
        throw std::runtime_error(paths.problem);
    }

    const fs::path finalDirectory = fs::absolute(
        fs::path(opt.outputDir) / "inverse").lexically_normal();
    const fs::path staging = makeUniqueSiblingPath(finalDirectory, "work");
    const fs::path observationDirectory = staging / "input_observations";
    fs::create_directories(staging);
    try {
        if (progress) {
            progress("Preparing versioned Mitsuba inverse-rendering job...", 0);
        }
        const fs::path heightPath = staging / "input_height.pfm";
        const fs::path albedoPath = staging / "input_albedo.pfm";
        const fs::path maskPath = staging / "input_mask.png";
        writePfm(heightPath, height, heightMask);
        writePfm(albedoPath, albedo, heightMask);
        writeImageChecked(maskPath, heightMask);

        if (progress) {
            progress("Encoding linear observations for the Mitsuba handoff...", 1);
        }
        const ObservationHandoff observations =
            writeObservationHandoff(observationDirectory, images, height.size());

        fs::path robustWeightPath;
        if (!diagnostics.robustWeight.empty() && diagnostics.robustWeight.size() == height.size()) {
            robustWeightPath = staging / "input_robust_weight.pfm";
            writePfm(robustWeightPath, diagnostics.robustWeight, heightMask);
        }

        const std::string selectedBackend = selectBackend(
            paths, opt.mitsubaBackendMode, staging, progress, cancellationRequested);
        result.selectedBackend = selectedBackend;
        const fs::path jobPath = staging / "job.json";
        writeJob(
            jobPath, staging, opt, lights, observations, selectedBackend,
            heightPath, albedoPath, maskPath, robustWeightPath);

        const int exitCode = runProcess(
            paths.python,
            {paths.worker.string(), "--job", jobPath.string()},
            paths.worker.parent_path(),
            staging / "backend.log",
            staging / "progress.txt",
            progress,
            cancellationRequested);
        if (exitCode != 0) {
            result.status = "failed";
            result.decision = "backend process exited with code " + std::to_string(exitCode);
            throw std::runtime_error(
                "Mitsuba inverse refinement failed. Baseline outputs are intact; inspect " +
                (staging / "backend.log").string());
        }

        const std::vector<std::string> required = {
            "result.json", "inverse_height.pfm", "inverse_height.png",
            "inverse_normal_rgb.png", "inverse_normal_x.png", "inverse_normal_y.png",
            "inverse_normal_z.png", "inverse_hillshade_ul.png", "inverse_surface.ply",
            "height_correction.pfm", "height_correction.png", "render_before.png",
            "render_after.png", "material_diffuse.png", "material_specular.png",
            "material_roughness.png"};
        for (const std::string& name : required) {
            requireNonempty(staging / name);
        }

        const std::string resultText = readText(staging / "result.json");
        populateDiagnostics(result, resultText, finalDirectory / "result.json");
        if (!result.succeeded) {
            throw std::runtime_error("Mitsuba backend returned an incomplete result. Inspect backend.log.");
        }
        std::error_code observationCleanupError;
        fs::remove_all(observationDirectory, observationCleanupError);
        replaceDirectoryTransactionally(staging, finalDirectory);
        if (progress) {
            progress(result.accepted
                ? "Mitsuba refinement accepted; baseline outputs retained beside inverse results."
                : "Mitsuba refinement rejected by holdout checks; inverse outputs equal the baseline.", 100);
        }
    } catch (...) {
        std::error_code observationCleanupError;
        fs::remove_all(observationDirectory, observationCleanupError);
        result.resultPath = absolutePathString(staging / "result.json");
        throw;
    }
}
