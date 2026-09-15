#include "gui_workflow.hpp"

#include "calibration.hpp"
#include "crop_ui.hpp"
#include "image_io.hpp"
#include "input_response.hpp"
#include "mask_ui.hpp"
#include "mitsuba_backend.hpp"
#include "photometric.hpp"
#include "project_io.hpp"
#include "relight_ui.hpp"
#include "scale_ui.hpp"
#include "sphere_ui.hpp"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <commdlg.h>
#include <commctrl.h>
#include <shlobj.h>
#include <shellapi.h>
#ifdef _MSC_VER
#pragma comment(linker, "/manifestdependency:\"type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")
#endif
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <atomic>
#include <chrono>
#include <deque>
#include <future>
#include <iomanip>
#include <memory>
#include <mutex>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

#ifndef WHAT_A_RELIEF_VERSION
#define WHAT_A_RELIEF_VERSION "0.2.4"
#endif

std::string formatShadowRefinementSummary(const ShadowRefinementSummary& summary) {
    if (!summary.attempted) {
        return {};
    }

    auto number = [](double value, int precision) {
        std::ostringstream out;
        out << std::fixed << std::setprecision(precision) << value;
        return out.str();
    };
    if (summary.applied) {
        std::string message = "Shadow refinement applied";
        if (std::isfinite(summary.balancedMismatchBefore) &&
            std::isfinite(summary.balancedMismatchAfter) &&
            summary.balancedMismatchBefore >= 0.0 &&
            summary.balancedMismatchAfter >= 0.0) {
            message += ": balanced mismatch " + number(summary.balancedMismatchBefore, 4) +
                " -> " + number(summary.balancedMismatchAfter, 4);
            if (summary.balancedMismatchBefore > 0.0) {
                const double improvement = 100.0 *
                    (summary.balancedMismatchBefore - summary.balancedMismatchAfter) /
                    summary.balancedMismatchBefore;
                message += " (" + number(improvement, 1) + "% lower)";
            }
        }
        message += ". Correction RMS: " + number(summary.correctionRmsPixels, 3) +
            " height pixels.";
        return message;
    }

    std::string reason;
    if (summary.decision == "rejected_insufficient_coherent_shadow_lights") {
        reason = "not enough coherent shadow-bearing lights";
    } else if (summary.decision == "rejected_insufficient_shadow_edge_constraints") {
        reason = "not enough reliable shadow-edge constraints";
    } else if (summary.decision == "rejected_fit_did_not_improve") {
        reason = "the fitted-light shadow agreement did not improve";
    } else if (summary.decision == "rejected_withheld_lights_worsened") {
        reason = "held-out lights worsened";
    } else if (summary.decision == "rejected_overall_agreement_did_not_improve") {
        reason = "overall shadow agreement did not improve";
    } else if (summary.decision == "rejected_normal_slope_inconsistency") {
        reason = "the correction conflicted with the photometric normal slopes";
    } else if (summary.decision == "rejected_validation_gate") {
        reason = "the correction failed validation";
    } else if (!summary.decision.empty()) {
        reason = summary.decision;
        std::replace(reason.begin(), reason.end(), '_', ' ');
    } else {
        reason = "the correction was not accepted";
    }
    return "Shadow refinement not applied: " + reason + "; original height retained.";
}

std::string formatLightGainSummary(const LightGainSummary& summary) {
    if (!summary.attempted && !summary.applied) return {};
    if (summary.attempted && summary.applied) {
        std::ostringstream out;
        out << std::fixed << std::setprecision(4)
            << "Specimen light-balance correction applied: held-out error "
            << summary.validationErrorBefore << " -> " << summary.validationErrorAfter
            << "; spatial log-RMS " << summary.stability
            << "; normal dispersion " << summary.normalDiversity << ".";
        return out.str();
    }
    std::string reason = summary.decision;
    std::replace(reason.begin(), reason.end(), '_', ' ');
    return "Specimen light-balance correction not applied: " + reason + ". Equal gains retained.";
}

namespace {

constexpr size_t kMinImages = 3;
constexpr size_t kMaxNeuralImages = 25;

#ifdef _WIN32

constexpr int kWindowWidth = 740;
constexpr int kWindowHeight = 780;
constexpr int kMargin = 20;
constexpr int kLabelWidth = 150;
constexpr int kControlX = 180;
constexpr int kControlWidth = 460;
constexpr int kButtonWidth = 150;
constexpr int kRowHeight = 30;
constexpr int kTabTop = 64;
constexpr int kTabHeight = 470;
constexpr int kPageX = 34;
constexpr int kPageY = 104;
constexpr int kPageWidth = kWindowWidth - 68;
constexpr int kPageHeight = kTabHeight - 52;

constexpr int kIdSelectImages = 1001;
constexpr int kIdImageStatus = 1002;
constexpr int kIdSelectOutput = 1003;
constexpr int kIdOutputStatus = 1004;
constexpr int kIdLighting = 1005;
constexpr int kIdMarkSphere = 1006;
constexpr int kIdSphereStatus = 1007;
constexpr int kIdCrop = 1008;
constexpr int kIdCropStatus = 1009;
constexpr int kIdSolver = 1010;
constexpr int kIdFlatten = 1011;
constexpr int kIdSrgb = 1012;
constexpr int kIdHeight = 1013;
constexpr int kIdMesh = 1014;
constexpr int kIdRelight = 1015;
constexpr int kIdStart = 1016;
constexpr int kIdCancel = 1017;
constexpr int kIdNearField = 1018;
constexpr int kIdRingRadius = 1019;
constexpr int kIdRingHeight = 1020;
constexpr int kIdSpecularDiagnostics = 1021;
constexpr int kIdPixelScale = 1022;
constexpr int kIdSelectLights = 1023;
constexpr int kIdLightsStatus = 1024;
constexpr int kIdClearLights = 1025;
constexpr int kIdNeuralFusion = 1026;
constexpr int kIdHeightMask = 1027;
constexpr int kIdClearHeightMask = 1028;
constexpr int kIdHeightMaskStatus = 1029;
constexpr int kIdHeightSolver = 1030;
constexpr int kIdHeightFlatten = 1031;
constexpr int kIdRtiExport = 1032;
constexpr int kIdRtiLayout = 1033;
constexpr int kIdRtiColor = 1034;
constexpr int kIdMarkScale = 1035;
constexpr int kIdNextStep = 1036;
constexpr int kIdTabs = 1037;
constexpr int kIdProgressBar = 1038;
constexpr int kIdPrintableMesh = 1039;
constexpr int kIdMeshStep = 1040;
constexpr int kIdPrintableThickness = 1041;
constexpr int kIdPrintableFillHoles = 1042;
constexpr int kIdShadowHeightRefinement = 1043;
constexpr int kIdMitsubaInverse = 1044;
constexpr int kIdMitsubaBackend = 1045;
constexpr int kIdMitsubaQuality = 1046;
constexpr int kIdSelectMitsubaPython = 1047;
constexpr int kIdMitsubaStatus = 1048;
constexpr int kIdShadowReferenceZ = 1050;
constexpr int kIdShadowLedDiameter = 1051;
constexpr int kIdMitsubaLightAngle = 1052;
constexpr int kIdRunStatus = 1053;
constexpr int kIdRunLog = 1054;
constexpr int kIdPreview = 1055;
constexpr int kIdPreviewCaption = 1056;
constexpr int kIdMakeCalibrationTarget = 1057;
constexpr int kIdCreateMicroscopeCalibration = 1058;
constexpr int kIdLoadMicroscopeCalibration = 1059;
constexpr int kIdClearMicroscopeCalibration = 1060;
constexpr int kIdMicroscopeCalibrationStatus = 1061;
constexpr int kIdEstimateLightGains = 1062;
constexpr UINT_PTR kRunTimer = 4001;
constexpr int kIdPromptEdit = 2001;
constexpr int kIdPromptOk = 2002;
constexpr int kIdPromptCancel = 2003;
constexpr int kIdNewProject = 3001;
constexpr int kIdOpenProject = 3002;
constexpr int kIdExit = 3003;
constexpr int kIdOpenResults = 3004;
constexpr int kIdOpenReview = 3005;
constexpr UINT kRunProjectMessage = WM_APP + 1;

struct ProgressChannel {
    std::mutex mutex;
    std::deque<ProgressUpdate> events;
};

struct CompletedRun {
    Options options;
    GuiRunResult result;
    std::string error;
};

struct SetupDialogState {
    Options* opt = nullptr;
    bool running = true;
    bool busy = false;
    bool initializing = false;
    bool viewing = false;
    bool selecting = false;
    bool exitAfterRun = false;
    int pendingAction = 0;
    GuiProcessor processor;
    std::shared_ptr<ProgressChannel> channel;
    std::future<CompletedRun> job;
    ProgressTiming timing;
    ProgressUpdate currentProgress;
    std::chrono::steady_clock::time_point started;
    cv::Mat preview;
    std::deque<std::string> logLines;
    HWND runLog = nullptr;
    HWND previewControl = nullptr;
    HWND previewCaption = nullptr;
    std::string lastOutput;
    std::string reviewPath;
    HWND hwnd = nullptr;
    HWND statusText = nullptr;
    HWND progressBar = nullptr;
    int progressBarMode = -1;
    HWND startButton = nullptr;
    HWND cancelButton = nullptr;
    HWND nextStepLabel = nullptr;
    HWND tabControl = nullptr;
    std::vector<HWND> tabPages;
    HWND imageStatus = nullptr;
    HWND outputStatus = nullptr;
    HWND lightsStatus = nullptr;
    HWND lightingCombo = nullptr;
    HWND sphereButton = nullptr;
    HWND sphereStatus = nullptr;
    HWND microscopeCalibrationStatus = nullptr;
    HWND estimateLightGainsCheck = nullptr;
    HWND cropButton = nullptr;
    HWND cropStatus = nullptr;
    HWND heightMaskButton = nullptr;
    HWND clearHeightMaskButton = nullptr;
    HWND heightMaskStatus = nullptr;
    HWND solverCombo = nullptr;
    HWND flattenCombo = nullptr;
    HWND heightSolverCombo = nullptr;
    HWND heightFlattenCombo = nullptr;
    HWND nearFieldCheck = nullptr;
    HWND ringRadiusEdit = nullptr;
    HWND ringHeightEdit = nullptr;
    HWND pixelScaleEdit = nullptr;
    HWND scaleButton = nullptr;
    HWND responseCombo = nullptr;
    HWND responseStatus = nullptr;
    HWND heightCheck = nullptr;
    HWND meshCheck = nullptr;
    HWND printableMeshCheck = nullptr;
    HWND printableFillHolesCheck = nullptr;
    HWND meshStepEdit = nullptr;
    HWND printableThicknessEdit = nullptr;
    HWND rtiCheck = nullptr;
    HWND rtiLayoutCombo = nullptr;
    HWND rtiColorCombo = nullptr;
    HWND relightCheck = nullptr;
    HWND specularDiagnosticsCheck = nullptr;
    HWND shadowHeightRefinementCheck = nullptr;
    HWND shadowReferenceZEdit = nullptr;
    HWND shadowLedDiameterEdit = nullptr;
    HWND sourceSizeLabel = nullptr;
    HWND sourceSizeUnits = nullptr;
    HWND neuralFusionCheck = nullptr;
    HWND mitsubaInverseCheck = nullptr;
    HWND mitsubaBackendCombo = nullptr;
    HWND mitsubaQualityCombo = nullptr;
    HWND mitsubaPythonButton = nullptr;
    HWND mitsubaStatus = nullptr;
    HWND mitsubaLightAngleEdit = nullptr;
    std::vector<HWND> sectionHeaders;
    bool nextStepRequired = true;
    HBRUSH requiredBrush = nullptr;
    HBRUSH optionalBrush = nullptr;
    HBRUSH sectionBrush = nullptr;
    int contentHeight = 0;
    int scrollY = 0;
};

struct NumberPromptState {
    std::string prompt;
    std::string label;
    double value = 0.0;
    bool running = true;
    bool accepted = false;
    HWND edit = nullptr;
};

class SelectionGuard {
public:
    explicit SelectionGuard(SetupDialogState& state) : state_(state), enabled_(IsWindowEnabled(state.hwnd)) {
        state_.selecting = true;
        EnableWindow(state_.hwnd, FALSE);
    }
    ~SelectionGuard() {
        state_.selecting = false;
        EnableWindow(state_.hwnd, enabled_);
        if (state_.pendingAction == kIdExit) {
            state_.pendingAction = 0;
            PostMessageA(state_.hwnd, WM_COMMAND, kIdExit, 0);
        }
    }
    SelectionGuard(const SelectionGuard&) = delete;
    SelectionGuard& operator=(const SelectionGuard&) = delete;
private:
    SetupDialogState& state_;
    BOOL enabled_;
};

void updateApplicationState(SetupDialogState& state);
void layoutApplicationFooter(SetupDialogState& state);
void setProgressBarMode(SetupDialogState& state, bool determinate);

std::atomic_bool gProgressCancelRequested{false};

std::vector<std::string> parseMultiSelectBuffer(const char* buffer) {
    std::vector<std::string> parts;
    const char* p = buffer;
    while (*p != '\0') {
        std::string part(p);
        parts.push_back(part);
        p += part.size() + 1;
    }

    if (parts.empty()) {
        return {};
    }
    if (parts.size() == 1) {
        return parts;
    }

    std::vector<std::string> paths;
    const std::string dir = parts.front();
    for (size_t i = 1; i < parts.size(); ++i) {
        paths.push_back(dir + "\\" + parts[i]);
    }
    return paths;
}

std::vector<std::string> chooseImageFiles(HWND owner) {
    std::vector<char> buffer(1024 * 1024, '\0');
    OPENFILENAMEA ofn = {};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = owner;
    ofn.lpstrFilter =
        "Image files\0*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp\0"
        "All files\0*.*\0";
    ofn.lpstrFile = buffer.data();
    ofn.nMaxFile = static_cast<DWORD>(buffer.size());
    ofn.lpstrTitle = "Select at least 3 photometric stereo images";
    ofn.Flags = OFN_ALLOWMULTISELECT | OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;

    if (!GetOpenFileNameA(&ofn)) {
        throw std::runtime_error("Image selection was canceled.");
    }
    return parseMultiSelectBuffer(buffer.data());
}

std::string chooseOutputFolder(HWND owner) {
    char displayName[MAX_PATH] = {};
    BROWSEINFOA info = {};
    info.hwndOwner = owner;
    info.pszDisplayName = displayName;
    info.lpszTitle = "Choose output folder for what-a-relief results";
    info.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE;

    PIDLIST_ABSOLUTE pidl = SHBrowseForFolderA(&info);
    if (pidl == nullptr) {
        throw std::runtime_error("Output folder selection was canceled.");
    }

    char path[MAX_PATH] = {};
    const BOOL ok = SHGetPathFromIDListA(pidl, path);
    CoTaskMemFree(pidl);
    if (!ok) {
        throw std::runtime_error("Could not read selected output folder.");
    }
    return std::string(path);
}

std::string chooseLightsFile(HWND owner) {
    char buffer[MAX_PATH] = {};
    OPENFILENAMEA ofn = {};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = owner;
    ofn.lpstrFilter =
        "Light vector CSV\0*.csv;*.txt\0"
        "All files\0*.*\0";
    ofn.lpstrFile = buffer;
    ofn.nMaxFile = static_cast<DWORD>(sizeof(buffer));
    ofn.lpstrTitle = "Select previous lights.csv or light_vectors.csv";
    ofn.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;

    if (!GetOpenFileNameA(&ofn)) {
        throw std::runtime_error("Light-vector file selection was canceled.");
    }
    return std::string(buffer);
}

std::string chooseMicroscopeCalibrationFile(HWND owner) {
    char buffer[MAX_PATH] = {};
    OPENFILENAMEA dialog = {};
    dialog.lStructSize = sizeof(dialog);
    dialog.hwndOwner = owner;
    dialog.lpstrFilter = "Microscope calibration JSON\0*.json\0All files\0*.*\0";
    dialog.lpstrFile = buffer;
    dialog.nMaxFile = static_cast<DWORD>(sizeof(buffer));
    dialog.lpstrTitle = "Select microscope checkerboard calibration";
    dialog.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;
    if (!GetOpenFileNameA(&dialog)) throw std::runtime_error("Calibration selection was canceled.");
    return std::string(buffer);
}

std::string chooseCalibrationSaveFile(
    HWND owner,
    const char* title,
    const char* filter,
    const char* defaultExtension,
    const char* suggestedName) {
    char buffer[MAX_PATH] = {};
    strncpy_s(buffer, suggestedName, _TRUNCATE);
    OPENFILENAMEA dialog = {};
    dialog.lStructSize = sizeof(dialog);
    dialog.hwndOwner = owner;
    dialog.lpstrFilter = filter;
    dialog.lpstrFile = buffer;
    dialog.nMaxFile = static_cast<DWORD>(sizeof(buffer));
    dialog.lpstrTitle = title;
    dialog.lpstrDefExt = defaultExtension;
    dialog.Flags = OFN_EXPLORER | OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT;
    if (!GetSaveFileNameA(&dialog)) throw std::runtime_error("Save selection was canceled.");
    return std::string(buffer);
}

std::string chooseMitsubaPython(HWND owner) {
    char buffer[MAX_PATH] = {};
    OPENFILENAMEA ofn = {};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = owner;
    ofn.lpstrFilter =
        "Python executable\0python.exe\0"
        "Executable files\0*.exe\0"
        "All files\0*.*\0";
    ofn.lpstrFile = buffer;
    ofn.nMaxFile = static_cast<DWORD>(sizeof(buffer));
    ofn.lpstrTitle = "Select the isolated what-a-relief Mitsuba python.exe";
    ofn.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;

    if (!GetOpenFileNameA(&ofn)) {
        throw std::runtime_error("Mitsuba backend selection was canceled.");
    }
    return std::string(buffer);
}

std::string baseName(const std::string& path) {
    return fs::path(path).filename().string();
}

std::string imageStatusText(const std::vector<std::string>& paths) {
    if (paths.empty()) {
        return "No images selected";
    }
    std::ostringstream out;
    out << paths.size() << " images selected";
    if (!paths.empty()) {
        out << " (first: " << baseName(paths.front()) << ")";
    }
    return out.str();
}

std::string outputStatusText(const std::string& path) {
    if (path.empty()) {
        return "No output folder selected";
    }
    return path;
}

std::string lightsStatusText(const Options& opt) {
    if (opt.lightsFile.empty()) {
        return "No previous calibration selected";
    }
    return baseName(opt.lightsFile) +
        (opt.lightsFileByOrder ? " loaded by image order; sphere not needed" : " loaded; sphere not needed");
}

std::string microscopeCalibrationStatusText(const Options& opt) {
    if (opt.microscopeCalibrationFile.empty()) return "No microscope calibration";
    return baseName(opt.microscopeCalibrationFile) + " loaded";
}

std::string heightMaskStatusText(const Options& opt) {
    if (opt.hasHeightMask) {
        return "Specimen mask marked";
    }
    if (!opt.heightMaskPath.empty()) {
        return baseName(opt.heightMaskPath) + " loaded";
    }
    return "No specimen mask";
}

void setControlFont(HWND control) {
    SendMessageA(control, WM_SETFONT, reinterpret_cast<WPARAM>(GetStockObject(DEFAULT_GUI_FONT)), TRUE);
}

HWND makeControl(
    HWND parent,
    const char* cls,
    const char* text,
    DWORD style,
    int id,
    int x,
    int y,
    int w,
    int h) {
    HWND control = CreateWindowExA(
        0,
        cls,
        text,
        WS_CHILD | WS_VISIBLE | WS_CLIPSIBLINGS | style,
        x,
        y,
        w,
        h,
        parent,
        reinterpret_cast<HMENU>(static_cast<intptr_t>(id)),
        GetModuleHandleA(nullptr),
        nullptr);
    setControlFont(control);
    return control;
}

HWND makeLabel(HWND parent, const char* text, int x, int y, int w, int h) {
    return makeControl(parent, "STATIC", text, SS_LEFT, 0, x, y + 5, w, h);
}

HWND makeSectionHeader(HWND parent, SetupDialogState& state, const char* text, int y) {
    HWND header = makeControl(parent, "STATIC", text, SS_LEFT, 0, kMargin, y, kWindowWidth - 2 * kMargin - 20, 24);
    state.sectionHeaders.push_back(header);
    return header;
}

LRESULT CALLBACK noWheelSubclassProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam, UINT_PTR subclassId, DWORD_PTR refData) {
    (void)wParam;
    (void)lParam;
    (void)subclassId;
    (void)refData;
    if (msg == WM_MOUSEWHEEL || msg == WM_MOUSEHWHEEL) {
        return 0;
    }
    return DefSubclassProc(hwnd, msg, wParam, lParam);
}

LRESULT CALLBACK tabPageSubclassProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam, UINT_PTR subclassId, DWORD_PTR refData) {
    (void)subclassId;
    (void)refData;
    if (msg == WM_COMMAND || msg == WM_NOTIFY) {
        HWND parent = GetParent(hwnd);
        if (parent != nullptr) {
            return SendMessageA(parent, msg, wParam, lParam);
        }
    }
    return DefSubclassProc(hwnd, msg, wParam, lParam);
}

void blockAccidentalWheel(HWND control) {
    SetWindowSubclass(control, noWheelSubclassProc, 1, 0);
}

HWND makeCombo(HWND parent, int id, int x, int y, int w) {
    HWND combo = makeControl(parent, "COMBOBOX", "", CBS_DROPDOWNLIST | WS_TABSTOP, id, x, y, w, 140);
    blockAccidentalWheel(combo);
    return combo;
}

HWND makeTabPage(HWND parent, SetupDialogState& state, const char* title) {
    TCITEMA item = {};
    item.mask = TCIF_TEXT;
    item.pszText = const_cast<char*>(title);
    const int index = static_cast<int>(state.tabPages.size());
    TabCtrl_InsertItem(state.tabControl, index, &item);

    HWND page = CreateWindowExA(
        WS_EX_CONTROLPARENT,
        "STATIC",
        "",
        WS_CHILD | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | (index == 0 ? WS_VISIBLE : 0),
        kPageX,
        kPageY,
        kPageWidth,
        kPageHeight,
        parent,
        nullptr,
        GetModuleHandleA(nullptr),
        nullptr);
    setControlFont(page);
    SetWindowSubclass(page, tabPageSubclassProc, 1, 0);
    state.tabPages.push_back(page);
    return page;
}

void showSelectedTabPage(SetupDialogState& state) {
    const int selected = std::max(0, TabCtrl_GetCurSel(state.tabControl));
    for (size_t i = 0; i < state.tabPages.size(); ++i) {
        ShowWindow(state.tabPages[i], static_cast<int>(i) == selected ? SW_SHOW : SW_HIDE);
    }
    HWND page = state.tabPages.at(static_cast<size_t>(selected));
    SetWindowPos(page, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
    RedrawWindow(page, nullptr, nullptr, RDW_INVALIDATE | RDW_ERASE | RDW_ALLCHILDREN);
}

LRESULT CALLBACK previewSubclass(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam, UINT_PTR, DWORD_PTR data) {
    if (msg != WM_PAINT && msg != WM_PRINTCLIENT) return DefSubclassProc(hwnd, msg, wParam, lParam);
    auto& state = *reinterpret_cast<SetupDialogState*>(data);
    PAINTSTRUCT paint = {};
    HDC dc = msg == WM_PAINT ? BeginPaint(hwnd, &paint) : reinterpret_cast<HDC>(wParam);
    RECT area = {};
    GetClientRect(hwnd, &area);
    FillRect(dc, &area, reinterpret_cast<HBRUSH>(COLOR_BTNFACE + 1));
    if (!state.preview.empty()) {
        const auto& frame = state.preview;
        const double scale = std::min(static_cast<double>(area.right) / frame.cols, static_cast<double>(area.bottom) / frame.rows);
        const int width = static_cast<int>(frame.cols * scale), height = static_cast<int>(frame.rows * scale);
        BITMAPINFO bitmap = {};
        bitmap.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bitmap.bmiHeader.biWidth = frame.cols;
        bitmap.bmiHeader.biHeight = -frame.rows;
        bitmap.bmiHeader.biPlanes = 1;
        bitmap.bmiHeader.biBitCount = 32;
        bitmap.bmiHeader.biCompression = BI_RGB;
        SetStretchBltMode(dc, HALFTONE);
        StretchDIBits(dc, (area.right - width) / 2, (area.bottom - height) / 2, width, height,
            0, 0, frame.cols, frame.rows, frame.data, &bitmap, DIB_RGB_COLORS, SRCCOPY);
    }
    if (msg == WM_PAINT) EndPaint(hwnd, &paint);
    return 0;
}

void addComboItem(HWND combo, const char* text) {
    SendMessageA(combo, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(text));
}

bool buttonChecked(HWND button) {
    return SendMessageA(button, BM_GETCHECK, 0, 0) == BST_CHECKED;
}

void setButtonChecked(HWND button, bool checked) {
    SendMessageA(button, BM_SETCHECK, checked ? BST_CHECKED : BST_UNCHECKED, 0);
}

void setEditDouble(HWND edit, double value) {
    std::ostringstream out;
    out << value;
    SetWindowTextA(edit, out.str().c_str());
}

void setEditInt(HWND edit, int value) {
    SetWindowTextA(edit, std::to_string(value).c_str());
}

double editDouble(HWND edit, const std::string& label) {
    char buffer[128] = {};
    GetWindowTextA(edit, buffer, static_cast<int>(sizeof(buffer)));
    try {
        size_t end = 0;
        const double value = std::stod(buffer, &end);
        if (end != std::string(buffer).size()) {
            throw std::runtime_error("bad");
        }
        return value;
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid " + label + ": " + std::string(buffer));
    }
}

int editInt(HWND edit, const std::string& label) {
    char buffer[128] = {};
    GetWindowTextA(edit, buffer, static_cast<int>(sizeof(buffer)));
    try {
        size_t end = 0;
        const int value = std::stoi(buffer, &end);
        if (end != std::string(buffer).size()) {
            throw std::runtime_error("bad");
        }
        return value;
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid " + label + ": " + std::string(buffer));
    }
}

LRESULT CALLBACK numberPromptWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (msg == WM_NCCREATE) {
        const CREATESTRUCTA* create = reinterpret_cast<const CREATESTRUCTA*>(lParam);
        SetWindowLongPtrA(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(create->lpCreateParams));
    }

    NumberPromptState* state = reinterpret_cast<NumberPromptState*>(GetWindowLongPtrA(hwnd, GWLP_USERDATA));
    switch (msg) {
    case WM_CREATE:
        if (state != nullptr) {
            makeControl(hwnd, "STATIC", state->prompt.c_str(), SS_LEFT, 0, 18, 18, 380, 48);
            state->edit = makeControl(hwnd, "EDIT", "", ES_LEFT | WS_BORDER | WS_TABSTOP, kIdPromptEdit, 18, 78, 150, kRowHeight);
            setEditDouble(state->edit, state->value);
            makeControl(hwnd, "STATIC", state->label.c_str(), SS_LEFT, 0, 178, 83, 180, kRowHeight);
            makeControl(hwnd, "BUTTON", "OK", BS_DEFPUSHBUTTON, kIdPromptOk, 170, 128, 90, 30);
            makeControl(hwnd, "BUTTON", "Cancel", BS_PUSHBUTTON, kIdPromptCancel, 275, 128, 90, 30);
            SetFocus(state->edit);
            SendMessageA(state->edit, EM_SETSEL, 0, -1);
            return 0;
        }
        break;
    case WM_COMMAND:
        if (state == nullptr) {
            break;
        }
        switch (LOWORD(wParam)) {
        case kIdPromptOk:
            try {
                state->value = editDouble(state->edit, state->label);
                state->accepted = true;
                state->running = false;
                DestroyWindow(hwnd);
            } catch (const std::exception& e) {
                MessageBoxA(hwnd, e.what(), "Value", MB_OK | MB_ICONWARNING);
            }
            return 0;
        case kIdPromptCancel:
            state->accepted = false;
            state->running = false;
            DestroyWindow(hwnd);
            return 0;
        default:
            break;
        }
        break;
    case WM_CLOSE:
        if (state != nullptr) {
            state->accepted = false;
            state->running = false;
        }
        DestroyWindow(hwnd);
        return 0;
    case WM_DESTROY:
        if (state != nullptr) {
            state->running = false;
        }
        return 0;
    default:
        break;
    }
    return DefWindowProcA(hwnd, msg, wParam, lParam);
}

double promptDouble(
    HWND owner,
    const std::string& title,
    const std::string& prompt,
    double defaultValue,
    const std::string& label) {
    const HINSTANCE instance = GetModuleHandleA(nullptr);
    const char* className = "WhatAReliefNumberPrompt";
    WNDCLASSA wc = {};
    wc.lpfnWndProc = numberPromptWndProc;
    wc.hInstance = instance;
    wc.lpszClassName = className;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);
    RegisterClassA(&wc);

    NumberPromptState state;
    state.prompt = prompt;
    state.label = label;
    state.value = defaultValue;

    HWND hwnd = CreateWindowExA(
        WS_EX_DLGMODALFRAME,
        className,
        title.c_str(),
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        430,
        220,
        owner,
        nullptr,
        instance,
        &state);
    if (hwnd == nullptr) {
        throw std::runtime_error("Could not create value prompt.");
    }

    EnableWindow(owner, FALSE);
    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);

    MSG msg = {};
    while (state.running && GetMessageA(&msg, nullptr, 0, 0) > 0) {
        if (!IsDialogMessageA(hwnd, &msg)) {
            TranslateMessage(&msg);
            DispatchMessageA(&msg);
        }
    }
    EnableWindow(owner, TRUE);
    SetForegroundWindow(owner);

    if (!state.accepted) {
        throw std::runtime_error("Scale length entry was canceled.");
    }
    if (!std::isfinite(state.value) || state.value <= 0.0) {
        throw std::runtime_error("Scale length must be a positive millimeter value.");
    }
    return state.value;
}

int comboSelection(HWND combo) {
    return static_cast<int>(SendMessageA(combo, CB_GETCURSEL, 0, 0));
}

void showOwnerMessage(HWND owner, const std::string& title, const std::string& text, UINT icon) {
    MessageBoxA(owner, text.c_str(), title.c_str(), MB_OK | icon);
}

bool confirmLightsFileBinding(
    HWND owner,
    const std::string& path,
    const std::vector<std::string>& imagePaths,
    bool& useFileOrder) {
    useFileOrder = false;
    if (imagePaths.empty()) {
        return true;
    }

    try {
        (void)loadLightsFile(path, imagePaths, nullptr, false);
        return true;
    } catch (const std::exception& identityError) {
        try {
            (void)loadLightsFile(path, imagePaths, nullptr, true);
        } catch (const std::exception& fileError) {
            showOwnerMessage(
                owner,
                "Light Vectors",
                std::string("The calibration file cannot be used with the selected images.\n\n") + fileError.what(),
                MB_ICONWARNING);
            return false;
        }

        std::ostringstream message;
        message
            << "The image names recorded in " << baseName(path)
            << " do not match the selected images.\n\n"
            << "Use the light vectors in CSV row order instead?\n\n"
            << "This assumes the selected images are in exactly the same lighting order as the older run. "
            << "Row 1 will apply to selected image 1, row 2 to selected image 2, and so on.\n\n"
            << "Choose OK only if the lighting order is the same. Choose Cancel to return to setup.\n\n"
            << "Name matching failed: " << identityError.what();
        const int answer = MessageBoxA(
            owner,
            message.str().c_str(),
            "Use Calibration by Image Order?",
            MB_OKCANCEL | MB_ICONWARNING | MB_DEFBUTTON2);
        if (answer != IDOK) {
            return false;
        }
        useFileOrder = true;
        return true;
    }
}

bool isSectionHeader(const SetupDialogState& state, HWND hwnd) {
    return std::find(state.sectionHeaders.begin(), state.sectionHeaders.end(), hwnd) != state.sectionHeaders.end();
}

std::string optionalThenStart(const std::string& text) {
    return text + " Next required (bottom right): run the project.";
}

std::string nextStepText(const SetupDialogState& state, bool& required) {
    const Options& opt = *state.opt;
    required = true;
    if (opt.imagePaths.empty()) {
        return "Next required (Project tab): select at least 3 images.";
    }
    if (opt.outputDir.empty()) {
        return "Next required (Project tab): choose an output folder.";
    }

    const bool calibrated = comboSelection(state.lightingCombo) == 0;
    const bool usingLightsFile = !opt.lightsFile.empty();
    if (!calibrated && opt.imagePaths.size() < 4) {
        return "Next required (Project tab): select at least 4 images for uncalibrated mode, or choose calibrated lighting.";
    }
    if (calibrated && !usingLightsFile && !opt.hasSphere) {
        return "Next required (Lighting tab): load a previous calibration, mark the highlight sphere, or choose uncalibrated lighting.";
    }
    if (calibrated && buttonChecked(state.nearFieldCheck) && opt.pixelScaleMm <= 0.0) {
        return "Next required (Project tab): set XY scale for near-field ring lighting.";
    }
    if (buttonChecked(state.printableMeshCheck) && opt.pixelScaleMm <= 0.0) {
        return "Next required (Project tab): set XY scale for printable mesh export.";
    }
    if (buttonChecked(state.mitsubaInverseCheck) && !resolveMitsubaBackend(opt).available()) {
        return "Next required (Advanced tab): locate or install the optional Mitsuba backend.";
    }
    if (buttonChecked(state.mitsubaInverseCheck) && buttonChecked(state.nearFieldCheck)) {
        try {
            const double diameter = editDouble(state.shadowLedDiameterEdit, "LED diameter");
            if (!std::isfinite(diameter) || diameter <= 0.0) {
                return "Next required (Advanced tab): enter the measured LED emitting diameter in mm for inverse refinement.";
            }
        } catch (const std::exception&) {
            return "Next required (Advanced tab): enter a positive LED emitting diameter in mm.";
        }
    }

    required = false;
    if (opt.pixelScaleMm <= 0.0) {
        return optionalThenStart("Optional (Project tab): set XY scale now if you want physical dimensions or printable meshes later.");
    }
    if (buttonChecked(state.heightCheck) && !opt.hasHeightMask && opt.heightMaskPath.empty()) {
        return optionalThenStart("Optional (Geometry tab): mark a specimen mask if height or mesh should ignore the background.");
    }
    if (calibrated && opt.imagePaths.size() >= kMinImages && !buttonChecked(state.rtiCheck)) {
        return optionalThenStart("Optional (Outputs tab): enable RTI export for Relight/OpenLIME or webRTIViewer.");
    }
    return "Next required (bottom right): run the project when the tabs look right.";
}

void updateSetupControls(SetupDialogState& state) {
    Options& opt = *state.opt;
    SetWindowTextA(state.imageStatus, imageStatusText(opt.imagePaths).c_str());
    SetWindowTextA(state.outputStatus, outputStatusText(opt.outputDir).c_str());
    SetWindowTextA(state.lightsStatus, lightsStatusText(opt).c_str());
    if (opt.hasSphere) {
        std::string sphereStatus = "Sphere marked";
        if (opt.sphereSelectionImageIndex >= 0 &&
            opt.sphereSelectionImageIndex < static_cast<int>(opt.imagePaths.size())) {
            sphereStatus += " in image " + std::to_string(opt.sphereSelectionImageIndex + 1);
        }
        if (!opt.sphereEdgePoints.empty() && opt.sphereFitRmsPixels >= 0.0) {
            std::ostringstream detail;
            detail << " (" << opt.sphereEdgePoints.size() << " points, RMS "
                   << std::fixed << std::setprecision(2) << opt.sphereFitRmsPixels << " px)";
            sphereStatus += detail.str();
        }
        SetWindowTextA(state.sphereStatus, sphereStatus.c_str());
    } else {
        SetWindowTextA(state.sphereStatus, "No sphere marked");
    }
    if (state.microscopeCalibrationStatus != nullptr) {
        SetWindowTextA(state.microscopeCalibrationStatus, microscopeCalibrationStatusText(opt).c_str());
    }
    SetWindowTextA(state.cropStatus, opt.hasCrop ? "Crop selected" : "No crop selected");
    SetWindowTextA(state.heightMaskStatus, heightMaskStatusText(opt).c_str());
    if (state.nextStepLabel != nullptr) {
        SetWindowTextA(state.nextStepLabel, nextStepText(state, state.nextStepRequired).c_str());
        InvalidateRect(state.nextStepLabel, nullptr, TRUE);
    }

    const bool calibrated = comboSelection(state.lightingCombo) == 0;
    const bool usingLightsFile = !opt.lightsFile.empty();
    if (state.estimateLightGainsCheck != nullptr) {
        const bool supportsEstimatedGains = calibrated && opt.imagePaths.size() >= 5;
        if (!supportsEstimatedGains && buttonChecked(state.estimateLightGainsCheck)) {
            setButtonChecked(state.estimateLightGainsCheck, false);
            opt.estimateLightGains = false;
        }
        EnableWindow(state.estimateLightGainsCheck, supportsEstimatedGains);
    }
    EnableWindow(state.lightingCombo, !usingLightsFile);
    EnableWindow(state.sphereButton, calibrated && !usingLightsFile);
    EnableWindow(state.cropButton, !opt.imagePaths.empty());
    EnableWindow(state.heightMaskButton, !opt.imagePaths.empty() && buttonChecked(state.heightCheck));
    EnableWindow(state.clearHeightMaskButton, opt.hasHeightMask || !opt.heightMaskPath.empty());
    EnableWindow(state.solverCombo, calibrated);
    const bool supportsSpecularDiagnostics = calibrated && comboSelection(state.solverCombo) == 0;
    if (!supportsSpecularDiagnostics && buttonChecked(state.specularDiagnosticsCheck)) {
        setButtonChecked(state.specularDiagnosticsCheck, false);
        state.opt->specularDiagnostics = false;
    }
    EnableWindow(state.specularDiagnosticsCheck, supportsSpecularDiagnostics);
    const bool heightEnabled = buttonChecked(state.heightCheck);
    const bool supportsShadowHeightRefinement = supportsSpecularDiagnostics &&
        heightEnabled && state.opt->imagePaths.size() >= 6;
    if (!supportsShadowHeightRefinement && buttonChecked(state.shadowHeightRefinementCheck)) {
        setButtonChecked(state.shadowHeightRefinementCheck, false);
        state.opt->shadowHeightRefinement = false;
    }
    EnableWindow(state.shadowHeightRefinementCheck, supportsShadowHeightRefinement);
    const bool shadowGeometryEnabled = supportsShadowHeightRefinement &&
        (buttonChecked(state.shadowHeightRefinementCheck) || buttonChecked(state.mitsubaInverseCheck)) &&
        buttonChecked(state.nearFieldCheck);
    EnableWindow(state.shadowReferenceZEdit, shadowGeometryEnabled);
    EnableWindow(state.shadowLedDiameterEdit, shadowGeometryEnabled);
    const bool supportsMitsuba = supportsSpecularDiagnostics && heightEnabled &&
        state.opt->imagePaths.size() >= 6;
    if (!supportsMitsuba && buttonChecked(state.mitsubaInverseCheck)) {
        setButtonChecked(state.mitsubaInverseCheck, false);
        state.opt->mitsubaInverseRefinement = false;
    }
    EnableWindow(state.mitsubaInverseCheck, supportsMitsuba);
    const bool mitsubaEnabled = supportsMitsuba && buttonChecked(state.mitsubaInverseCheck);
    EnableWindow(state.mitsubaBackendCombo, mitsubaEnabled);
    EnableWindow(state.mitsubaQualityCombo, mitsubaEnabled);
    EnableWindow(state.mitsubaPythonButton, mitsubaEnabled);
    EnableWindow(state.mitsubaLightAngleEdit, mitsubaEnabled && !buttonChecked(state.nearFieldCheck));
    const bool angularSource = mitsubaEnabled && !buttonChecked(state.nearFieldCheck);
    ShowWindow(state.shadowLedDiameterEdit, angularSource ? SW_HIDE : SW_SHOW);
    ShowWindow(state.mitsubaLightAngleEdit, angularSource ? SW_SHOW : SW_HIDE);
    SetWindowTextA(state.sourceSizeLabel, angularSource ? "Angular Diameter" : "LED Diameter");
    SetWindowTextA(state.sourceSizeUnits, angularSource
        ? "degrees; directional inverse approximation"
        : "mm; inverse requires >0; shadow allows 0");
    if (state.mitsubaStatus != nullptr) {
        const std::string status = mitsubaEnabled
            ? describeMitsubaBackend(opt)
            : "Optional backend is inactive; standard processing is self-contained.";
        SetWindowTextA(state.mitsubaStatus, status.c_str());
    }
    EnableWindow(state.heightSolverCombo, heightEnabled);
    EnableWindow(state.heightFlattenCombo, heightEnabled);
    EnableWindow(state.meshCheck, heightEnabled || buttonChecked(state.meshCheck));
    EnableWindow(state.printableMeshCheck, heightEnabled || buttonChecked(state.printableMeshCheck));
    EnableWindow(state.printableFillHolesCheck, buttonChecked(state.printableMeshCheck));
    EnableWindow(state.meshStepEdit, buttonChecked(state.meshCheck) || buttonChecked(state.printableMeshCheck));
    EnableWindow(state.printableThicknessEdit, buttonChecked(state.printableMeshCheck));
    EnableWindow(state.nearFieldCheck, calibrated && !usingLightsFile);
    const bool supportsNeuralFusion = calibrated &&
        state.opt->imagePaths.size() >= kMinImages &&
        state.opt->imagePaths.size() <= kMaxNeuralImages;
    if (!supportsNeuralFusion && buttonChecked(state.neuralFusionCheck)) {
        setButtonChecked(state.neuralFusionCheck, false);
        state.opt->neuralFusion = false;
    }
    const bool supportsRti = calibrated && state.opt->imagePaths.size() >= kMinImages;
    EnableWindow(state.rtiCheck, supportsRti);
    EnableWindow(state.rtiLayoutCombo, supportsRti && buttonChecked(state.rtiCheck));
    EnableWindow(state.rtiColorCombo, supportsRti && buttonChecked(state.rtiCheck));
    EnableWindow(state.neuralFusionCheck, supportsNeuralFusion);
    const bool nearField = calibrated && !usingLightsFile && buttonChecked(state.nearFieldCheck);
    EnableWindow(state.ringRadiusEdit, nearField);
    EnableWindow(state.ringHeightEdit, nearField);
    EnableWindow(state.pixelScaleEdit, TRUE);
    EnableWindow(state.scaleButton, !opt.imagePaths.empty());
    updateApplicationState(state);
}

void updateSetupScrollInfo(SetupDialogState& state) {
    RECT client = {};
    GetClientRect(state.hwnd, &client);
    const int pageHeight = std::max(1, static_cast<int>(client.bottom - client.top));

    SCROLLINFO info = {};
    info.cbSize = sizeof(info);
    info.fMask = SIF_RANGE | SIF_PAGE | SIF_POS;
    info.nMin = 0;
    info.nMax = std::max(0, state.contentHeight - 1);
    info.nPage = static_cast<UINT>(pageHeight);
    info.nPos = state.scrollY;
    SetScrollInfo(state.hwnd, SB_VERT, &info, TRUE);
}

void scrollSetupWindow(SetupDialogState& state, int targetY) {
    RECT client = {};
    GetClientRect(state.hwnd, &client);
    const int pageHeight = std::max(1, static_cast<int>(client.bottom - client.top));
    const int maxScroll = std::max(0, state.contentHeight - pageHeight);
    targetY = std::clamp(targetY, 0, maxScroll);
    if (targetY == state.scrollY) {
        return;
    }

    const int delta = state.scrollY - targetY;
    state.scrollY = targetY;
    ScrollWindowEx(
        state.hwnd,
        0,
        delta,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        SW_SCROLLCHILDREN | SW_INVALIDATE);
    updateSetupScrollInfo(state);
}

void refreshInputResponse(SetupDialogState& state) {
    if (state.responseStatus == nullptr) return;
    state.opt->inputResponseMode = static_cast<InputResponseMode>(comboSelection(state.responseCombo));
    try {
        resolveInputResponses(*state.opt);
        SetWindowTextA(state.responseStatus, inputResponseSummary(*state.opt).c_str());
    } catch (const std::exception& e) {
        SetWindowTextA(state.responseStatus, e.what());
    }
}

void clearSphereSelection(Options& opt) {
    opt.hasSphere = false;
    opt.sphere = {};
    opt.sphereSelectionImageIndex = 0;
    opt.sphereEdgePoints.clear();
    opt.sphereFitRmsPixels = -1.0;
    opt.sphereFitMaxResidualPixels = -1.0;
    opt.sphereFitCoverageDegrees = -1.0;
}

void selectImages(SetupDialogState& state) {
    try {
        state.opt->imagePaths = chooseImageFiles(state.hwnd);
        clearSphereSelection(*state.opt);
        state.opt->hasCrop = false;
        state.opt->heightMask.release();
        state.opt->hasHeightMask = false;
        state.opt->heightMaskPath.clear();
        state.opt->lightsFileByOrder = false;
        if (!state.opt->imagePaths.empty() &&
            (state.opt->outputDir.empty() || state.opt->outputDir == "out")) {
            state.opt->outputDir = (fs::path(state.opt->imagePaths.front()).parent_path() / "what-a-relief").string();
        }
        double tagScale = state.opt->imagePaths.empty() ? 0.0 : readPixelScaleMmFromImage(state.opt->imagePaths.front());
        if (!state.opt->microscopeCalibrationFile.empty()) {
            tagScale = loadMicroscopeCalibration(state.opt->microscopeCalibrationFile).pixelScaleMm;
        }
        if (tagScale > 0.0 && state.pixelScaleEdit != nullptr) {
            state.opt->pixelScaleMm = tagScale;
            setEditDouble(state.pixelScaleEdit, tagScale);
        }
        refreshInputResponse(state);
        updateSetupControls(state);
    } catch (const std::exception& e) {
        showOwnerMessage(state.hwnd, "Image Selection", e.what(), MB_ICONINFORMATION);
    }
}

void selectOutput(SetupDialogState& state) {
    try {
        state.opt->outputDir = chooseOutputFolder(state.hwnd);
        updateSetupControls(state);
    } catch (const std::exception& e) {
        showOwnerMessage(state.hwnd, "Output Folder", e.what(), MB_ICONINFORMATION);
    }
}

void selectLightsFile(SetupDialogState& state) {
    try {
        const std::string selectedPath = chooseLightsFile(state.hwnd);
        bool useFileOrder = false;
        if (!confirmLightsFileBinding(
                state.hwnd, selectedPath, state.opt->imagePaths, useFileOrder)) {
            return;
        }
        state.opt->lightsFile = selectedPath;
        state.opt->lightsFileByOrder = useFileOrder;
        clearSphereSelection(*state.opt);
        const bool hasMetadata = loadLightsFileMetadata(state.opt->lightsFile, *state.opt);
        if (!hasMetadata) {
            state.opt->lightingModel = LightingModel::Directional;
        }
        if (state.nearFieldCheck != nullptr) {
            setButtonChecked(state.nearFieldCheck, state.opt->lightingModel == LightingModel::NearFieldRing);
        }
        if (state.ringRadiusEdit != nullptr) {
            setEditDouble(state.ringRadiusEdit, state.opt->ringLightRadiusMm);
        }
        if (state.ringHeightEdit != nullptr) {
            setEditDouble(state.ringHeightEdit, state.opt->ringLightHeightMm);
        }
        if (state.pixelScaleEdit != nullptr) {
            setEditDouble(state.pixelScaleEdit, state.opt->pixelScaleMm);
        }
        if (state.shadowLedDiameterEdit != nullptr) {
            setEditDouble(state.shadowLedDiameterEdit, state.opt->shadowLedDiameterMm);
        }
        if (state.lightingCombo != nullptr) {
            SendMessageA(state.lightingCombo, CB_SETCURSEL, 0, 0);
        }
        updateSetupControls(state);
    } catch (const std::exception& e) {
        showOwnerMessage(state.hwnd, "Light Vectors", e.what(), MB_ICONINFORMATION);
    }
}

void clearLightsFile(SetupDialogState& state) {
    state.opt->lightsFile.clear();
    state.opt->lightsFileByOrder = false;
    if (state.nearFieldCheck != nullptr) {
        setButtonChecked(state.nearFieldCheck, state.opt->lightingModel == LightingModel::NearFieldRing);
    }
    updateSetupControls(state);
}

void markSphere(SetupDialogState& state) {
    if (state.opt->imagePaths.empty()) {
        showOwnerMessage(state.hwnd, "Mark Sphere", "Select images first.", MB_ICONWARNING);
        return;
    }
    try {
        SelectionGuard selection(state);
        const SphereSelection chosen = chooseSphereInteractive(
            state.opt->imagePaths.size(),
            [&](size_t imageIndex) { return loadDisplayImage(*state.opt, imageIndex); },
            state.opt->imagePaths,
            static_cast<size_t>(std::max(0, state.opt->sphereSelectionImageIndex)));
        state.opt->sphere = chosen.sphere;
        state.opt->sphereSelectionImageIndex = static_cast<int>(chosen.imageIndex);
        state.opt->sphereEdgePoints = chosen.edgePoints;
        state.opt->sphereFitRmsPixels = chosen.fitRmsPixels;
        state.opt->sphereFitMaxResidualPixels = chosen.fitMaxResidualPixels;
        state.opt->sphereFitCoverageDegrees = chosen.fitCoverageDegrees;
        state.opt->hasSphere = true;
        updateSetupControls(state);
    } catch (const std::exception& e) {
        showOwnerMessage(state.hwnd, "Mark Sphere", e.what(), MB_ICONWARNING);
    }
}

void cropSurface(SetupDialogState& state) {
    if (state.opt->imagePaths.empty()) {
        showOwnerMessage(state.hwnd, "Crop Surface", "Select images first.", MB_ICONWARNING);
        return;
    }
    try {
        SelectionGuard selection(state);
        state.opt->crop = chooseCropInteractive(loadDisplayImage(*state.opt));
        state.opt->hasCrop = true;
        updateSetupControls(state);
    } catch (const std::exception& e) {
        showOwnerMessage(state.hwnd, "Crop Surface", e.what(), MB_ICONWARNING);
    }
}

void markHeightMask(SetupDialogState& state) {
    if (state.opt->imagePaths.empty()) {
        showOwnerMessage(state.hwnd, "Specimen Mask", "Select images first.", MB_ICONWARNING);
        return;
    }
    try {
        SelectionGuard selection(state);
        state.opt->heightMask = chooseHeightMaskInteractive(loadDisplayImage(*state.opt));
        state.opt->hasHeightMask = true;
        state.opt->heightMaskPath.clear();
        updateSetupControls(state);
    } catch (const std::exception& e) {
        showOwnerMessage(state.hwnd, "Specimen Mask", e.what(), MB_ICONWARNING);
    }
}

void markScaleLine(SetupDialogState& state) {
    if (state.opt->imagePaths.empty()) {
        showOwnerMessage(state.hwnd, "Image Scale", "Select images first.", MB_ICONWARNING);
        return;
    }
    try {
        SelectionGuard selection(state);
        const double pixels = chooseScaleLineInteractive(loadDisplayImage(*state.opt));
        std::ostringstream prompt;
        prompt << "The line is " << pixels << " pixels long.\n\n"
               << "Enter the real length of that line:";
        const double knownLengthMm = promptDouble(state.hwnd, "Image Scale", prompt.str(), 1.0, "mm");
        state.opt->pixelScaleMm = knownLengthMm / pixels;
        setEditDouble(state.pixelScaleEdit, state.opt->pixelScaleMm);
        std::ostringstream message;
        message << "Scale line: " << pixels << " pixels\n"
                << "Known length: " << knownLengthMm << " mm\n\n"
                << "Pixel scale: " << state.opt->pixelScaleMm << " mm/pixel";
        showOwnerMessage(state.hwnd, "Image Scale", message.str(), MB_ICONINFORMATION);
        updateSetupControls(state);
    } catch (const std::exception& e) {
        showOwnerMessage(state.hwnd, "Image Scale", e.what(), MB_ICONWARNING);
    }
}

void clearHeightMask(SetupDialogState& state) {
    state.opt->heightMask.release();
    state.opt->hasHeightMask = false;
    state.opt->heightMaskPath.clear();
    updateSetupControls(state);
}

void makeCalibrationTarget(SetupDialogState& state) {
    try {
        const int columns = static_cast<int>(std::lround(promptDouble(
            state.hwnd, "Checkerboard Target", "Number of printed squares across (5-80):",
            state.opt->calibrationGridColumns, "squares")));
        const int rows = static_cast<int>(std::lround(promptDouble(
            state.hwnd, "Checkerboard Target", "Number of printed squares down (5-80):",
            state.opt->calibrationGridRows, "squares")));
        const double squareMm = promptDouble(
            state.hwnd, "Checkerboard Target", "Printed size of each square:",
            state.opt->calibrationSquareMm, "mm");
        const std::string path = chooseCalibrationSaveFile(
            state.hwnd,
            "Save exact-size printable checkerboard",
            "Scalable Vector Graphics\0*.svg\0All files\0*.*\0",
            "svg",
            "what-a-relief-checkerboard.svg");
        writePrintableCheckerboardSvg(
            path,
            columns,
            rows,
            squareMm,
            state.opt->calibrationPageWidthMm,
            state.opt->calibrationPageHeightMm);
        state.opt->calibrationGridColumns = columns;
        state.opt->calibrationGridRows = rows;
        state.opt->calibrationSquareMm = squareMm;
        showOwnerMessage(
            state.hwnd,
            "Checkerboard Target",
            "Target written to:\n" + path +
                "\n\nPrint at 100% / Actual size, disable Fit to page, and verify the 10 mm check line before calibration.",
            MB_ICONINFORMATION);
    } catch (const std::exception& e) {
        showOwnerMessage(state.hwnd, "Checkerboard Target", e.what(), MB_ICONINFORMATION);
    }
}

void createMicroscopeCalibrationInteractive(SetupDialogState& state) {
    try {
        std::vector<std::string> paths = chooseImageFiles(state.hwnd);
        if (paths.size() < 3) throw std::runtime_error("Select one checkerboard image for each of at least three lighting positions.");
        const int columns = static_cast<int>(std::lround(promptDouble(
            state.hwnd, "Microscope Calibration", "Printed checkerboard squares across:",
            state.opt->calibrationGridColumns, "squares")));
        const int rows = static_cast<int>(std::lround(promptDouble(
            state.hwnd, "Microscope Calibration", "Printed checkerboard squares down:",
            state.opt->calibrationGridRows, "squares")));
        const double squareMm = promptDouble(
            state.hwnd, "Microscope Calibration", "Printed size of each checkerboard square:",
            state.opt->calibrationSquareMm, "mm");
        const std::string path = chooseCalibrationSaveFile(
            state.hwnd,
            "Save microscope calibration",
            "Microscope calibration JSON\0*.json\0All files\0*.*\0",
            "json",
            "microscope_calibration.json");

        Options calibrationOptions;
        calibrationOptions.imagePaths = std::move(paths);
        calibrationOptions.inputResponseMode = state.opt->inputResponseMode;
        resolveInputResponses(calibrationOptions);
        SelectionGuard selection(state);
        const MicroscopeCalibration calibration = createMicroscopeCalibration(
            calibrationOptions, columns, rows, squareMm);
        saveMicroscopeCalibration(path, calibration);

        state.opt->microscopeCalibrationFile = path;
        state.opt->estimateLightGains = true;
        state.opt->pixelScaleMm = calibration.pixelScaleMm;
        state.opt->calibrationGridColumns = columns;
        state.opt->calibrationGridRows = rows;
        state.opt->calibrationSquareMm = squareMm;
        clearSphereSelection(*state.opt);
        state.opt->hasCrop = false;
        state.opt->heightMask.release();
        state.opt->hasHeightMask = false;
        setEditDouble(state.pixelScaleEdit, calibration.pixelScaleMm);
        setButtonChecked(state.estimateLightGainsCheck, true);
        std::ostringstream message;
        message << "Calibration saved and selected.\n\n"
                << "Working-plane fit RMS: " << calibration.reprojectionRmsPixels << " pixels\n"
                << "Maximum corner residual: " << calibration.reprojectionMaxPixels << " pixels\n"
                << "Board coverage: " << 100.0 * calibration.boardCoverageFraction << "%\n"
                << "Pixel scale: " << calibration.pixelScaleMm << " mm/pixel\n"
                << "Lighting positions: " << calibration.lightGains.size()
                << "\n\nUse specimen images from the same camera resolution and lighting order. "
                   "The target must have been flat and fronto-parallel at the specimen working plane.";
        showOwnerMessage(state.hwnd, "Microscope Calibration", message.str(), MB_ICONINFORMATION);
        updateSetupControls(state);
    } catch (const std::exception& e) {
        showOwnerMessage(state.hwnd, "Microscope Calibration", e.what(), MB_ICONWARNING);
    }
}

void loadMicroscopeCalibrationInteractive(SetupDialogState& state) {
    try {
        const std::string path = chooseMicroscopeCalibrationFile(state.hwnd);
        const MicroscopeCalibration calibration = loadMicroscopeCalibration(path);
        state.opt->microscopeCalibrationFile = path;
        state.opt->estimateLightGains = true;
        state.opt->pixelScaleMm = calibration.pixelScaleMm;
        clearSphereSelection(*state.opt);
        state.opt->hasCrop = false;
        state.opt->heightMask.release();
        state.opt->hasHeightMask = false;
        setEditDouble(state.pixelScaleEdit, calibration.pixelScaleMm);
        setButtonChecked(state.estimateLightGainsCheck, true);
        updateSetupControls(state);
    } catch (const std::exception& e) {
        showOwnerMessage(state.hwnd, "Microscope Calibration", e.what(), MB_ICONINFORMATION);
    }
}

void clearMicroscopeCalibration(SetupDialogState& state) {
    state.opt->microscopeCalibrationFile.clear();
    state.opt->lightGains.clear();
    state.opt->lightGainSource = "equal";
    state.opt->pixelScaleMm = 0.0;
    clearSphereSelection(*state.opt);
    state.opt->hasCrop = false;
    state.opt->heightMask.release();
    state.opt->hasHeightMask = false;
    setEditDouble(state.pixelScaleEdit, 0.0);
    updateSetupControls(state);
}

void selectMitsubaPython(SetupDialogState& state) {
    try {
        state.opt->mitsubaPythonPath = chooseMitsubaPython(state.hwnd);
        updateSetupControls(state);
    } catch (const std::exception& e) {
        showOwnerMessage(state.hwnd, "Mitsuba Backend", e.what(), MB_ICONINFORMATION);
    }
}

bool validateAndAccept(SetupDialogState& state) {
    Options& opt = *state.opt;
    if (opt.imagePaths.size() < kMinImages) {
        showOwnerMessage(
            state.hwnd,
            "Setup",
            "Select at least 3 images. You selected " + std::to_string(opt.imagePaths.size()) + ".",
            MB_ICONWARNING);
        return false;
    }
    if (opt.outputDir.empty()) {
        showOwnerMessage(state.hwnd, "Setup", "Choose an output folder.", MB_ICONWARNING);
        return false;
    }

    opt.uncalibratedLighting = comboSelection(state.lightingCombo) == 1;
    opt.estimateLightGains = !opt.uncalibratedLighting && buttonChecked(state.estimateLightGainsCheck);
    if (opt.uncalibratedLighting) {
        opt.lightsFile.clear();
        opt.lightsFileByOrder = false;
    }
    if (opt.uncalibratedLighting && opt.imagePaths.size() < 4) {
        showOwnerMessage(state.hwnd, "Setup", "Uncalibrated no-sphere mode requires at least 4 images.", MB_ICONWARNING);
        return false;
    }
    if (!opt.uncalibratedLighting && !opt.lightsFile.empty()) {
        if (opt.lightsFileByOrder) {
            try {
                (void)loadLightsFile(opt.lightsFile, opt.imagePaths, nullptr, true);
            } catch (const std::exception& e) {
                showOwnerMessage(
                    state.hwnd,
                    "Setup",
                    std::string("The order-based calibration is no longer valid.\n\n") + e.what(),
                    MB_ICONWARNING);
                return false;
            }
        } else {
            bool useFileOrder = false;
            if (!confirmLightsFileBinding(
                    state.hwnd, opt.lightsFile, opt.imagePaths, useFileOrder)) {
                return false;
            }
            opt.lightsFileByOrder = useFileOrder;
            updateSetupControls(state);
        }
    }
    if (!opt.uncalibratedLighting && buttonChecked(state.neuralFusionCheck) &&
        (opt.imagePaths.size() < kMinImages || opt.imagePaths.size() > kMaxNeuralImages)) {
        showOwnerMessage(
            state.hwnd,
            "Setup",
            "Experimental neural fusion currently supports 3 to 25 calibrated images.",
            MB_ICONWARNING);
        return false;
    }
    if (!opt.uncalibratedLighting && opt.lightsFile.empty() && !opt.hasSphere) {
        const int answer = MessageBoxA(
            state.hwnd,
            "Calibrated mode needs the highlight sphere.\n\nOpen the sphere marker now?",
            "Setup",
            MB_YESNO | MB_ICONQUESTION | MB_DEFBUTTON1);
        if (answer != IDYES) {
            return false;
        }
        markSphere(state);
        if (!opt.hasSphere) {
            return false;
        }
    }

    opt.solverMode = comboSelection(state.solverCombo) == 0 ? NormalSolverMode::Robust : NormalSolverMode::Standard;
    try {
        opt.pixelScaleMm = editDouble(state.pixelScaleEdit, "pixel scale");
        opt.shadowReferenceZMm = editDouble(
            state.shadowReferenceZEdit, "shadow reference surface Z");
        opt.shadowLedDiameterMm = editDouble(
            state.shadowLedDiameterEdit, "shadow LED diameter");
        opt.mitsubaLightAngleDegrees = editDouble(state.mitsubaLightAngleEdit, "source angular diameter");
    } catch (const std::exception& e) {
        showOwnerMessage(state.hwnd, "Setup", e.what(), MB_ICONWARNING);
        return false;
    }
    if (!std::isfinite(opt.pixelScaleMm) || opt.pixelScaleMm < 0.0) {
        showOwnerMessage(state.hwnd, "Setup", "Pixel scale must be non-negative. Use 0 to read TIFF tags when possible.", MB_ICONWARNING);
        return false;
    }
    if (!opt.uncalibratedLighting && buttonChecked(state.nearFieldCheck)) {
        opt.lightingModel = LightingModel::NearFieldRing;
        try {
            opt.ringLightRadiusMm = editDouble(state.ringRadiusEdit, "ring light radius");
            opt.ringLightHeightMm = editDouble(state.ringHeightEdit, "ring light height");
        } catch (const std::exception& e) {
            showOwnerMessage(state.hwnd, "Setup", e.what(), MB_ICONWARNING);
            return false;
        }
        if (opt.pixelScaleMm <= 0.0 && !opt.imagePaths.empty()) {
            opt.pixelScaleMm = readPixelScaleMmFromImage(opt.imagePaths.front());
            if (opt.pixelScaleMm > 0.0) {
                setEditDouble(state.pixelScaleEdit, opt.pixelScaleMm);
            }
        }
        if (!std::isfinite(opt.ringLightRadiusMm) || opt.ringLightRadiusMm <= 0.0 ||
            !std::isfinite(opt.ringLightHeightMm) || opt.ringLightHeightMm <= 0.0) {
            showOwnerMessage(state.hwnd, "Setup", "Ring light radius and height must be positive millimeter values.", MB_ICONWARNING);
            return false;
        }
        if (!std::isfinite(opt.pixelScaleMm) || opt.pixelScaleMm <= 0.0) {
            showOwnerMessage(
                state.hwnd,
                "Setup",
                "Near-field ring lighting needs a pixel scale in mm/pixel.\n\n"
                "Enter it manually, or use a TIFF with readable physical scale tags.",
                MB_ICONWARNING);
            return false;
        }
    } else {
        opt.lightingModel = LightingModel::Directional;
    }
    switch (comboSelection(state.flattenCombo)) {
    case 1:
        opt.flattenMode = FlattenMode::Gentle;
        break;
    case 2:
        opt.flattenMode = FlattenMode::Strong;
        break;
    default:
        opt.flattenMode = FlattenMode::None;
        break;
    }
    opt.inputResponseMode = static_cast<InputResponseMode>(comboSelection(state.responseCombo));
    try {
        resolveInputResponses(opt);
    } catch (const std::exception& e) {
        showOwnerMessage(state.hwnd, "Input Response (Processing tab)", e.what(), MB_ICONWARNING);
        return false;
    }
    if (!opt.microscopeCalibrationFile.empty()) {
        try {
            const MicroscopeCalibration calibration = loadMicroscopeCalibration(opt.microscopeCalibrationFile);
            if (calibration.lightGains.size() != opt.imagePaths.size()) {
                showOwnerMessage(
                    state.hwnd,
                    "Microscope Calibration",
                    "The calibration has " + std::to_string(calibration.lightGains.size()) +
                        " lighting positions, but the specimen stack has " +
                        std::to_string(opt.imagePaths.size()) +
                        ". Select the same positions in the same order.",
                    MB_ICONWARNING);
                return false;
            }
            opt.pixelScaleMm = calibration.pixelScaleMm;
            setEditDouble(state.pixelScaleEdit, opt.pixelScaleMm);
        } catch (const std::exception& e) {
            showOwnerMessage(state.hwnd, "Microscope Calibration", e.what(), MB_ICONWARNING);
            return false;
        }
    }
    opt.calculateHeight = buttonChecked(state.heightCheck);
    opt.meshStep = editInt(state.meshStepEdit, "mesh step");
    opt.printableThicknessMm = editDouble(state.printableThicknessEdit, "printable base thickness");
    if (opt.meshStep < 1) {
        showOwnerMessage(state.hwnd, "Setup", "Mesh step must be at least 1.", MB_ICONWARNING);
        return false;
    }
    if (!std::isfinite(opt.printableThicknessMm) || opt.printableThicknessMm <= 0.0) {
        showOwnerMessage(state.hwnd, "Setup", "Printable base thickness must be positive.", MB_ICONWARNING);
        return false;
    }
    opt.heightSolverMode = comboSelection(state.heightSolverCombo) == 1 ? HeightSolverMode::FastDct : HeightSolverMode::RobustMasked;
    switch (comboSelection(state.heightFlattenCombo)) {
    case 1:
        opt.heightFlattenMode = HeightFlattenMode::Plane;
        break;
    case 2:
        opt.heightFlattenMode = HeightFlattenMode::Radial;
        break;
    case 3:
        opt.heightFlattenMode = HeightFlattenMode::Quadratic;
        break;
    default:
        opt.heightFlattenMode = HeightFlattenMode::None;
        break;
    }
    if (buttonChecked(state.meshCheck)) {
        opt.calculateHeight = true;
        opt.meshPath = (fs::path(opt.outputDir) / "surface.ply").string();
    } else {
        opt.meshPath.clear();
    }
    if (buttonChecked(state.printableMeshCheck)) {
        opt.calculateHeight = true;
        opt.printableMeshPath = (fs::path(opt.outputDir) / "printable_surface.ply").string();
        opt.printableFillHoles = buttonChecked(state.printableFillHolesCheck);
        if (opt.pixelScaleMm <= 0.0 && !opt.imagePaths.empty()) {
            opt.pixelScaleMm = readPixelScaleMmFromImage(opt.imagePaths.front());
            if (opt.pixelScaleMm > 0.0) {
                setEditDouble(state.pixelScaleEdit, opt.pixelScaleMm);
            }
        }
        if (!std::isfinite(opt.pixelScaleMm) || opt.pixelScaleMm <= 0.0) {
            showOwnerMessage(
                state.hwnd,
                "Setup",
                "Printable mesh export needs XY scale. Enter mm/pixel on the Project tab or draw a scale line.",
                MB_ICONWARNING);
            return false;
        }
    } else {
        opt.printableMeshPath.clear();
        opt.printableFillHoles = false;
    }
    if (!std::isfinite(opt.shadowReferenceZMm) ||
        !std::isfinite(opt.shadowLedDiameterMm) || opt.shadowLedDiameterMm < 0.0) {
        showOwnerMessage(
            state.hwnd,
            "Setup",
            "Shadow reference Z must be finite and LED diameter must be non-negative.",
            MB_ICONWARNING);
        return false;
    }
    opt.exportRti = !opt.uncalibratedLighting && opt.imagePaths.size() >= kMinImages && buttonChecked(state.rtiCheck);
    const int rtiLayoutIndex = comboSelection(state.rtiLayoutCombo);
    if (rtiLayoutIndex == 1) {
        opt.rtiLayoutMode = RtiLayoutMode::DeepZoom;
    } else if (rtiLayoutIndex == 2) {
        opt.rtiLayoutMode = RtiLayoutMode::WebRtiViewer;
    } else {
        opt.rtiLayoutMode = RtiLayoutMode::Image;
    }
    opt.rtiColorMode = comboSelection(state.rtiColorCombo) == 1 ? RtiColorMode::Lrgb : RtiColorMode::Rgb;
    opt.rtiPath = opt.exportRti ? (fs::path(opt.outputDir) / "rti").string() : std::string();
    opt.openRelightViewer = buttonChecked(state.relightCheck);
    opt.specularDiagnostics = buttonChecked(state.specularDiagnosticsCheck);
    opt.shadowHeightRefinement = !opt.uncalibratedLighting &&
        opt.solverMode == NormalSolverMode::Robust &&
        opt.calculateHeight &&
        opt.imagePaths.size() >= 6 &&
        buttonChecked(state.shadowHeightRefinementCheck);
    if (opt.shadowHeightRefinement &&
        opt.lightingModel == LightingModel::NearFieldRing &&
        opt.shadowReferenceZMm >= opt.ringLightHeightMm) {
        showOwnerMessage(
            state.hwnd,
            "Setup",
            "The shadow reference surface Z must be below the ring-light height.",
            MB_ICONWARNING);
        return false;
    }
    opt.mitsubaInverseRefinement = !opt.uncalibratedLighting &&
        opt.solverMode == NormalSolverMode::Robust &&
        opt.calculateHeight &&
        opt.imagePaths.size() >= 6 &&
        buttonChecked(state.mitsubaInverseCheck);
    const int mitsubaBackendIndex = comboSelection(state.mitsubaBackendCombo);
    opt.mitsubaBackendMode = mitsubaBackendIndex == 1
        ? MitsubaBackendMode::Cuda
        : (mitsubaBackendIndex == 2 ? MitsubaBackendMode::Cpu : MitsubaBackendMode::Auto);
    const int mitsubaQualityIndex = comboSelection(state.mitsubaQualityCombo);
    if (mitsubaQualityIndex == 0) {
        opt.mitsubaQualityMode = MitsubaQualityMode::Preview;
    } else if (mitsubaQualityIndex == 2) {
        opt.mitsubaQualityMode = MitsubaQualityMode::Research;
    } else {
        opt.mitsubaQualityMode = MitsubaQualityMode::Standard;
    }
    if (opt.mitsubaInverseRefinement && opt.shadowHeightRefinement) {
        showOwnerMessage(
            state.hwnd,
            "Setup",
            "Choose either Mitsuba inverse refinement or cast-shadow height refinement, not both.",
            MB_ICONWARNING);
        return false;
    }
    if (opt.mitsubaInverseRefinement) {
        if (opt.mitsubaQualityMode == MitsubaQualityMode::Ultra) {
            const int answer = MessageBoxA(
                state.hwnd,
                "Ultra runs the inverse renderer on a grid capped at 1024 pixels on the longer side of the fitting-mask bounds.\n\n"
                "It optimizes an absolute height field initialized by the classical reconstruction, rather than "
                "an upsampled bounded correction. The classical result remains the validation baseline and fallback.\n\n"
                "Runtime and memory grow with pixel count, light count, samples, and iterations. Large images can "
                "take many hours or exhaust GPU/RAM. Ultra automatically lowers stochastic optimization samples "
                "to limit peak renderer memory. Inputs larger than the 1024-pixel limit are area-reduced for the "
                "solve and accepted height is interpolated back to the source grid. The original "
                "height is retained if the worker fails or its "
                "candidate does not pass validation.\n\nContinue with Ultra?",
                "Ultra Mitsuba Inverse Solution",
                MB_YESNO | MB_ICONWARNING | MB_DEFBUTTON2);
            if (answer != IDYES) {
                return false;
            }
        }
        if (opt.lightingModel == LightingModel::NearFieldRing &&
            (opt.shadowLedDiameterMm <= 0.0 || opt.shadowReferenceZMm >= opt.ringLightHeightMm)) {
            showOwnerMessage(state.hwnd, "Setup", "Near-field inverse refinement requires a positive LED diameter and a reference Z below the ring lights (Advanced tab).", MB_ICONWARNING);
            return false;
        }
        if (!std::isfinite(opt.mitsubaLightAngleDegrees) || opt.mitsubaLightAngleDegrees < 0.1 || opt.mitsubaLightAngleDegrees > 10.0) {
            showOwnerMessage(state.hwnd, "Setup", "Source angular diameter must be between 0.1 and 10 degrees.", MB_ICONWARNING);
            return false;
        }
        const MitsubaBackendPaths backend = resolveMitsubaBackend(opt);
        if (!backend.available()) {
            showOwnerMessage(state.hwnd, "Mitsuba Backend", backend.problem, MB_ICONWARNING);
            return false;
        }
    }
    opt.neuralFusion = !opt.uncalibratedLighting &&
        opt.imagePaths.size() >= kMinImages &&
        opt.imagePaths.size() <= kMaxNeuralImages &&
        buttonChecked(state.neuralFusionCheck);
    if (opt.neuralFusion) {
        showOwnerMessage(
            state.hwnd,
            "Neural Fusion Height Note",
            "Experimental neural fusion only affects the normal-map style outputs.\n\n"
            "Height preview and PLY mesh stay on the classical geometry path to avoid exaggerated shape.",
            MB_ICONINFORMATION);
    }
    return true;
}

void createSetupControls(HWND hwnd, SetupDialogState& state) {
    state.initializing = true;
    state.requiredBrush = CreateSolidBrush(RGB(255, 238, 205));
    state.optionalBrush = CreateSolidBrush(RGB(232, 244, 255));
    state.sectionBrush = CreateSolidBrush(RGB(232, 236, 242));

    state.nextStepLabel = makeControl(hwnd, "STATIC", "", SS_LEFT, kIdNextStep, kMargin, 16, kWindowWidth - 2 * kMargin - 16, 34);
    state.tabControl = makeControl(hwnd, WC_TABCONTROLA, "", WS_TABSTOP, kIdTabs, kMargin, kTabTop, kWindowWidth - 2 * kMargin - 16, kTabHeight);

    HWND projectPage = makeTabPage(hwnd, state, "Project");
    int y = 16;
    makeLabel(projectPage, "Images", kMargin, y, kLabelWidth, kRowHeight);
    makeControl(projectPage, "BUTTON", "Select Images...", BS_PUSHBUTTON, kIdSelectImages, kControlX, y, kButtonWidth, kRowHeight);
    state.imageStatus = makeControl(projectPage, "STATIC", "", SS_LEFT, kIdImageStatus, kControlX + kButtonWidth + 12, y + 5, 330, kRowHeight);

    y += 42;
    makeLabel(projectPage, "Output Folder", kMargin, y, kLabelWidth, kRowHeight);
    makeControl(projectPage, "BUTTON", "Choose Folder...", BS_PUSHBUTTON, kIdSelectOutput, kControlX, y, kButtonWidth, kRowHeight);
    state.outputStatus = makeControl(projectPage, "STATIC", "", SS_LEFT, kIdOutputStatus, kControlX + kButtonWidth + 12, y + 5, 330, kRowHeight);

    y += 42;
    makeLabel(projectPage, "Pixel Scale", kMargin, y, kLabelWidth, kRowHeight);
    state.pixelScaleEdit = makeControl(projectPage, "EDIT", "", ES_LEFT | WS_BORDER | WS_TABSTOP, kIdPixelScale, kControlX, y, 120, kRowHeight);
    setEditDouble(state.pixelScaleEdit, state.opt->pixelScaleMm);
    makeControl(projectPage, "STATIC", "mm/pixel; 0 reads TIFF tags", SS_LEFT, 0, kControlX + 135, y + 5, 300, kRowHeight);

    y += 36;
    makeLabel(projectPage, "Scale Line", kMargin, y, kLabelWidth, kRowHeight);
    state.scaleButton = makeControl(projectPage, "BUTTON", "Draw Scale Line...", BS_PUSHBUTTON, kIdMarkScale, kControlX, y, 150, kRowHeight);
    makeControl(projectPage, "STATIC", "enter length after drawing", SS_LEFT, 0, kControlX + 165, y + 5, 260, kRowHeight);

    HWND lightingPage = makeTabPage(hwnd, state, "Lighting");
    y = 16;
    makeLabel(lightingPage, "Previous Calib.", kMargin, y, kLabelWidth, kRowHeight);
    makeControl(lightingPage, "BUTTON", "Load CSV...", BS_PUSHBUTTON, kIdSelectLights, kControlX, y, kButtonWidth, kRowHeight);
    makeControl(lightingPage, "BUTTON", "Clear", BS_PUSHBUTTON, kIdClearLights, kControlX + kButtonWidth + 12, y, 70, kRowHeight);
    state.lightsStatus = makeControl(lightingPage, "STATIC", "", SS_LEFT, kIdLightsStatus, kControlX + kButtonWidth + 94, y + 5, 260, kRowHeight);

    y += 42;
    makeLabel(lightingPage, "Lighting", kMargin, y, kLabelWidth, kRowHeight);
    state.lightingCombo = makeCombo(lightingPage, kIdLighting, kControlX, y, kControlWidth);
    addComboItem(state.lightingCombo, "Calibrated: loaded CSV or mark sphere");
    addComboItem(state.lightingCombo, "No sphere: estimate unknown lighting (relative)");
    SendMessageA(state.lightingCombo, CB_SETCURSEL, state.opt->uncalibratedLighting ? 1 : 0, 0);

    y += 42;
    makeLabel(lightingPage, "Sphere", kMargin, y, kLabelWidth, kRowHeight);
    state.sphereButton = makeControl(lightingPage, "BUTTON", "Mark Sphere...", BS_PUSHBUTTON, kIdMarkSphere, kControlX, y, kButtonWidth, kRowHeight);
    state.sphereStatus = makeControl(lightingPage, "STATIC", "", SS_LEFT, kIdSphereStatus, kControlX + kButtonWidth + 12, y + 5, 330, kRowHeight);

    y += 42;
    makeLabel(lightingPage, "Microscope", kMargin, y, kLabelWidth, kRowHeight);
    makeControl(lightingPage, "BUTTON", "Make Target...", BS_PUSHBUTTON | WS_TABSTOP,
        kIdMakeCalibrationTarget, kControlX, y, 112, kRowHeight);
    makeControl(lightingPage, "BUTTON", "Calibrate...", BS_PUSHBUTTON | WS_TABSTOP,
        kIdCreateMicroscopeCalibration, kControlX + 120, y, 96, kRowHeight);
    makeControl(lightingPage, "BUTTON", "Load...", BS_PUSHBUTTON | WS_TABSTOP,
        kIdLoadMicroscopeCalibration, kControlX + 224, y, 76, kRowHeight);
    makeControl(lightingPage, "BUTTON", "Clear", BS_PUSHBUTTON | WS_TABSTOP,
        kIdClearMicroscopeCalibration, kControlX + 308, y, 62, kRowHeight);

    y += 32;
    state.microscopeCalibrationStatus = makeControl(
        lightingPage, "STATIC", "", SS_LEFT, kIdMicroscopeCalibrationStatus,
        kControlX, y + 5, kControlWidth, kRowHeight);

    y += 34;
    state.estimateLightGainsCheck = makeControl(
        lightingPage,
        "BUTTON",
        "Estimate per-stack relative light balance (validated)",
        BS_AUTOCHECKBOX | WS_TABSTOP,
        kIdEstimateLightGains,
        kControlX,
        y,
        kControlWidth,
        24);
    setButtonChecked(state.estimateLightGainsCheck, state.opt->estimateLightGains);

    y += 34;
    makeLabel(lightingPage, "Near Field", kMargin, y, kLabelWidth, kRowHeight);
    state.nearFieldCheck = makeControl(lightingPage, "BUTTON", "Use ring light point-source model", BS_AUTOCHECKBOX, kIdNearField, kControlX, y, kControlWidth, 24);
    setButtonChecked(state.nearFieldCheck, state.opt->lightingModel == LightingModel::NearFieldRing);

    y += 34;
    makeLabel(lightingPage, "Ring Radius", kMargin, y, kLabelWidth, kRowHeight);
    state.ringRadiusEdit = makeControl(lightingPage, "EDIT", "", ES_LEFT | WS_BORDER | WS_TABSTOP, kIdRingRadius, kControlX, y, 120, kRowHeight);
    setEditDouble(state.ringRadiusEdit, state.opt->ringLightRadiusMm);
    makeControl(lightingPage, "STATIC", "mm from image/crop center", SS_LEFT, 0, kControlX + 135, y + 5, 300, kRowHeight);

    y += 34;
    makeLabel(lightingPage, "Light Height", kMargin, y, kLabelWidth, kRowHeight);
    state.ringHeightEdit = makeControl(lightingPage, "EDIT", "", ES_LEFT | WS_BORDER | WS_TABSTOP, kIdRingHeight, kControlX, y, 120, kRowHeight);
    setEditDouble(state.ringHeightEdit, state.opt->ringLightHeightMm);
    makeControl(lightingPage, "STATIC", "mm above sample plane", SS_LEFT, 0, kControlX + 135, y + 5, 300, kRowHeight);

    HWND geometryPage = makeTabPage(hwnd, state, "Geometry");
    y = 16;
    makeLabel(geometryPage, "Crop", kMargin, y, kLabelWidth, kRowHeight);
    state.cropButton = makeControl(geometryPage, "BUTTON", "Crop Surface...", BS_PUSHBUTTON, kIdCrop, kControlX, y, kButtonWidth, kRowHeight);
    state.cropStatus = makeControl(geometryPage, "STATIC", "", SS_LEFT, kIdCropStatus, kControlX + kButtonWidth + 12, y + 5, 330, kRowHeight);

    y += 42;
    makeLabel(geometryPage, "Height Mask", kMargin, y, kLabelWidth, kRowHeight);
    state.heightMaskButton = makeControl(geometryPage, "BUTTON", "Mark Specimen...", BS_PUSHBUTTON, kIdHeightMask, kControlX, y, kButtonWidth, kRowHeight);
    state.clearHeightMaskButton = makeControl(geometryPage, "BUTTON", "Clear", BS_PUSHBUTTON, kIdClearHeightMask, kControlX + kButtonWidth + 12, y, 70, kRowHeight);
    state.heightMaskStatus = makeControl(geometryPage, "STATIC", "", SS_LEFT, kIdHeightMaskStatus, kControlX + kButtonWidth + 94, y + 5, 260, kRowHeight);

    y += 42;
    state.heightCheck = makeControl(geometryPage, "BUTTON", "Calculate height preview", BS_AUTOCHECKBOX, kIdHeight, kControlX, y, kControlWidth, 24);
    setButtonChecked(state.heightCheck, state.opt->calculateHeight || !state.opt->meshPath.empty() || !state.opt->printableMeshPath.empty());

    y += 36;
    makeLabel(geometryPage, "Height Solver", kMargin, y, kLabelWidth, kRowHeight);
    state.heightSolverCombo = makeCombo(geometryPage, kIdHeightSolver, kControlX, y, kControlWidth);
    addComboItem(state.heightSolverCombo, "Robust masked (weighted, less warping)");
    addComboItem(state.heightSolverCombo, "Fast DCT/Poisson");
    SendMessageA(state.heightSolverCombo, CB_SETCURSEL, state.opt->heightSolverMode == HeightSolverMode::FastDct ? 1 : 0, 0);

    y += 40;
    makeLabel(geometryPage, "Height Drift Correction", kMargin, y, kLabelWidth, kRowHeight);
    state.heightFlattenCombo = makeCombo(geometryPage, kIdHeightFlatten, kControlX, y, kControlWidth);
    addComboItem(state.heightFlattenCombo, "None (keep broad integrated shape)");
    addComboItem(state.heightFlattenCombo, "Plane leveling (least-squares; height and PLY only)");
    addComboItem(state.heightFlattenCombo, "Radial/dome correction (height and PLY only)");
    addComboItem(state.heightFlattenCombo, "Quadratic correction (height and PLY only)");
    int heightFlattenIndex = 0;
    if (state.opt->heightFlattenMode == HeightFlattenMode::Plane) {
        heightFlattenIndex = 1;
    } else if (state.opt->heightFlattenMode == HeightFlattenMode::Radial) {
        heightFlattenIndex = 2;
    } else if (state.opt->heightFlattenMode == HeightFlattenMode::Quadratic) {
        heightFlattenIndex = 3;
    }
    SendMessageA(state.heightFlattenCombo, CB_SETCURSEL, heightFlattenIndex, 0);

    y += 40;
    state.meshCheck = makeControl(geometryPage, "BUTTON", "Export PLY mesh from height preview", BS_AUTOCHECKBOX, kIdMesh, kControlX, y, kControlWidth, 24);
    setButtonChecked(state.meshCheck, !state.opt->meshPath.empty());

    y += 32;
    state.printableMeshCheck = makeControl(geometryPage, "BUTTON", "Export printable solids (also inverse, when enabled)", BS_AUTOCHECKBOX, kIdPrintableMesh, kControlX, y, kControlWidth, 24);
    setButtonChecked(state.printableMeshCheck, !state.opt->printableMeshPath.empty());

    y += 32;
    state.printableFillHolesCheck = makeControl(geometryPage, "BUTTON", "Smart-fill enclosed surface holes (printable PLY only)", BS_AUTOCHECKBOX, kIdPrintableFillHoles, kControlX, y, kControlWidth, 24);
    setButtonChecked(state.printableFillHolesCheck, state.opt->printableFillHoles);

    y += 36;
    makeLabel(geometryPage, "Mesh Step", kMargin, y, kLabelWidth, kRowHeight);
    state.meshStepEdit = makeControl(geometryPage, "EDIT", "", ES_LEFT | WS_BORDER | WS_TABSTOP, kIdMeshStep, kControlX, y, 120, kRowHeight);
    setEditInt(state.meshStepEdit, state.opt->meshStep);
    makeLabel(geometryPage, "1 = full; 2 = about 1/4 as many vertices", kControlX + 130, y, 300, kRowHeight);

    y += 40;
    makeLabel(geometryPage, "Print Base", kMargin, y, kLabelWidth, kRowHeight);
    state.printableThicknessEdit = makeControl(geometryPage, "EDIT", "", ES_LEFT | WS_BORDER | WS_TABSTOP, kIdPrintableThickness, kControlX, y, 120, kRowHeight);
    setEditDouble(state.printableThicknessEdit, state.opt->printableThicknessMm);
    makeLabel(geometryPage, "mm thick", kControlX + 130, y, 120, kRowHeight);

    HWND processingPage = makeTabPage(hwnd, state, "Processing");
    y = 16;
    makeLabel(processingPage, "Normal Solver", kMargin, y, kLabelWidth, kRowHeight);
    state.solverCombo = makeCombo(processingPage, kIdSolver, kControlX, y, kControlWidth);
    addComboItem(state.solverCombo, "Robust: reject shadows/highlights and downweight outliers");
    addComboItem(state.solverCombo, "Standard least squares");
    SendMessageA(state.solverCombo, CB_SETCURSEL, state.opt->solverMode == NormalSolverMode::Standard ? 1 : 0, 0);

    y += 28;
    makeLabel(processingPage, "Robust rejection needs 4+ usable lights per pixel; 3-light pixels use least squares.", kControlX, y, kControlWidth, kRowHeight);

    y += 36;
    makeLabel(processingPage, "Relief Flattening", kMargin, y, kLabelWidth, kRowHeight);
    state.flattenCombo = makeCombo(processingPage, kIdFlatten, kControlX, y, kControlWidth);
    addComboItem(state.flattenCombo, "None (no slope filtering)");
    addComboItem(state.flattenCombo, "Gentle (Gaussian scale-space slope subtraction)");
    addComboItem(state.flattenCombo, "Strong (Gaussian scale-space slope subtraction)");
    int flattenIndex = 0;
    if (state.opt->flattenMode == FlattenMode::Gentle) {
        flattenIndex = 1;
    } else if (state.opt->flattenMode == FlattenMode::Strong) {
        flattenIndex = 2;
    }
    SendMessageA(state.flattenCombo, CB_SETCURSEL, flattenIndex, 0);

    y += 42;
    makeLabel(processingPage, "Input Response", kMargin, y, kLabelWidth, kRowHeight);
    state.responseCombo = makeCombo(processingPage, kIdSrgb, kControlX, y, kControlWidth);
    addComboItem(state.responseCombo, "Auto (color metadata; report untagged assumptions)");
    addComboItem(state.responseCombo, "Linear (ignore color metadata; do not decode)");
    addComboItem(state.responseCombo, "sRGB (ignore color metadata; decode once)");
    SendMessageA(state.responseCombo, CB_SETCURSEL, static_cast<int>(state.opt->inputResponseMode), 0);
    y += 30;
    state.responseStatus = makeLabel(processingPage, "", kControlX, y, kControlWidth, 64);
    refreshInputResponse(state);

    HWND outputsPage = makeTabPage(hwnd, state, "Outputs");
    y = 16;
    state.relightCheck = makeControl(outputsPage, "BUTTON", "Open interactive relight viewer after processing", BS_AUTOCHECKBOX, kIdRelight, kControlX, y, kControlWidth, 24);
    setButtonChecked(state.relightCheck, state.opt->openRelightViewer);

    y += 36;
    state.rtiCheck = makeControl(outputsPage, "BUTTON", "Export RTI package (3+ calibrated images; small stacks use 3-term PTM)", BS_AUTOCHECKBOX, kIdRtiExport, kControlX, y, kControlWidth, 24);
    setButtonChecked(state.rtiCheck, state.opt->exportRti);

    y += 38;
    makeLabel(outputsPage, "RTI Layout", kMargin, y, kLabelWidth, kRowHeight);
    state.rtiLayoutCombo = makeCombo(outputsPage, kIdRtiLayout, kControlX, y, kControlWidth);
    addComboItem(state.rtiLayoutCombo, "Image package (info.json + plane JPEGs)");
    addComboItem(state.rtiLayoutCombo, "DeepZoom package (tiled pyramids)");
    addComboItem(state.rtiLayoutCombo, "webRTIViewer folder (info.xml + component tiles)");
    int rtiLayoutIndex = 0;
    if (state.opt->rtiLayoutMode == RtiLayoutMode::DeepZoom) {
        rtiLayoutIndex = 1;
    } else if (state.opt->rtiLayoutMode == RtiLayoutMode::WebRtiViewer) {
        rtiLayoutIndex = 2;
    }
    SendMessageA(state.rtiLayoutCombo, CB_SETCURSEL, rtiLayoutIndex, 0);

    y += 42;
    makeLabel(outputsPage, "RTI Color", kMargin, y, kLabelWidth, kRowHeight);
    state.rtiColorCombo = makeCombo(outputsPage, kIdRtiColor, kControlX, y, kControlWidth);
    addComboItem(state.rtiColorCombo, "RGB PTM (direct color coefficients)");
    addComboItem(state.rtiColorCombo, "LRGB PTM (base image + luminance coefficients)");
    SendMessageA(state.rtiColorCombo, CB_SETCURSEL, state.opt->rtiColorMode == RtiColorMode::Lrgb ? 1 : 0, 0);

    HWND advancedPage = makeTabPage(hwnd, state, "Advanced");
    y = 16;
    state.specularDiagnosticsCheck = makeControl(advancedPage, "BUTTON", "Write additional robust diagnostic images (summary and per-light)", BS_AUTOCHECKBOX, kIdSpecularDiagnostics, kControlX, y, kControlWidth, 24);
    setButtonChecked(state.specularDiagnosticsCheck, state.opt->specularDiagnostics);

    y += 36;
    state.shadowHeightRefinementCheck = makeControl(advancedPage, "BUTTON", "Experimental: refine broad height/mesh shape from global cast shadows", BS_AUTOCHECKBOX, kIdShadowHeightRefinement, kControlX, y, kControlWidth, 24);
    setButtonChecked(state.shadowHeightRefinementCheck, state.opt->shadowHeightRefinement);

    y += 26;
    makeLabel(advancedPage, "Requires 6+ calibrated robust images and height; sphere directions are supported.", kControlX + 20, y, kControlWidth - 20, 18);

    y += 20;
    makeLabel(advancedPage, "Z and LED size are near-field only; weak evidence is rejected unchanged.", kControlX + 20, y, kControlWidth - 20, kRowHeight);

    y += 34;
    makeLabel(advancedPage, "Reference Surface Z", kMargin, y, kLabelWidth, kRowHeight);
    state.shadowReferenceZEdit = makeControl(advancedPage, "EDIT", "", ES_LEFT | WS_BORDER | WS_TABSTOP, kIdShadowReferenceZ, kControlX, y, 120, kRowHeight);
    setEditDouble(state.shadowReferenceZEdit, state.opt->shadowReferenceZMm);
    makeLabel(advancedPage, "near-field mm above calibration datum", kControlX + 135, y, 290, kRowHeight);

    y += 34;
    state.sourceSizeLabel = makeLabel(advancedPage, "LED Diameter", kMargin, y, kLabelWidth, kRowHeight);
    state.shadowLedDiameterEdit = makeControl(advancedPage, "EDIT", "", ES_LEFT | WS_BORDER | WS_TABSTOP, kIdShadowLedDiameter, kControlX, y, 120, kRowHeight);
    setEditDouble(state.shadowLedDiameterEdit, state.opt->shadowLedDiameterMm);
    state.mitsubaLightAngleEdit = makeControl(advancedPage, "EDIT", "", ES_LEFT | WS_BORDER | WS_TABSTOP, kIdMitsubaLightAngle, kControlX, y, 120, kRowHeight);
    setEditDouble(state.mitsubaLightAngleEdit, state.opt->mitsubaLightAngleDegrees);
    state.sourceSizeUnits = makeLabel(advancedPage, "mm; inverse requires >0; shadow allows 0", kControlX + 135, y, 290, kRowHeight);

    y += 36;
    state.neuralFusionCheck = makeControl(advancedPage, "BUTTON", "Experimental: PS-FCN neural prior + fusion (3 to 25 calibrated images)", BS_AUTOCHECKBOX, kIdNeuralFusion, kControlX, y, kControlWidth, 24);
    setButtonChecked(state.neuralFusionCheck, state.opt->neuralFusion);

    y += 38;
    state.mitsubaInverseCheck = makeControl(advancedPage, "BUTTON", "Experimental: Mitsuba inverse geometry", BS_AUTOCHECKBOX, kIdMitsubaInverse, kControlX, y, kControlWidth, 24);
    setButtonChecked(state.mitsubaInverseCheck, state.opt->mitsubaInverseRefinement);

    y += 25;
    makeLabel(advancedPage, "Requires 6+ calibrated robust images and height; writes separate inverse outputs.", kControlX + 20, y, kControlWidth - 20, kRowHeight);

    y += 30;
    makeLabel(advancedPage, "Compute", kMargin, y, kLabelWidth, kRowHeight);
    state.mitsubaBackendCombo = makeCombo(advancedPage, kIdMitsubaBackend, kControlX, y, kControlWidth);
    addComboItem(state.mitsubaBackendCombo, "Auto (NVIDIA CUDA, then LLVM CPU)");
    addComboItem(state.mitsubaBackendCombo, "NVIDIA GPU (CUDA)");
    addComboItem(state.mitsubaBackendCombo, "CPU (LLVM)");
    int mitsubaBackendIndex = 0;
    if (state.opt->mitsubaBackendMode == MitsubaBackendMode::Cuda) {
        mitsubaBackendIndex = 1;
    } else if (state.opt->mitsubaBackendMode == MitsubaBackendMode::Cpu) {
        mitsubaBackendIndex = 2;
    }
    SendMessageA(state.mitsubaBackendCombo, CB_SETCURSEL, mitsubaBackendIndex, 0);

    y += 38;
    makeLabel(advancedPage, "Quality", kMargin, y, kLabelWidth, kRowHeight);
    state.mitsubaQualityCombo = makeCombo(advancedPage, kIdMitsubaQuality, kControlX, y, kControlWidth);
    addComboItem(state.mitsubaQualityCombo, "Preview (64 px grid limit, 12 iterations)");
    addComboItem(state.mitsubaQualityCombo, "Standard (128 px grid limit, 24 iterations)");
    addComboItem(state.mitsubaQualityCombo, "High detail (256 px grid limit, 50 iterations)");
    // Ultra remains supported by the CLI, project format, and backend for
    // controlled experiments, but is intentionally not offered in the GUI.
    int mitsubaQualityIndex = 1;
    if (state.opt->mitsubaQualityMode == MitsubaQualityMode::Preview) {
        mitsubaQualityIndex = 0;
    } else if (state.opt->mitsubaQualityMode == MitsubaQualityMode::Research) {
        mitsubaQualityIndex = 2;
    }
    SendMessageA(state.mitsubaQualityCombo, CB_SETCURSEL, mitsubaQualityIndex, 0);

    y += 40;
    state.mitsubaPythonButton = makeControl(advancedPage, "BUTTON", "Locate Backend...", BS_PUSHBUTTON, kIdSelectMitsubaPython, kControlX, y, kButtonWidth, kRowHeight);
    state.mitsubaStatus = makeControl(advancedPage, "STATIC", "", SS_LEFT, kIdMitsubaStatus, kControlX + kButtonWidth + 12, y, 298, 42);

    HWND progressPage = makeTabPage(hwnd, state, "Progress");
    state.previewCaption = makeControl(progressPage, "STATIC", "Mitsuba previews appear every 5 iterations. Provisional geometry only.", SS_LEFT,
        kIdPreviewCaption, 16, 4, 640, 30);
    state.previewControl = makeControl(progressPage, "STATIC", "", SS_OWNERDRAW, kIdPreview, 16, 38, 640, 246);
    SetWindowSubclass(state.previewControl, previewSubclass, 1, reinterpret_cast<DWORD_PTR>(&state));
    state.runLog = makeControl(progressPage, "EDIT", "", ES_MULTILINE | ES_READONLY | WS_VSCROLL | ES_AUTOVSCROLL,
        kIdRunLog, 16, 296, 640, 106);

    state.statusText = makeControl(hwnd, "EDIT", "Ready. Start a project, or use File > Open Completed Project.",
        ES_MULTILINE | ES_READONLY | ES_AUTOVSCROLL | WS_VSCROLL, kIdRunStatus, kMargin, 546, 684, 88);
    state.progressBar = makeControl(hwnd, PROGRESS_CLASSA, "", 0, kIdProgressBar, kMargin, 642, 684, 18);
    SendMessageA(state.progressBar, PBM_SETRANGE, 0, MAKELPARAM(0, 100));
    ShowWindow(state.progressBar, SW_HIDE);
    state.startButton = makeControl(hwnd, "BUTTON", "Start", BS_DEFPUSHBUTTON, kIdStart, kWindowWidth - 290, 674, 110, 34);
    state.cancelButton = makeControl(hwnd, "BUTTON", "Cancel Run", BS_PUSHBUTTON, kIdCancel, kWindowWidth - 165, 674, 110, 34);
    EnableWindow(state.cancelButton, FALSE);

    TabCtrl_SetCurSel(state.tabControl, 0);
    showSelectedTabPage(state);
    updateSetupControls(state);
    state.initializing = false;
    layoutApplicationFooter(state);
}

void layoutApplicationFooter(SetupDialogState& state) {
    if (!state.startButton) return;
    RECT client = {};
    GetClientRect(state.hwnd, &client);
    const int width = client.right - client.left;
    const int startY = client.bottom - 49;
    const int progressY = startY - 18;
    MoveWindow(state.statusText, kMargin, 546, width - 2 * kMargin, std::max(24, progressY - 554), TRUE);
    MoveWindow(state.progressBar, kMargin, progressY, width - 2 * kMargin, 10, TRUE);
    MoveWindow(state.startButton, width - 250, startY, 110, 34, TRUE);
    MoveWindow(state.cancelButton, width - 125, startY, 110, 34, TRUE);
}

void setProgressBarMode(SetupDialogState& state, bool determinate) {
    const int requestedMode = determinate ? 1 : 0;
    if (state.progressBarMode == requestedMode) {
        return;
    }

    SendMessageA(state.progressBar, PBM_SETMARQUEE, FALSE, 0);
    const LONG_PTR style = GetWindowLongPtrA(state.progressBar, GWL_STYLE);
    const LONG_PTR requestedStyle = determinate ? style & ~PBS_MARQUEE : style | PBS_MARQUEE;
    if (requestedStyle != style) {
        SetWindowLongPtrA(state.progressBar, GWL_STYLE, requestedStyle);
        SetWindowPos(
            state.progressBar,
            nullptr,
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED);
    }
    if (!determinate) {
        SendMessageA(state.progressBar, PBM_SETMARQUEE, TRUE, 40);
    }
    state.progressBarMode = requestedMode;
}

void updateApplicationState(SetupDialogState& state) {
    const std::string projectName = state.opt->imagePaths.empty() ? "New Project" :
        fs::path(state.opt->imagePaths.front()).parent_path().filename().string();
    const std::string title = "what-a-relief " WHAT_A_RELIEF_VERSION " - " + projectName +
        (state.busy ? " (Processing)" : "");
    SetWindowTextA(state.hwnd, title.c_str());
    for (HWND page : state.tabPages) EnableWindow(page, !state.busy || page == GetParent(state.runLog));
    EnableWindow(state.tabControl, TRUE);
    EnableWindow(state.startButton, !state.busy && !state.viewing);
    EnableWindow(state.cancelButton, state.busy && !gProgressCancelRequested);
    const HMENU menu = GetMenu(state.hwnd);
    for (int id : {kIdNewProject, kIdOpenProject}) {
        EnableMenuItem(menu, id, MF_BYCOMMAND | (state.busy ? MF_GRAYED : MF_ENABLED));
    }
    EnableMenuItem(menu, kIdOpenResults, MF_BYCOMMAND |
        (!state.busy && !state.lastOutput.empty() ? MF_ENABLED : MF_GRAYED));
    EnableMenuItem(menu, kIdOpenReview, MF_BYCOMMAND |
        (!state.busy && !state.reviewPath.empty() ? MF_ENABLED : MF_GRAYED));
    DrawMenuBar(state.hwnd);
}

void resetProject(SetupDialogState& state, Options options) {
    options.guiMode = true;
    options.noGui = false;
    options.mitsubaPythonPath = state.opt->mitsubaPythonPath;
    options.mitsubaWorkerPath = state.opt->mitsubaWorkerPath;
    state.initializing = true;
    while (HWND child = GetWindow(state.hwnd, GW_CHILD)) DestroyWindow(child);
    for (HBRUSH brush : {state.requiredBrush, state.optionalBrush, state.sectionBrush}) {
        if (brush) DeleteObject(brush);
    }
    SetupDialogState fresh;
    fresh.hwnd = state.hwnd;
    fresh.opt = state.opt;
    fresh.processor = state.processor;
    *fresh.opt = std::move(options);
    state = std::move(fresh);
    gProgressCancelRequested = false;
    createSetupControls(state.hwnd, state);
    updateApplicationState(state);
}

void openProject(SetupDialogState& state) {
    std::vector<char> filename(32768, '\0');
    OPENFILENAMEA dialog = {};
    dialog.lStructSize = sizeof(dialog);
    dialog.hwndOwner = state.hwnd;
    dialog.lpstrFilter = "Completed project (run_manifest.json)\0run_manifest.json\0JSON files\0*.json\0";
    dialog.lpstrFile = filename.data();
    dialog.nMaxFile = static_cast<DWORD>(filename.size());
    dialog.lpstrTitle = "Open a completed project's run_manifest.json";
    dialog.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    if (!GetOpenFileNameA(&dialog)) return;
    try {
        LoadedProject project = loadCompletedProject(filename.data());
        std::string message = "Opened completed project: " + project.completedOutputDir +
            "\r\nReview the tabs, then Start. Previous outputs will be preserved.";
        for (const auto& warning : project.warnings) message += "\r\n" + warning;
        resetProject(state, std::move(project.options));
        state.lastOutput = project.completedOutputDir;
        const fs::path review = fs::path(state.lastOutput) / "inverse" / "unvalidated_candidate" / "review.html";
        if (fs::is_regular_file(review)) state.reviewPath = review.string();
        SetWindowTextA(state.statusText, message.c_str());
        updateApplicationState(state);
    } catch (const std::exception& e) {
        showOwnerMessage(state.hwnd, "Open Project", e.what(), MB_ICONERROR);
    }
}

void cancelRun(SetupDialogState& state, bool exitAfterRun) {
    state.exitAfterRun = state.exitAfterRun || exitAfterRun;
    gProgressCancelRequested = true;
    SetWindowTextA(state.statusText, exitAfterRun
        ? "Canceling at the next processing checkpoint, then exiting..."
        : "Canceling at the next processing checkpoint...");
    updateApplicationState(state);
}

std::string durationText(double seconds) {
    const auto value = static_cast<long long>(std::max(0.0, seconds));
    std::ostringstream text;
    if (value >= 3600) text << value / 3600 << "h ";
    text << (value / 60) % 60 << "m " << value % 60 << "s";
    return text.str();
}

void refreshRunProgress(SetupDialogState& state) {
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - state.started).count();
    std::deque<ProgressUpdate> events;
    { std::lock_guard<std::mutex> lock(state.channel->mutex); events.swap(state.channel->events); }
    for (const auto& event : events) {
        if (!event.preview.empty()) {
            cv::cvtColor(event.preview, state.preview, cv::COLOR_BGR2BGRA);
            const std::string caption = "Iteration " + std::to_string(event.previewIteration) + "/" + std::to_string(event.previewTotal) +
                " (not validated). RGB normals (left) / diffuse hillshade (right).";
            SetWindowTextA(state.previewCaption, caption.c_str());
            InvalidateRect(state.previewControl, nullptr, FALSE);
        }
        if (state.logLines.empty() || event.message != state.currentProgress.message) {
            state.logLines.push_back(durationText(elapsed) + "  " + event.message);
            while (state.logLines.size() > 200) state.logLines.pop_front();
        }
        state.timing.update(event, elapsed);
        state.currentProgress = event;
        state.currentProgress.preview.release();
    }
    if (!events.empty()) {
        std::string text;
        for (const auto& line : state.logLines) text += line + "\r\n";
        SetWindowTextA(state.runLog, text.c_str());
        SendMessageA(state.runLog, EM_SETSEL, text.size(), text.size());
        SendMessageA(state.runLog, EM_SCROLLCARET, 0, 0);
    }
    const auto& update = state.currentProgress;
    const bool measured = update.total > 0;
    setProgressBarMode(state, measured);
    if (measured) SendMessageA(state.progressBar, PBM_SETPOS, static_cast<int>(100 * std::clamp(update.completed / update.total, 0.0, 1.0)), 0);
    const double remaining = state.timing.remaining(elapsed);
    std::string text = update.message + "\r\nElapsed " + durationText(elapsed) + " | Stage ETA: " +
        (remaining >= 0 ? "about " + durationText(remaining) : "estimating...");
    if (!update.estimateScope.empty()) text += " (" + update.estimateScope + ")";
    if (gProgressCancelRequested) text = "Cancel requested; waiting for a safe checkpoint.\r\nElapsed " + durationText(elapsed);
    SetWindowTextA(state.statusText, text.c_str());
}

void startProjectWorker(SetupDialogState& state) {
    state.channel = std::make_shared<ProgressChannel>();
    state.started = std::chrono::steady_clock::now();
    TabCtrl_SetCurSel(state.tabControl, static_cast<int>(state.tabPages.size() - 1));
    showSelectedTabPage(state);
    state.timing = {};
    state.currentProgress = {"Starting photometric stereo...", 0, "Starting"};
    state.logLines.clear();
    state.preview.release();
    SetWindowTextA(state.runLog, "");
    SetWindowTextA(state.previewCaption, "Mitsuba previews appear every 5 iterations. Provisional geometry only.");
    InvalidateRect(state.previewControl, nullptr, TRUE);
    const auto channel = state.channel;
    Options options = *state.opt;
    options.heightMask = options.heightMask.clone();
    state.job = std::async(std::launch::async, [options = std::move(options), processor = state.processor, channel]() mutable {
        CompletedRun completed;
        try {
            completed.result = processor(options, [channel](const ProgressUpdate& update) {
                if (gProgressCancelRequested) throw std::runtime_error("Processing canceled by user.");
                std::lock_guard<std::mutex> lock(channel->mutex);
                channel->events.push_back(update);
                while (channel->events.size() > 256) channel->events.pop_front();
            });
        } catch (const std::exception& error) {
            completed.error = error.what();
        } catch (...) {
            completed.error = "Unexpected processing failure.";
        }
        completed.options = std::move(options);
        return completed;
    });
    SetTimer(state.hwnd, kRunTimer, 125, nullptr);
}

void finishProject(SetupDialogState& state) {
    CompletedRun job = state.job.get();
    *state.opt = std::move(job.options);
    GuiRunResult result = std::move(job.result);
    bool completed = false;
    try {
        if (!job.error.empty()) throw std::runtime_error(job.error);
        completed = true;
        state.lastOutput = state.opt->outputDir;
        state.reviewPath = result.inverse.candidateSaved ? result.inverse.candidateReviewPath : "";
        std::string message = "Complete. Outputs: " + state.lastOutput +
            "\r\nAdjust the tabs and Run Again, or use File > New Project / Open Completed Project.";
        if (!state.opt->microscopeCalibrationFile.empty()) {
            message += "\r\nMicroscope working-plane correction and calibrated pixel scale applied.";
        }
        const std::string shadowSummary = formatShadowRefinementSummary(result.shadow);
        if (!shadowSummary.empty()) {
            message += "\r\n" + shadowSummary;
        }
        const std::string lightGainSummary = formatLightGainSummary(result.lightGain);
        if (!lightGainSummary.empty()) message += "\r\n" + lightGainSummary;
        if (result.inverse.attempted && !result.inverse.accepted) {
            message += "\r\nInverse refinement not accepted: " + result.inverse.decision + ". Baseline retained.";
            if (result.inverse.candidateSaved) message += " Use File > Review Inverse Candidate to inspect the unvalidated result.";
        }
        SetWindowTextA(state.statusText, message.c_str());
        SetWindowTextA(state.startButton, "Run Again");
    } catch (const std::exception& e) {
        const std::string message = gProgressCancelRequested ? "Run canceled. Settings retained; ready to try again." :
            std::string("Processing failed: ") + e.what() + "\r\nSettings retained. Correct the problem and Start again, or use File > New Project.";
        SetWindowTextA(state.statusText, message.c_str());
    }
    state.busy = false;
    KillTimer(state.hwnd, kRunTimer);
    SendMessageA(state.progressBar, PBM_SETMARQUEE, FALSE, 0);
    state.progressBarMode = -1;
    ShowWindow(state.progressBar, SW_HIDE);
    updateSetupControls(state);
    updateApplicationState(state);
    showSelectedTabPage(state);
    if (state.exitAfterRun) {
        PostMessageA(state.hwnd, WM_CLOSE, 0, 0);
        return;
    }
    if (completed && !result.relightNormals.empty()) {
        state.viewing = true;
        updateApplicationState(state);
        try {
            launchRelightViewer(result.relightNormals, result.relightMask, state.lastOutput,
                [&]() { return state.pendingAction != 0; });
        } catch (const std::exception& e) {
            SetWindowTextA(state.statusText, ("Outputs are complete. Relighting could not open: " + std::string(e.what())).c_str());
        }
        state.viewing = false;
        updateApplicationState(state);
        if (state.pendingAction) {
            const int action = state.pendingAction;
            state.pendingAction = 0;
            PostMessageA(state.hwnd, WM_COMMAND, action, 0);
        }
    }
}

LRESULT CALLBACK setupWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (msg == WM_NCCREATE) {
        const CREATESTRUCTA* create = reinterpret_cast<const CREATESTRUCTA*>(lParam);
        SetWindowLongPtrA(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(create->lpCreateParams));
    }

    SetupDialogState* state = reinterpret_cast<SetupDialogState*>(GetWindowLongPtrA(hwnd, GWLP_USERDATA));
    switch (msg) {
    case WM_COMMAND:
        if (state == nullptr || state->initializing) {
            break;
        }
        if (state->selecting) {
            if (LOWORD(wParam) == kIdExit) state->pendingAction = kIdExit;
            return 0;
        }
        if (state->busy && LOWORD(wParam) != kIdCancel && LOWORD(wParam) != kIdExit) return 0;
        if (state->viewing && (LOWORD(wParam) == kIdNewProject || LOWORD(wParam) == kIdOpenProject || LOWORD(wParam) == kIdExit)) {
            state->pendingAction = LOWORD(wParam);
            return 0;
        }
        switch (LOWORD(wParam)) {
        case kIdNewProject:
            resetProject(*state, Options{});
            return 0;
        case kIdOpenProject:
            openProject(*state);
            return 0;
        case kIdExit:
            SendMessageA(hwnd, WM_CLOSE, 0, 0);
            return 0;
        case kIdOpenResults:
            if (!state->lastOutput.empty()) openGuiReviewFile(state->lastOutput);
            return 0;
        case kIdOpenReview:
            if (!state->reviewPath.empty()) openGuiReviewFile(state->reviewPath);
            return 0;
        case kIdSelectImages:
            selectImages(*state);
            return 0;
        case kIdSelectOutput:
            selectOutput(*state);
            return 0;
        case kIdSelectLights:
            selectLightsFile(*state);
            return 0;
        case kIdClearLights:
            clearLightsFile(*state);
            return 0;
        case kIdMakeCalibrationTarget:
            makeCalibrationTarget(*state);
            return 0;
        case kIdCreateMicroscopeCalibration:
            createMicroscopeCalibrationInteractive(*state);
            return 0;
        case kIdLoadMicroscopeCalibration:
            loadMicroscopeCalibrationInteractive(*state);
            return 0;
        case kIdClearMicroscopeCalibration:
            clearMicroscopeCalibration(*state);
            return 0;
        case kIdEstimateLightGains:
            state->opt->estimateLightGains = buttonChecked(state->estimateLightGainsCheck);
            updateSetupControls(*state);
            return 0;
        case kIdLighting:
            if (HIWORD(wParam) == CBN_SELCHANGE) {
                updateSetupControls(*state);
            }
            return 0;
        case kIdSolver:
            if (HIWORD(wParam) == CBN_SELCHANGE) {
                updateSetupControls(*state);
            }
            return 0;
        case kIdSrgb:
            if (HIWORD(wParam) == CBN_SELCHANGE) refreshInputResponse(*state);
            return 0;
        case kIdNearField:
            updateSetupControls(*state);
            return 0;
        case kIdPixelScale:
            if (HIWORD(wParam) == EN_CHANGE) {
                try {
                    state->opt->pixelScaleMm = editDouble(state->pixelScaleEdit, "pixel scale");
                } catch (const std::exception&) {
                    // Let Start validation report incomplete manual edits.
                }
                if (state->heightMaskStatus != nullptr) {
                    updateSetupControls(*state);
                }
            }
            return 0;
        case kIdHeight:
            if (!buttonChecked(state->heightCheck)) {
                setButtonChecked(state->meshCheck, false);
                setButtonChecked(state->printableMeshCheck, false);
                setButtonChecked(state->printableFillHolesCheck, false);
            }
            updateSetupControls(*state);
            return 0;
        case kIdMesh:
            if (buttonChecked(state->meshCheck)) {
                setButtonChecked(state->heightCheck, true);
            }
            updateSetupControls(*state);
            return 0;
        case kIdPrintableMesh:
            if (buttonChecked(state->printableMeshCheck)) {
                setButtonChecked(state->heightCheck, true);
            } else {
                setButtonChecked(state->printableFillHolesCheck, false);
            }
            updateSetupControls(*state);
            return 0;
        case kIdPrintableFillHoles:
            updateSetupControls(*state);
            return 0;
        case kIdShadowHeightRefinement:
            if (buttonChecked(state->shadowHeightRefinementCheck)) {
                setButtonChecked(state->mitsubaInverseCheck, false);
            }
            updateSetupControls(*state);
            return 0;
        case kIdMitsubaInverse:
            if (buttonChecked(state->mitsubaInverseCheck)) {
                setButtonChecked(state->shadowHeightRefinementCheck, false);
            }
            updateSetupControls(*state);
            return 0;
        case kIdMitsubaBackend:
        case kIdMitsubaQuality:
            if (HIWORD(wParam) == CBN_SELCHANGE) {
                updateSetupControls(*state);
            }
            return 0;
        case kIdShadowLedDiameter:
            if (HIWORD(wParam) == EN_CHANGE && state->heightMaskStatus != nullptr && state->mitsubaStatus != nullptr) {
                updateSetupControls(*state);
            }
            return 0;
        case kIdSelectMitsubaPython:
            selectMitsubaPython(*state);
            return 0;
        case kIdRtiExport:
            updateSetupControls(*state);
            return 0;
        case kIdMarkSphere:
            markSphere(*state);
            return 0;
        case kIdCrop:
            cropSurface(*state);
            return 0;
        case kIdHeightMask:
            markHeightMask(*state);
            return 0;
        case kIdClearHeightMask:
            clearHeightMask(*state);
            return 0;
        case kIdMarkScale:
            markScaleLine(*state);
            return 0;
        case kIdStart:
            if (state->viewing) return 0;
            try {
                if (validateAndAccept(*state)) {
                    const std::string fresh = freshProjectOutputDirectory(state->opt->outputDir);
                    if (fresh != state->opt->outputDir) {
                        state->opt->outputDir = fresh;
                        if (!state->opt->meshPath.empty()) state->opt->meshPath = (fs::path(fresh) / "surface.ply").string();
                        if (!state->opt->printableMeshPath.empty()) state->opt->printableMeshPath = (fs::path(fresh) / "printable_surface.ply").string();
                        if (state->opt->exportRti) state->opt->rtiPath = (fs::path(fresh) / "rti").string();
                    }
                    state->busy = true;
                    state->exitAfterRun = false;
                    gProgressCancelRequested = false;
                    updateSetupControls(*state);
                    updateApplicationState(*state);
                    SetWindowTextA(state->statusText, "Starting photometric stereo...");
                    setProgressBarMode(*state, false);
                    ShowWindow(state->progressBar, SW_SHOW);
                    PostMessageA(hwnd, kRunProjectMessage, 0, 0);
                }
            } catch (const std::exception& e) {
                showOwnerMessage(hwnd, "Project Setup", e.what(), MB_ICONWARNING);
            }
            return 0;
        case kIdCancel:
            if (state->busy) cancelRun(*state, false);
            return 0;
        default:
            break;
        }
        break;
    case kRunProjectMessage:
        if (state && state->busy) {
            try { startProjectWorker(*state); }
            catch (const std::exception& error) {
                state->busy = false;
                SendMessageA(state->progressBar, PBM_SETMARQUEE, FALSE, 0);
                state->progressBarMode = -1;
                ShowWindow(state->progressBar, SW_HIDE);
                SetWindowTextA(state->statusText, ("Could not start processing: " + std::string(error.what())).c_str());
                updateApplicationState(*state);
            }
        }
        return 0;
    case WM_TIMER:
        if (state && wParam == kRunTimer && state->busy) {
            refreshRunProgress(*state);
            if (state->job.valid() && state->job.wait_for(std::chrono::seconds(0)) == std::future_status::ready) finishProject(*state);
        }
        return 0;
    case WM_NOTIFY:
        if (state != nullptr) {
            const NMHDR* notify = reinterpret_cast<const NMHDR*>(lParam);
            if (notify != nullptr && notify->idFrom == kIdTabs && notify->code == TCN_SELCHANGE) {
                showSelectedTabPage(*state);
                return 0;
            }
        }
        break;
    case WM_VSCROLL:
        if (state != nullptr) {
            SCROLLINFO info = {};
            info.cbSize = sizeof(info);
            info.fMask = SIF_ALL;
            GetScrollInfo(hwnd, SB_VERT, &info);

            int targetY = state->scrollY;
            switch (LOWORD(wParam)) {
            case SB_LINEUP:
                targetY -= kRowHeight;
                break;
            case SB_LINEDOWN:
                targetY += kRowHeight;
                break;
            case SB_PAGEUP:
                targetY -= static_cast<int>(info.nPage);
                break;
            case SB_PAGEDOWN:
                targetY += static_cast<int>(info.nPage);
                break;
            case SB_THUMBTRACK:
            case SB_THUMBPOSITION:
                targetY = info.nTrackPos;
                break;
            case SB_TOP:
                targetY = 0;
                break;
            case SB_BOTTOM:
                targetY = state->contentHeight;
                break;
            default:
                break;
            }
            scrollSetupWindow(*state, targetY);
            return 0;
        }
        break;
    case WM_MOUSEWHEEL:
        if (state != nullptr) {
            const int notches = GET_WHEEL_DELTA_WPARAM(wParam) / WHEEL_DELTA;
            scrollSetupWindow(*state, state->scrollY - notches * 3 * kRowHeight);
            return 0;
        }
        break;
    case WM_SIZE:
        if (state != nullptr) {
            layoutApplicationFooter(*state);
            updateSetupScrollInfo(*state);
            scrollSetupWindow(*state, state->scrollY);
            return 0;
        }
        break;
    case WM_GETMINMAXINFO: {
        auto* limits = reinterpret_cast<MINMAXINFO*>(lParam);
        limits->ptMinTrackSize = {kWindowWidth, 724};
        return 0;
    }
    case WM_CTLCOLORSTATIC:
        if (state != nullptr) {
            HDC dc = reinterpret_cast<HDC>(wParam);
            HWND control = reinterpret_cast<HWND>(lParam);
            if (control == state->nextStepLabel) {
                const COLORREF color = state->nextStepRequired ? RGB(255, 238, 205) : RGB(232, 244, 255);
                SetBkColor(dc, color);
                SetTextColor(dc, RGB(38, 38, 38));
                return reinterpret_cast<LRESULT>(state->nextStepRequired ? state->requiredBrush : state->optionalBrush);
            }
            if (isSectionHeader(*state, control)) {
                SetBkColor(dc, RGB(232, 236, 242));
                SetTextColor(dc, RGB(30, 50, 70));
                return reinterpret_cast<LRESULT>(state->sectionBrush);
            }
        }
        break;
    case WM_CLOSE:
        if (state != nullptr) {
            if (state->selecting) {
                state->pendingAction = kIdExit;
                return 0;
            }
            if (state->busy) {
                cancelRun(*state, true);
                return 0;
            }
            if (state->viewing) {
                state->pendingAction = kIdExit;
                return 0;
            }
            state->running = false;
        }
        DestroyWindow(hwnd);
        return 0;
    case WM_DESTROY:
        if (state != nullptr) {
            if (state->requiredBrush != nullptr) {
                DeleteObject(state->requiredBrush);
                state->requiredBrush = nullptr;
            }
            if (state->optionalBrush != nullptr) {
                DeleteObject(state->optionalBrush);
                state->optionalBrush = nullptr;
            }
            if (state->sectionBrush != nullptr) {
                DeleteObject(state->sectionBrush);
                state->sectionBrush = nullptr;
            }
            state->running = false;
        }
        return 0;
    default:
        break;
    }
    return DefWindowProcA(hwnd, msg, wParam, lParam);
}

#else

#endif

} // namespace

void showGuiInfo(const std::string& title, const std::string& text) {
#ifdef _WIN32
    MessageBoxA(nullptr, text.c_str(), title.c_str(), MB_OK | MB_ICONINFORMATION);
#else
    (void)title;
    (void)text;
#endif
}

bool askGuiYesNo(const std::string& title, const std::string& text, bool defaultYes) {
#ifdef _WIN32
    const UINT defaultButton = defaultYes ? MB_DEFBUTTON1 : MB_DEFBUTTON2;
    return MessageBoxA(nullptr, text.c_str(), title.c_str(), MB_YESNO | MB_ICONQUESTION | defaultButton) == IDYES;
#else
    (void)title;
    (void)text;
    return defaultYes;
#endif
}

void openGuiReviewFile(const std::string& path) {
#ifdef _WIN32
    const auto result = reinterpret_cast<INT_PTR>(ShellExecuteW(
        nullptr, L"open", fs::path(path).wstring().c_str(), nullptr, nullptr, SW_SHOWNORMAL));
    if (result <= 32) {
        showGuiInfo("what-a-relief Review", "Could not open the review. It is available at:\n\n" + path);
    }
#else
    (void)path;
#endif
}

bool guiProgressCancellationRequested() {
#ifdef _WIN32
    return gProgressCancelRequested.load();
#else
    return false;
#endif
}

void launchGuiApplication(Options& opt, const GuiProcessor& processor, bool visible) {
#ifdef _WIN32
    const HINSTANCE instance = GetModuleHandleA(nullptr);
    INITCOMMONCONTROLSEX controls = {};
    controls.dwSize = sizeof(controls);
    controls.dwICC = ICC_TAB_CLASSES | ICC_PROGRESS_CLASS;
    InitCommonControlsEx(&controls);

    const char* className = "WhatAReliefSetupWindow";
    WNDCLASSA wc = {};
    wc.lpfnWndProc = setupWndProc;
    wc.hInstance = instance;
    wc.lpszClassName = className;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);
    RegisterClassA(&wc);

    SetupDialogState state;
    state.opt = &opt;
    state.processor = processor;
    gProgressCancelRequested = false;
    HMENU menu = CreateMenu();
    HMENU file = CreatePopupMenu();
    AppendMenuA(file, MF_STRING, kIdNewProject, "&New Project\tCtrl+N");
    AppendMenuA(file, MF_STRING, kIdOpenProject, "&Open Completed Project...\tCtrl+O");
    AppendMenuA(file, MF_SEPARATOR, 0, nullptr);
    AppendMenuA(file, MF_STRING, kIdOpenResults, "Open &Output Folder");
    AppendMenuA(file, MF_STRING, kIdOpenReview, "&Review Inverse Candidate");
    AppendMenuA(file, MF_SEPARATOR, 0, nullptr);
    AppendMenuA(file, MF_STRING, kIdExit, "E&xit\tAlt+F4");
    AppendMenuA(menu, MF_POPUP, reinterpret_cast<UINT_PTR>(file), "&File");
    const std::string windowTitle = "what-a-relief " WHAT_A_RELIEF_VERSION;
    HWND hwnd = CreateWindowExA(
        WS_EX_DLGMODALFRAME,
        className,
        windowTitle.c_str(),
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX | WS_THICKFRAME | WS_CLIPCHILDREN,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        kWindowWidth,
        kWindowHeight,
        nullptr,
        menu,
        instance,
        &state);
    if (hwnd == nullptr) {
        DestroyMenu(menu);
        throw std::runtime_error("Could not create setup window.");
    }

    state.hwnd = hwnd;
    createSetupControls(hwnd, state);
    updateApplicationState(state);
    MONITORINFO monitor = {};
    monitor.cbSize = sizeof(monitor);
    if (GetMonitorInfoA(MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST), &monitor)) {
        RECT bounds = {};
        GetWindowRect(hwnd, &bounds);
        const int height = std::min(kWindowHeight, static_cast<int>(monitor.rcWork.bottom - monitor.rcWork.top));
        const int x = std::max(static_cast<int>(monitor.rcWork.left), std::min(static_cast<int>(bounds.left), static_cast<int>(monitor.rcWork.right) - kWindowWidth));
        const int y = std::max(static_cast<int>(monitor.rcWork.top), std::min(static_cast<int>(bounds.top), static_cast<int>(monitor.rcWork.bottom) - height));
        SetWindowPos(hwnd, nullptr, x, y, kWindowWidth, height, SWP_NOZORDER | SWP_NOACTIVATE);
    }
    ShowWindow(hwnd, visible ? SW_SHOW : SW_HIDE);
    UpdateWindow(hwnd);

    ACCEL shortcuts[] = {{FVIRTKEY | FCONTROL, 'N', kIdNewProject}, {FVIRTKEY | FCONTROL, 'O', kIdOpenProject}};
    HACCEL accelerators = CreateAcceleratorTableA(shortcuts, 2);
    MSG msg = {};
    while (state.running && GetMessageA(&msg, nullptr, 0, 0) > 0) {
        if (!TranslateAcceleratorA(hwnd, accelerators, &msg) && !IsDialogMessageA(hwnd, &msg)) {
            TranslateMessage(&msg);
            DispatchMessageA(&msg);
        }
    }

    DestroyAcceleratorTable(accelerators);
    if (IsWindow(hwnd)) DestroyWindow(hwnd);
#else
    (void)opt;
    (void)processor;
    (void)visible;
    throw std::runtime_error("GUI image loading is only implemented on Windows. Use --image arguments on this platform.");
#endif
}
