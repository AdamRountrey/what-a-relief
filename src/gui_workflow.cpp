#include "gui_workflow.hpp"

#include "crop_ui.hpp"
#include "image_io.hpp"
#include "photometric.hpp"
#include "sphere_ui.hpp"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <commdlg.h>
#include <shlobj.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr size_t kMinImages = 3;
constexpr size_t kMaxImages = 25;

#ifdef _WIN32

constexpr int kWindowWidth = 700;
constexpr int kWindowHeight = 820;
constexpr int kMargin = 20;
constexpr int kLabelWidth = 150;
constexpr int kControlX = 180;
constexpr int kControlWidth = 460;
constexpr int kButtonWidth = 150;
constexpr int kRowHeight = 30;

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

struct SetupDialogState {
    Options* opt = nullptr;
    bool running = true;
    bool accepted = false;
    HWND hwnd = nullptr;
    HWND imageStatus = nullptr;
    HWND outputStatus = nullptr;
    HWND lightsStatus = nullptr;
    HWND lightingCombo = nullptr;
    HWND sphereButton = nullptr;
    HWND sphereStatus = nullptr;
    HWND cropButton = nullptr;
    HWND cropStatus = nullptr;
    HWND solverCombo = nullptr;
    HWND flattenCombo = nullptr;
    HWND nearFieldCheck = nullptr;
    HWND ringRadiusEdit = nullptr;
    HWND ringHeightEdit = nullptr;
    HWND pixelScaleEdit = nullptr;
    HWND srgbCheck = nullptr;
    HWND heightCheck = nullptr;
    HWND meshCheck = nullptr;
    HWND relightCheck = nullptr;
    HWND specularDiagnosticsCheck = nullptr;
};

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
    std::vector<char> buffer(65536, '\0');
    OPENFILENAMEA ofn = {};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = owner;
    ofn.lpstrFilter =
        "Image files\0*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp\0"
        "All files\0*.*\0";
    ofn.lpstrFile = buffer.data();
    ofn.nMaxFile = static_cast<DWORD>(buffer.size());
    ofn.lpstrTitle = "Select 3 to 25 photometric stereo images";
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
    info.lpszTitle = "Choose output folder for What A Relief results";
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

std::string lightsStatusText(const std::string& path) {
    if (path.empty()) {
        return "No previous calibration selected";
    }
    return baseName(path) + " loaded; sphere not needed";
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
        WS_CHILD | WS_VISIBLE | style,
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

int comboSelection(HWND combo) {
    return static_cast<int>(SendMessageA(combo, CB_GETCURSEL, 0, 0));
}

void showOwnerMessage(HWND owner, const std::string& title, const std::string& text, UINT icon) {
    MessageBoxA(owner, text.c_str(), title.c_str(), MB_OK | icon);
}

void updateSetupControls(SetupDialogState& state) {
    Options& opt = *state.opt;
    SetWindowTextA(state.imageStatus, imageStatusText(opt.imagePaths).c_str());
    SetWindowTextA(state.outputStatus, outputStatusText(opt.outputDir).c_str());
    SetWindowTextA(state.lightsStatus, lightsStatusText(opt.lightsFile).c_str());
    SetWindowTextA(state.sphereStatus, opt.hasSphere ? "Sphere marked" : "No sphere marked");
    SetWindowTextA(state.cropStatus, opt.hasCrop ? "Crop selected" : "No crop selected");

    const bool calibrated = comboSelection(state.lightingCombo) == 0;
    const bool usingLightsFile = !opt.lightsFile.empty();
    EnableWindow(state.lightingCombo, !usingLightsFile);
    EnableWindow(state.sphereButton, calibrated && !usingLightsFile);
    EnableWindow(state.cropButton, !calibrated);
    EnableWindow(state.solverCombo, calibrated);
    EnableWindow(state.nearFieldCheck, calibrated);
    const bool nearField = calibrated && buttonChecked(state.nearFieldCheck);
    EnableWindow(state.ringRadiusEdit, nearField);
    EnableWindow(state.ringHeightEdit, nearField);
    EnableWindow(state.pixelScaleEdit, nearField);
}

void selectImages(SetupDialogState& state) {
    try {
        state.opt->imagePaths = chooseImageFiles(state.hwnd);
        const double tagScale = state.opt->imagePaths.empty() ? 0.0 : readPixelScaleMmFromImage(state.opt->imagePaths.front());
        if (tagScale > 0.0 && state.pixelScaleEdit != nullptr) {
            state.opt->pixelScaleMm = tagScale;
            setEditDouble(state.pixelScaleEdit, tagScale);
        }
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
        state.opt->lightsFile = chooseLightsFile(state.hwnd);
        state.opt->hasSphere = false;
        loadLightsFileMetadata(state.opt->lightsFile, *state.opt);
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
        state.opt->sphere = chooseSphereInteractive(loadDisplayImage(state.opt->imagePaths.front()));
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
        state.opt->crop = chooseCropInteractive(loadDisplayImage(state.opt->imagePaths.front()));
        state.opt->hasCrop = true;
        updateSetupControls(state);
    } catch (const std::exception& e) {
        showOwnerMessage(state.hwnd, "Crop Surface", e.what(), MB_ICONWARNING);
    }
}

bool validateAndAccept(SetupDialogState& state) {
    Options& opt = *state.opt;
    if (opt.imagePaths.size() < kMinImages || opt.imagePaths.size() > kMaxImages) {
        showOwnerMessage(
            state.hwnd,
            "Setup",
            "Select 3 to 25 images. You selected " + std::to_string(opt.imagePaths.size()) + ".",
            MB_ICONWARNING);
        return false;
    }
    if (opt.outputDir.empty()) {
        showOwnerMessage(state.hwnd, "Setup", "Choose an output folder.", MB_ICONWARNING);
        return false;
    }

    opt.uncalibratedLighting = comboSelection(state.lightingCombo) == 1;
    if (opt.uncalibratedLighting) {
        opt.lightsFile.clear();
    }
    if (opt.uncalibratedLighting && opt.imagePaths.size() < 4) {
        showOwnerMessage(state.hwnd, "Setup", "Uncalibrated no-sphere mode requires at least 4 images.", MB_ICONWARNING);
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
    if (!opt.uncalibratedLighting && buttonChecked(state.nearFieldCheck)) {
        opt.lightingModel = LightingModel::NearFieldRing;
        try {
            opt.ringLightRadiusMm = editDouble(state.ringRadiusEdit, "ring light radius");
            opt.ringLightHeightMm = editDouble(state.ringHeightEdit, "ring light height");
            opt.pixelScaleMm = editDouble(state.pixelScaleEdit, "pixel scale");
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
    opt.srgb = buttonChecked(state.srgbCheck);
    opt.calculateHeight = buttonChecked(state.heightCheck);
    if (buttonChecked(state.meshCheck)) {
        opt.calculateHeight = true;
        opt.meshPath = (fs::path(opt.outputDir) / "surface.ply").string();
    } else {
        opt.meshPath.clear();
    }
    opt.openRelightViewer = buttonChecked(state.relightCheck);
    opt.specularDiagnostics = buttonChecked(state.specularDiagnosticsCheck);
    return true;
}

void createSetupControls(HWND hwnd, SetupDialogState& state) {
    int y = kMargin;
    makeLabel(hwnd, "Images", kMargin, y, kLabelWidth, kRowHeight);
    makeControl(hwnd, "BUTTON", "Select Images...", BS_PUSHBUTTON, kIdSelectImages, kControlX, y, kButtonWidth, kRowHeight);
    state.imageStatus = makeControl(hwnd, "STATIC", "", SS_LEFT, kIdImageStatus, kControlX + kButtonWidth + 12, y + 5, 300, kRowHeight);

    y += 45;
    makeLabel(hwnd, "Output Folder", kMargin, y, kLabelWidth, kRowHeight);
    makeControl(hwnd, "BUTTON", "Choose Folder...", BS_PUSHBUTTON, kIdSelectOutput, kControlX, y, kButtonWidth, kRowHeight);
    state.outputStatus = makeControl(hwnd, "STATIC", "", SS_LEFT, kIdOutputStatus, kControlX + kButtonWidth + 12, y + 5, 300, kRowHeight);

    y += 55;
    makeLabel(hwnd, "Previous Calib.", kMargin, y, kLabelWidth, kRowHeight);
    makeControl(hwnd, "BUTTON", "Load CSV...", BS_PUSHBUTTON, kIdSelectLights, kControlX, y, kButtonWidth, kRowHeight);
    makeControl(hwnd, "BUTTON", "Clear", BS_PUSHBUTTON, kIdClearLights, kControlX + kButtonWidth + 12, y, 70, kRowHeight);
    state.lightsStatus = makeControl(hwnd, "STATIC", "", SS_LEFT, kIdLightsStatus, kControlX + kButtonWidth + 94, y + 5, 220, kRowHeight);

    y += 45;
    makeLabel(hwnd, "Lighting", kMargin, y, kLabelWidth, kRowHeight);
    state.lightingCombo = makeControl(hwnd, "COMBOBOX", "", CBS_DROPDOWNLIST | WS_TABSTOP, kIdLighting, kControlX, y, kControlWidth, 120);
    addComboItem(state.lightingCombo, "Calibrated: loaded CSV or mark sphere");
    addComboItem(state.lightingCombo, "No sphere: estimate unknown lighting (relative)");
    SendMessageA(state.lightingCombo, CB_SETCURSEL, state.opt->uncalibratedLighting ? 1 : 0, 0);

    y += 45;
    makeLabel(hwnd, "Near Field", kMargin, y, kLabelWidth, kRowHeight);
    state.nearFieldCheck = makeControl(hwnd, "BUTTON", "Use ring light point-source model", BS_AUTOCHECKBOX, kIdNearField, kControlX, y, kControlWidth, 24);
    setButtonChecked(state.nearFieldCheck, state.opt->lightingModel == LightingModel::NearFieldRing);

    y += 30;
    makeLabel(hwnd, "Ring Radius", kMargin, y, kLabelWidth, kRowHeight);
    state.ringRadiusEdit = makeControl(hwnd, "EDIT", "", ES_LEFT | WS_BORDER | WS_TABSTOP, kIdRingRadius, kControlX, y, 120, kRowHeight);
    setEditDouble(state.ringRadiusEdit, state.opt->ringLightRadiusMm);
    makeControl(hwnd, "STATIC", "mm from image/crop center", SS_LEFT, 0, kControlX + 135, y + 5, 260, kRowHeight);

    y += 30;
    makeLabel(hwnd, "Light Height", kMargin, y, kLabelWidth, kRowHeight);
    state.ringHeightEdit = makeControl(hwnd, "EDIT", "", ES_LEFT | WS_BORDER | WS_TABSTOP, kIdRingHeight, kControlX, y, 120, kRowHeight);
    setEditDouble(state.ringHeightEdit, state.opt->ringLightHeightMm);
    makeControl(hwnd, "STATIC", "mm above sample plane", SS_LEFT, 0, kControlX + 135, y + 5, 260, kRowHeight);

    y += 30;
    makeLabel(hwnd, "Pixel Scale", kMargin, y, kLabelWidth, kRowHeight);
    state.pixelScaleEdit = makeControl(hwnd, "EDIT", "", ES_LEFT | WS_BORDER | WS_TABSTOP, kIdPixelScale, kControlX, y, 120, kRowHeight);
    setEditDouble(state.pixelScaleEdit, state.opt->pixelScaleMm);
    makeControl(hwnd, "STATIC", "mm/pixel; 0 reads TIFF tags", SS_LEFT, 0, kControlX + 135, y + 5, 280, kRowHeight);

    y += 45;
    makeLabel(hwnd, "Sphere", kMargin, y, kLabelWidth, kRowHeight);
    state.sphereButton = makeControl(hwnd, "BUTTON", "Mark Sphere...", BS_PUSHBUTTON, kIdMarkSphere, kControlX, y, kButtonWidth, kRowHeight);
    state.sphereStatus = makeControl(hwnd, "STATIC", "", SS_LEFT, kIdSphereStatus, kControlX + kButtonWidth + 12, y + 5, 300, kRowHeight);

    y += 45;
    makeLabel(hwnd, "Crop", kMargin, y, kLabelWidth, kRowHeight);
    state.cropButton = makeControl(hwnd, "BUTTON", "Crop Surface...", BS_PUSHBUTTON, kIdCrop, kControlX, y, kButtonWidth, kRowHeight);
    state.cropStatus = makeControl(hwnd, "STATIC", "", SS_LEFT, kIdCropStatus, kControlX + kButtonWidth + 12, y + 5, 300, kRowHeight);

    y += 55;
    makeLabel(hwnd, "Normal Solver", kMargin, y, kLabelWidth, kRowHeight);
    state.solverCombo = makeControl(hwnd, "COMBOBOX", "", CBS_DROPDOWNLIST | WS_TABSTOP, kIdSolver, kControlX, y, kControlWidth, 120);
    addComboItem(state.solverCombo, "Robust: reject shadows/highlights and downweight outliers");
    addComboItem(state.solverCombo, "Standard least squares");
    SendMessageA(state.solverCombo, CB_SETCURSEL, state.opt->solverMode == NormalSolverMode::Standard ? 1 : 0, 0);

    y += 45;
    makeLabel(hwnd, "Relief Flattening", kMargin, y, kLabelWidth, kRowHeight);
    state.flattenCombo = makeControl(hwnd, "COMBOBOX", "", CBS_DROPDOWNLIST | WS_TABSTOP, kIdFlatten, kControlX, y, kControlWidth, 120);
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

    y += 55;
    state.srgbCheck = makeControl(hwnd, "BUTTON", "Treat JPEG/PNG input as sRGB", BS_AUTOCHECKBOX, kIdSrgb, kControlX, y, kControlWidth, 24);
    setButtonChecked(state.srgbCheck, state.opt->srgb);
    y += 30;
    state.heightCheck = makeControl(hwnd, "BUTTON", "Calculate height preview", BS_AUTOCHECKBOX, kIdHeight, kControlX, y, kControlWidth, 24);
    setButtonChecked(state.heightCheck, state.opt->calculateHeight);
    y += 30;
    state.meshCheck = makeControl(hwnd, "BUTTON", "Export PLY mesh from height preview", BS_AUTOCHECKBOX, kIdMesh, kControlX, y, kControlWidth, 24);
    setButtonChecked(state.meshCheck, !state.opt->meshPath.empty());
    y += 30;
    state.specularDiagnosticsCheck = makeControl(hwnd, "BUTTON", "Write experimental specular-cue diagnostics", BS_AUTOCHECKBOX, kIdSpecularDiagnostics, kControlX, y, kControlWidth, 24);
    setButtonChecked(state.specularDiagnosticsCheck, state.opt->specularDiagnostics);
    y += 30;
    state.relightCheck = makeControl(hwnd, "BUTTON", "Open interactive relight viewer after processing", BS_AUTOCHECKBOX, kIdRelight, kControlX, y, kControlWidth, 24);
    setButtonChecked(state.relightCheck, state.opt->openRelightViewer);

    y += 50;
    makeControl(hwnd, "BUTTON", "Start", BS_DEFPUSHBUTTON, kIdStart, kControlX + 210, y, 110, 34);
    makeControl(hwnd, "BUTTON", "Cancel", BS_PUSHBUTTON, kIdCancel, kControlX + 335, y, 110, 34);

    updateSetupControls(state);
}

LRESULT CALLBACK setupWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (msg == WM_NCCREATE) {
        const CREATESTRUCTA* create = reinterpret_cast<const CREATESTRUCTA*>(lParam);
        SetWindowLongPtrA(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(create->lpCreateParams));
    }

    SetupDialogState* state = reinterpret_cast<SetupDialogState*>(GetWindowLongPtrA(hwnd, GWLP_USERDATA));
    switch (msg) {
    case WM_COMMAND:
        if (state == nullptr) {
            break;
        }
        switch (LOWORD(wParam)) {
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
        case kIdLighting:
            if (HIWORD(wParam) == CBN_SELCHANGE) {
                updateSetupControls(*state);
            }
            return 0;
        case kIdNearField:
            updateSetupControls(*state);
            return 0;
        case kIdMarkSphere:
            markSphere(*state);
            return 0;
        case kIdCrop:
            cropSurface(*state);
            return 0;
        case kIdStart:
            if (validateAndAccept(*state)) {
                state->accepted = true;
                state->running = false;
                DestroyWindow(hwnd);
            }
            return 0;
        case kIdCancel:
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

void launchGuiWorkflow(Options& opt) {
#ifdef _WIN32
    const HINSTANCE instance = GetModuleHandleA(nullptr);
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
    HWND hwnd = CreateWindowExA(
        WS_EX_DLGMODALFRAME,
        className,
        "What A Relief Setup",
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        kWindowWidth,
        kWindowHeight,
        nullptr,
        nullptr,
        instance,
        &state);
    if (hwnd == nullptr) {
        throw std::runtime_error("Could not create setup window.");
    }

    state.hwnd = hwnd;
    createSetupControls(hwnd, state);
    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);

    MSG msg = {};
    while (state.running && GetMessageA(&msg, nullptr, 0, 0) > 0) {
        if (!IsDialogMessageA(hwnd, &msg)) {
            TranslateMessage(&msg);
            DispatchMessageA(&msg);
        }
    }

    if (!state.accepted) {
        throw std::runtime_error("GUI setup was canceled.");
    }
#else
    (void)opt;
    throw std::runtime_error("GUI image loading is only implemented on Windows. Use --image arguments on this platform.");
#endif
}
