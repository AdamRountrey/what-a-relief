#include "gui_workflow.hpp"

#include "crop_ui.hpp"
#include "image_io.hpp"
#include "mask_ui.hpp"
#include "photometric.hpp"
#include "scale_ui.hpp"
#include "sphere_ui.hpp"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <commdlg.h>
#include <commctrl.h>
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

#ifndef WHAT_A_RELIEF_VERSION
#define WHAT_A_RELIEF_VERSION "0.2.1"
#endif

namespace {

constexpr size_t kMinImages = 3;
constexpr size_t kMaxImages = 25;

#ifdef _WIN32

constexpr int kWindowWidth = 740;
constexpr int kWindowHeight = 640;
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
constexpr int kIdPromptEdit = 2001;
constexpr int kIdPromptOk = 2002;
constexpr int kIdPromptCancel = 2003;

struct SetupDialogState {
    Options* opt = nullptr;
    bool running = true;
    bool accepted = false;
    HWND hwnd = nullptr;
    HWND nextStepLabel = nullptr;
    HWND tabControl = nullptr;
    std::vector<HWND> tabPages;
    HWND imageStatus = nullptr;
    HWND outputStatus = nullptr;
    HWND lightsStatus = nullptr;
    HWND lightingCombo = nullptr;
    HWND sphereButton = nullptr;
    HWND sphereStatus = nullptr;
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
    HWND srgbCheck = nullptr;
    HWND heightCheck = nullptr;
    HWND meshCheck = nullptr;
    HWND printableMeshCheck = nullptr;
    HWND meshStepEdit = nullptr;
    HWND printableThicknessEdit = nullptr;
    HWND rtiCheck = nullptr;
    HWND rtiLayoutCombo = nullptr;
    HWND rtiColorCombo = nullptr;
    HWND relightCheck = nullptr;
    HWND specularDiagnosticsCheck = nullptr;
    HWND neuralFusionCheck = nullptr;
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

HWND gProgressWindow = nullptr;
HWND gProgressLabel = nullptr;
HWND gProgressBar = nullptr;

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
        WS_CHILD | (index == 0 ? WS_VISIBLE : 0),
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

LRESULT CALLBACK progressWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    (void)wParam;
    (void)lParam;
    if (msg == WM_CLOSE) {
        return 0;
    }
    if (msg == WM_DESTROY) {
        if (gProgressWindow == hwnd) {
            gProgressWindow = nullptr;
            gProgressLabel = nullptr;
            gProgressBar = nullptr;
        }
        return 0;
    }
    return DefWindowProcA(hwnd, msg, wParam, lParam);
}

void pumpGuiMessages() {
    MSG msg = {};
    while (PeekMessageA(&msg, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessageA(&msg);
    }
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

bool isSectionHeader(const SetupDialogState& state, HWND hwnd) {
    return std::find(state.sectionHeaders.begin(), state.sectionHeaders.end(), hwnd) != state.sectionHeaders.end();
}

std::string optionalThenStart(const std::string& text) {
    return text + " Next required (bottom right): click Start.";
}

std::string nextStepText(const SetupDialogState& state, bool& required) {
    const Options& opt = *state.opt;
    required = true;
    if (opt.imagePaths.empty()) {
        return "Next required (Project tab): select 3 to 25 images.";
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
    return "Next required (bottom right): click Start when the tabs look right.";
}

void updateSetupControls(SetupDialogState& state) {
    Options& opt = *state.opt;
    SetWindowTextA(state.imageStatus, imageStatusText(opt.imagePaths).c_str());
    SetWindowTextA(state.outputStatus, outputStatusText(opt.outputDir).c_str());
    SetWindowTextA(state.lightsStatus, lightsStatusText(opt.lightsFile).c_str());
    SetWindowTextA(state.sphereStatus, opt.hasSphere ? "Sphere marked" : "No sphere marked");
    SetWindowTextA(state.cropStatus, opt.hasCrop ? "Crop selected" : "No crop selected");
    SetWindowTextA(state.heightMaskStatus, heightMaskStatusText(opt).c_str());
    if (state.nextStepLabel != nullptr) {
        SetWindowTextA(state.nextStepLabel, nextStepText(state, state.nextStepRequired).c_str());
        InvalidateRect(state.nextStepLabel, nullptr, TRUE);
    }

    const bool calibrated = comboSelection(state.lightingCombo) == 0;
    const bool usingLightsFile = !opt.lightsFile.empty();
    EnableWindow(state.lightingCombo, !usingLightsFile);
    EnableWindow(state.sphereButton, calibrated && !usingLightsFile);
    EnableWindow(state.cropButton, !opt.imagePaths.empty());
    EnableWindow(state.heightMaskButton, !opt.imagePaths.empty() && buttonChecked(state.heightCheck));
    EnableWindow(state.clearHeightMaskButton, opt.hasHeightMask || !opt.heightMaskPath.empty());
    EnableWindow(state.solverCombo, calibrated);
    const bool heightEnabled = buttonChecked(state.heightCheck);
    EnableWindow(state.heightSolverCombo, heightEnabled);
    EnableWindow(state.heightFlattenCombo, heightEnabled);
    EnableWindow(state.meshCheck, heightEnabled || buttonChecked(state.meshCheck));
    EnableWindow(state.printableMeshCheck, heightEnabled || buttonChecked(state.printableMeshCheck));
    EnableWindow(state.meshStepEdit, buttonChecked(state.meshCheck) || buttonChecked(state.printableMeshCheck));
    EnableWindow(state.printableThicknessEdit, buttonChecked(state.printableMeshCheck));
    EnableWindow(state.nearFieldCheck, calibrated && !usingLightsFile);
    const bool supportsNeuralFusion = calibrated &&
        state.opt->imagePaths.size() >= kMinImages &&
        state.opt->imagePaths.size() <= kMaxImages;
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

void selectImages(SetupDialogState& state) {
    try {
        state.opt->imagePaths = chooseImageFiles(state.hwnd);
        state.opt->hasSphere = false;
        state.opt->hasCrop = false;
        state.opt->heightMask.release();
        state.opt->hasHeightMask = false;
        state.opt->heightMaskPath.clear();
        if (!state.opt->imagePaths.empty() &&
            (state.opt->outputDir.empty() || state.opt->outputDir == "out")) {
            state.opt->outputDir = (fs::path(state.opt->imagePaths.front()).parent_path() / "what-a-relief").string();
        }
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

void markHeightMask(SetupDialogState& state) {
    if (state.opt->imagePaths.empty()) {
        showOwnerMessage(state.hwnd, "Specimen Mask", "Select images first.", MB_ICONWARNING);
        return;
    }
    try {
        state.opt->heightMask = chooseHeightMaskInteractive(loadDisplayImage(state.opt->imagePaths.front()));
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
        const double pixels = chooseScaleLineInteractive(loadDisplayImage(state.opt->imagePaths.front()));
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
    if (!opt.uncalibratedLighting && !opt.lightsFile.empty()) {
        try {
            (void)loadLightsFile(opt.lightsFile, opt.imagePaths, nullptr);
        } catch (const std::exception& e) {
            showOwnerMessage(
                state.hwnd,
                "Setup",
                std::string("Loaded calibration does not match the selected images.\n\n") + e.what(),
                MB_ICONWARNING);
            return false;
        }
    }
    if (!opt.uncalibratedLighting && buttonChecked(state.neuralFusionCheck) &&
        (opt.imagePaths.size() < kMinImages || opt.imagePaths.size() > kMaxImages)) {
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
    opt.srgb = buttonChecked(state.srgbCheck);
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
    opt.neuralFusion = !opt.uncalibratedLighting &&
        opt.imagePaths.size() >= kMinImages &&
        opt.imagePaths.size() <= kMaxImages &&
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
    state.printableMeshCheck = makeControl(geometryPage, "BUTTON", "Export watertight printable PLY solid", BS_AUTOCHECKBOX, kIdPrintableMesh, kControlX, y, kControlWidth, 24);
    setButtonChecked(state.printableMeshCheck, !state.opt->printableMeshPath.empty());

    y += 40;
    makeLabel(geometryPage, "Mesh Step", kMargin, y, kLabelWidth, kRowHeight);
    state.meshStepEdit = makeControl(geometryPage, "EDIT", "", ES_LEFT | WS_BORDER | WS_TABSTOP, kIdMeshStep, kControlX, y, 120, kRowHeight);
    setEditInt(state.meshStepEdit, state.opt->meshStep);

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
    state.srgbCheck = makeControl(processingPage, "BUTTON", "Treat JPEG/PNG input as sRGB", BS_AUTOCHECKBOX, kIdSrgb, kControlX, y, kControlWidth, 24);
    setButtonChecked(state.srgbCheck, state.opt->srgb);

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
    state.specularDiagnosticsCheck = makeControl(advancedPage, "BUTTON", "Write experimental specular-cue diagnostics", BS_AUTOCHECKBOX, kIdSpecularDiagnostics, kControlX, y, kControlWidth, 24);
    setButtonChecked(state.specularDiagnosticsCheck, state.opt->specularDiagnostics);

    y += 36;
    state.neuralFusionCheck = makeControl(advancedPage, "BUTTON", "Experimental: PS-FCN neural prior + fusion (3 to 25 calibrated images)", BS_AUTOCHECKBOX, kIdNeuralFusion, kControlX, y, kControlWidth, 24);
    setButtonChecked(state.neuralFusionCheck, state.opt->neuralFusion);

    makeControl(hwnd, "BUTTON", "Start", BS_DEFPUSHBUTTON, kIdStart, kWindowWidth - 290, kTabTop + kTabHeight + 22, 110, 34);
    makeControl(hwnd, "BUTTON", "Cancel", BS_PUSHBUTTON, kIdCancel, kWindowWidth - 165, kTabTop + kTabHeight + 22, 110, 34);

    TabCtrl_SetCurSel(state.tabControl, 0);
    showSelectedTabPage(state);
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
            }
            updateSetupControls(*state);
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
            updateSetupScrollInfo(*state);
            scrollSetupWindow(*state, state->scrollY);
            return 0;
        }
        break;
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
            state->accepted = false;
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

void showGuiProgress(const std::string& title, const std::string& text) {
#ifdef _WIN32
    INITCOMMONCONTROLSEX controls = {};
    controls.dwSize = sizeof(controls);
    controls.dwICC = ICC_PROGRESS_CLASS;
    InitCommonControlsEx(&controls);

    const HINSTANCE instance = GetModuleHandleA(nullptr);
    const char* className = "WhatAReliefProgressWindow";
    WNDCLASSA wc = {};
    wc.lpfnWndProc = progressWndProc;
    wc.hInstance = instance;
    wc.lpszClassName = className;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);
    RegisterClassA(&wc);

    if (gProgressWindow != nullptr) {
        if (gProgressLabel != nullptr) {
            SetWindowTextA(gProgressLabel, text.c_str());
        }
        if (gProgressBar != nullptr) {
            SendMessageA(gProgressBar, PBM_SETPOS, 0, 0);
        }
        pumpGuiMessages();
        return;
    }

    gProgressWindow = CreateWindowExA(
        WS_EX_DLGMODALFRAME | WS_EX_TOPMOST,
        className,
        title.c_str(),
        WS_OVERLAPPED | WS_CAPTION,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        520,
        150,
        nullptr,
        nullptr,
        instance,
        nullptr);
    if (gProgressWindow == nullptr) {
        return;
    }

    gProgressLabel = makeControl(gProgressWindow, "STATIC", text.c_str(), SS_LEFT, 0, 20, 20, 460, 28);
    gProgressBar = makeControl(gProgressWindow, PROGRESS_CLASSA, "", 0, kIdProgressBar, 20, 60, 460, 24);
    SendMessageA(gProgressBar, PBM_SETRANGE, 0, MAKELPARAM(0, 100));
    SendMessageA(gProgressBar, PBM_SETPOS, 0, 0);
    ShowWindow(gProgressWindow, SW_SHOW);
    UpdateWindow(gProgressWindow);
    pumpGuiMessages();
#else
    (void)title;
    (void)text;
#endif
}

void updateGuiProgress(const std::string& text, int percent) {
#ifdef _WIN32
    if (gProgressWindow == nullptr) {
        return;
    }
    if (gProgressLabel != nullptr) {
        SetWindowTextA(gProgressLabel, text.c_str());
    }
    if (gProgressBar != nullptr) {
        SendMessageA(gProgressBar, PBM_SETPOS, std::clamp(percent, 0, 100), 0);
    }
    UpdateWindow(gProgressWindow);
    pumpGuiMessages();
#else
    (void)text;
    (void)percent;
#endif
}

void closeGuiProgress() {
#ifdef _WIN32
    if (gProgressWindow != nullptr) {
        DestroyWindow(gProgressWindow);
        pumpGuiMessages();
    }
#endif
}

void launchGuiWorkflow(Options& opt) {
#ifdef _WIN32
    const HINSTANCE instance = GetModuleHandleA(nullptr);
    INITCOMMONCONTROLSEX controls = {};
    controls.dwSize = sizeof(controls);
    controls.dwICC = ICC_TAB_CLASSES;
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
    const std::string windowTitle = "what-a-relief " WHAT_A_RELIEF_VERSION;
    HWND hwnd = CreateWindowExA(
        WS_EX_DLGMODALFRAME,
        className,
        windowTitle.c_str(),
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
