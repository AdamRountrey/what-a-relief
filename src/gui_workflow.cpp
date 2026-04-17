#include "gui_workflow.hpp"

#include "crop_ui.hpp"
#include "image_io.hpp"
#include "sphere_ui.hpp"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <commdlg.h>
#include <shlobj.h>
#endif

#include <algorithm>
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

std::vector<std::string> chooseImageFiles() {
    std::vector<char> buffer(65536, '\0');
    OPENFILENAMEA ofn = {};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = nullptr;
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

std::string chooseOutputFolder() {
    char displayName[MAX_PATH] = {};
    BROWSEINFOA info = {};
    info.hwndOwner = nullptr;
    info.pszDisplayName = displayName;
    info.lpszTitle = "Choose output folder for photometric stereo results";
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

#else

#endif

std::string joinLines(const std::vector<std::string>& values) {
    std::ostringstream out;
    for (const std::string& value : values) {
        out << "  " << value << '\n';
    }
    return out.str();
}

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
    showGuiInfo(
        "What A Relief",
        "Select 3 to 25 images.\n\n"
        "After that, choose an output folder and whether to use highlight-sphere calibration.");

    opt.imagePaths = chooseImageFiles();
    if (opt.imagePaths.size() < kMinImages || opt.imagePaths.size() > kMaxImages) {
        throw std::runtime_error(
            "Select 3 to 25 images. You selected " +
            std::to_string(opt.imagePaths.size()) + ".\n\n" +
            joinLines(opt.imagePaths));
    }

    opt.outputDir = chooseOutputFolder();
    const bool useSphere = askGuiYesNo(
        "Lighting Calibration",
        "Use the highlight sphere to calibrate light directions?\n\n"
        "Yes gives the most physically meaningful normals and height.\n"
        "No skips sphere marking and estimates relative topography from the image stack. No requires at least 4 images.",
        true);
    opt.uncalibratedLighting = !useSphere;
    if (opt.uncalibratedLighting && opt.imagePaths.size() < 4) {
        throw std::runtime_error("Uncalibrated no-sphere mode requires at least 4 images.");
    }
    if (useSphere) {
        showGuiInfo(
            "Mark the highlight sphere",
            "In the next window, click three points on the sphere edge.\n\n"
            "Mouse wheel or +/- zooms. Right-drag, WASD, or arrow keys pan.\n"
            "Press Enter or Space to accept. Press R to reset.");

        opt.sphere = chooseSphereInteractive(loadDisplayImage(opt.imagePaths.front()));
        opt.hasSphere = true;
    } else if (askGuiYesNo(
                   "Crop Surface Region?",
                   "Crop to the surface area for the uncalibrated solve?\n\n"
                   "This is recommended when the image includes a shiny sphere, fixture, or dark background.",
                   true)) {
        opt.crop = chooseCropInteractive(loadDisplayImage(opt.imagePaths.front()));
        opt.hasCrop = true;
    }
    opt.calculateHeight = askGuiYesNo(
        "Calculate Height Preview?",
        "Calculate height.png and height.pfm?\n\n"
        "No is faster and still writes normals, albedo, residual, lights, and liquid_metal.png.\n"
        "Yes adds a fast DCT/Poisson preview integration of the normals.",
        false);
    if (opt.calculateHeight && askGuiYesNo(
                                   "Export PLY Mesh?",
                                   "Export a 3D PLY mesh from the height preview?\n\n"
                                   "The mesh is a visualization surface, not calibrated metrology.",
                                   false)) {
        opt.meshPath = (fs::path(opt.outputDir) / "surface.ply").string();
    }
    showGuiInfo(
        "Processing",
        "Processing will start now.\n\n"
        "Large microscope images can still take a little while. Height preview is only calculated if you selected it.");
#else
    (void)opt;
    throw std::runtime_error("GUI image loading is only implemented on Windows. Use --image arguments on this platform.");
#endif
}
