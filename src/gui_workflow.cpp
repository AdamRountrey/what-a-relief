#include "gui_workflow.hpp"

#include "image_io.hpp"
#include "sphere_ui.hpp"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <commdlg.h>
#include <shlobj.h>
#endif

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#ifdef _WIN32

void showMessage(const std::string& title, const std::string& text) {
    MessageBoxA(nullptr, text.c_str(), title.c_str(), MB_OK | MB_ICONINFORMATION);
}

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
    ofn.lpstrTitle = "Select exactly 4 or 8 photometric stereo images";
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

void showMessage(const std::string&, const std::string&) {
}

#endif

std::string joinLines(const std::vector<std::string>& values) {
    std::ostringstream out;
    for (const std::string& value : values) {
        out << "  " << value << '\n';
    }
    return out.str();
}

} // namespace

void launchGuiWorkflow(Options& opt) {
#ifdef _WIN32
    showMessage(
        "Photometric Stereo Spheres",
        "Select exactly 4 or 8 images.\n\n"
        "After that, choose an output folder, then mark the highlight sphere by dragging from its center to its edge.");

    opt.imagePaths = chooseImageFiles();
    if (opt.imagePaths.size() != 4 && opt.imagePaths.size() != 8) {
        throw std::runtime_error(
            "Select exactly 4 or 8 images. You selected " +
            std::to_string(opt.imagePaths.size()) + ".\n\n" +
            joinLines(opt.imagePaths));
    }

    opt.outputDir = chooseOutputFolder();
    showMessage(
        "Mark the highlight sphere",
        "In the next window, click-drag from the center of the highlight sphere to the sphere edge.\n\n"
        "Press Enter or Space to accept. Press R to redraw.");

    opt.sphere = chooseSphereInteractive(loadDisplayImage(opt.imagePaths.front()));
    opt.hasSphere = true;
#else
    (void)opt;
    throw std::runtime_error("GUI image loading is only implemented on Windows. Use --image arguments on this platform.");
#endif
}
