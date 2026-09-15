#include "gui_workflow.hpp"
#include "checked_io.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#define NOMINMAX
#include <windows.h>
#include <commctrl.h>
#include <atomic>
#include <thread>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;
namespace {
constexpr int startId = 1016, cancelId = 1017, statusId = 1053;
constexpr int mitsubaQualityId = 1046;
constexpr int newId = 3001, openId = 3002, exitId = 3003;
struct Driver {
    Options* options = nullptr;
    int phase = 0;
    std::atomic_int runs{0};
    bool exitWhileBusy = false;
    HWND window = nullptr;
    ULONGLONG began = GetTickCount64();
    std::string failure;
    std::string firstOutput;
    fs::path capturePath;
};
Driver* active = nullptr;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

BOOL CALLBACK findWindow(HWND window, LPARAM data) {
    char name[128] = {};
    GetClassNameA(window, name, sizeof(name));
    if (std::string(name) == "WhatAReliefSetupWindow") *reinterpret_cast<HWND*>(data) = window;
    return TRUE;
}

std::string status(HWND window) {
    char text[4096] = {};
    GetWindowTextA(GetDlgItem(window, statusId), text, sizeof(text));
    return text;
}

void captureWindow(HWND window, const fs::path& path) {
    RECT bounds = {};
    GetWindowRect(window, &bounds);
    const int width = bounds.right - bounds.left, height = bounds.bottom - bounds.top;
    HDC screen = GetDC(nullptr);
    HDC memory = CreateCompatibleDC(screen);
    BITMAPINFO info = {};
    info.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    info.bmiHeader.biWidth = width;
    info.bmiHeader.biHeight = -height;
    info.bmiHeader.biPlanes = 1;
    info.bmiHeader.biBitCount = 32;
    void* bits = nullptr;
    HBITMAP bitmap = CreateDIBSection(screen, &info, DIB_RGB_COLORS, &bits, nullptr, 0);
    HGDIOBJ previous = SelectObject(memory, bitmap);
    if (IsWindowVisible(window)) {
        RedrawWindow(window, nullptr, nullptr, RDW_INVALIDATE | RDW_ERASE | RDW_ALLCHILDREN | RDW_UPDATENOW);
        PrintWindow(window, memory, 2);
    } else {
        SendMessageA(window, WM_PRINT, reinterpret_cast<WPARAM>(memory), PRF_CLIENT | PRF_NONCLIENT | PRF_CHILDREN | PRF_ERASEBKGND);
    }
    cv::Mat image;
    if (bits) cv::cvtColor(cv::Mat(height, width, CV_8UC4, bits), image, cv::COLOR_BGRA2BGR);
    SelectObject(memory, previous);
    DeleteObject(bitmap);
    DeleteDC(memory);
    ReleaseDC(nullptr, screen);
    if (!image.empty()) {
        writeImageChecked(path, image);
        std::cout << "GUI capture: " << path.string() << '\n';
    }
}

void CALLBACK drive(HWND, UINT, UINT_PTR, DWORD) {
    if (GetTickCount64() - active->began > 20000) {
        std::cerr << "GUI lifecycle timed out: " << active->failure << '\n';
        std::exit(2);
    }
    HWND window = nullptr;
    EnumThreadWindows(GetCurrentThreadId(), findWindow, reinterpret_cast<LPARAM>(&window));
    if (!window || !GetDlgItem(window, startId) || !IsWindowEnabled(GetDlgItem(window, startId))) return;
    try {
        if (!active->window) active->window = window;
        require(active->window == window, "Application recreated its main window");
        if (active->phase == 0) {
            require(SendMessageA(GetDlgItem(window, 1038), PBM_SETMARQUEE, FALSE, 0) != 0,
                "Modern common-controls manifest missing; activity bar would not animate");
            require(GetMenuState(GetMenu(window), openId, MF_BYCOMMAND) != static_cast<UINT>(-1), "File > Open is missing");
            HWND quality = nullptr;
            for (HWND page = GetWindow(window, GW_CHILD); page; page = GetWindow(page, GW_HWNDNEXT)) {
                if (HWND candidate = GetDlgItem(page, mitsubaQualityId)) quality = candidate;
            }
            require(quality && SendMessageA(quality, CB_GETCOUNT, 0, 0) == 3,
                "Mitsuba quality selector must expose exactly Preview, Standard, and High detail");
            require(SendMessageA(quality, CB_GETCURSEL, 0, 0) == 1,
                "A stored Ultra selection must fall back to Standard in the GUI");
            for (int index = 0; index < 3; ++index) {
                char label[256] = {};
                SendMessageA(quality, CB_GETLBTEXT, index, reinterpret_cast<LPARAM>(label));
                require(std::string(label).find("Ultra") == std::string::npos,
                    "Ultra remains visible in the GUI quality selector");
            }
            active->phase = 1;
            SendMessageA(window, WM_COMMAND, startId, 0);
        } else if (active->phase == 1 && !active->exitWhileBusy) {
            require(active->runs == 1 && status(window).find("Complete.") != std::string::npos,
                "Completion did not return to editable window");
            require(status(window).find("Shadow refinement applied") != std::string::npos &&
                status(window).find("25.0% lower") != std::string::npos,
                "GUI completion did not expose the accepted shadow-refinement benefit");
            HWND tabs = GetDlgItem(window, 1037);
            TabCtrl_SetCurSel(tabs, 5);
            NMHDR changed{tabs, 1037, TCN_SELCHANGE};
            SendMessageA(window, WM_NOTIFY, 1037, reinterpret_cast<LPARAM>(&changed));
            HWND advanced = nullptr;
            for (HWND page = GetWindow(window, GW_CHILD); page; page = GetWindow(page, GW_HWNDNEXT)) {
                if (GetDlgItem(page, 1044)) advanced = page;
            }
            require(advanced && (GetWindowLongPtrA(advanced, GWL_STYLE) & WS_VISIBLE) &&
                IsWindowEnabled(advanced) && GetWindow(advanced, GW_HWNDPREV) == nullptr, "Advanced page hidden, disabled, or behind another control after completion");
            require((GetWindowLongPtrA(GetDlgItem(advanced, 1044), GWL_STYLE) & WS_VISIBLE) != 0, "Advanced controls disappeared");
            captureWindow(window, active->capturePath.parent_path() / "advanced-after-run.png");
            TabCtrl_SetCurSel(tabs, 6);
            SendMessageA(window, WM_NOTIFY, 1037, reinterpret_cast<LPARAM>(&changed));
            captureWindow(window, active->capturePath.parent_path() / "progress-after-run.png");
            TabCtrl_SetCurSel(tabs, 0);
            SendMessageA(window, WM_NOTIFY, 1037, reinterpret_cast<LPARAM>(&changed));
            captureWindow(window, active->capturePath);
            for (int height : {724, 780}) {
                SetWindowPos(window, nullptr, 0, 0, 740, height, SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
                RECT client = {}, button = {}, text = {};
                GetClientRect(window, &client);
                GetWindowRect(GetDlgItem(window, startId), &button);
                MapWindowPoints(nullptr, window, reinterpret_cast<POINT*>(&button), 2);
                GetWindowRect(GetDlgItem(window, statusId), &text);
                MapWindowPoints(nullptr, window, reinterpret_cast<POINT*>(&text), 2);
                require(button.bottom <= client.bottom && button.right <= client.right, "Run button clipped outside the client area");
                require(text.bottom < button.top && text.top >= 534, "Footer overlaps setup tabs or Run button");
                if (height == 724) captureWindow(window, active->capturePath.parent_path() / "compact-window.png");
            }
            active->phase = 2;
            SendMessageA(window, WM_COMMAND, startId, 0);
        } else if (active->phase == 2) {
            require(active->runs == 2 && status(window).find("Processing failed:") != std::string::npos, "Error did not retain session");
            active->phase = 3;
            SendMessageA(window, WM_COMMAND, startId, 0);
        } else if (active->phase == 3) {
            require(active->runs == 3 && status(window).find("Run canceled.") != std::string::npos, "Cancellation did not retain session");
            SendMessageA(window, WM_COMMAND, newId, 0);
            require(IsWindow(window) && active->options->imagePaths.empty() && !active->options->hasSphere &&
                !active->options->hasHeightMask && active->options->lightsFile.empty(), "New project leaked prior inputs");
            active->phase = 4;
            SendMessageA(window, WM_COMMAND, exitId, 0);
        }
    } catch (const std::exception& e) {
        active->failure = e.what();
        SendMessageA(window, WM_CLOSE, 0, 0);
    }
}

GuiRunResult process(Options& opt, const GuiProgress& progress) {
    ++active->runs;
    const HWND window = active->window;
    require(IsWindow(window), "Main window closed during processing");
    require(!IsWindowEnabled(GetDlgItem(window, startId)), "Reentrant Start is enabled");
    require((GetMenuState(GetMenu(window), newId, MF_BYCOMMAND) & MF_GRAYED) != 0, "New project enabled during solve");
    const auto images = opt.imagePaths;
    SendMessageA(window, WM_COMMAND, newId, 0);
    SendMessageA(window, WM_COMMAND, startId, 0);
    require(opt.imagePaths == images, "Queued project command modified live solve inputs");
    if (active->exitWhileBusy) {
        SendMessageA(window, WM_CLOSE, 0, 0);
        require(IsWindow(window), "Exit destroyed window while processor still owned state");
        progress({"Cancel for exit", 20});
    }
    if (active->runs == 1) {
        fs::create_directories(opt.outputDir);
        CheckedOutputFile sentinel(fs::path(opt.outputDir) / "completed.txt");
        sentinel.stream() << "keep previous results";
        sentinel.commit();
        active->firstOutput = opt.outputDir;
        ProgressUpdate update{"Synthetic run", 50, "Test iterations", 5, 12, 10, "current stage"};
        update.preview = cv::Mat(32, 64, CV_8UC3, cv::Scalar(180, 110, 130));
        update.previewIteration = 5;
        update.previewTotal = 12;
        progress(update);
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        GuiRunResult result;
        result.shadow.attempted = true;
        result.shadow.applied = true;
        result.shadow.decision = "accepted";
        result.shadow.balancedMismatchBefore = 0.2;
        result.shadow.balancedMismatchAfter = 0.15;
        result.shadow.correctionRmsPixels = 1.25;
        return result;
    }
    require(opt.outputDir != active->firstOutput && fs::is_regular_file(fs::path(active->firstOutput) / "completed.txt"), "Rerun overwrote previous output");
    if (active->runs == 2) throw std::runtime_error("synthetic processing failure");
    SendMessageA(window, WM_COMMAND, cancelId, 0);
    progress({"Cancel checkpoint", 50});
    throw std::runtime_error("Cancel did not propagate");
}
} // namespace

int main(int argc, char**) {
    try {
        ShadowRefinementSummary accepted;
        accepted.attempted = true;
        accepted.applied = true;
        accepted.balancedMismatchBefore = 0.2;
        accepted.balancedMismatchAfter = 0.15;
        accepted.correctionRmsPixels = 1.25;
        const std::string acceptedMessage = formatShadowRefinementSummary(accepted);
        require(
            acceptedMessage.find("Shadow refinement applied") != std::string::npos &&
                acceptedMessage.find("0.2000 -> 0.1500 (25.0% lower)") != std::string::npos &&
                acceptedMessage.find("1.250 height pixels") != std::string::npos,
            "Accepted shadow-refinement summary omitted its benefit");

        ShadowRefinementSummary rejected;
        rejected.attempted = true;
        rejected.decision = "rejected_withheld_lights_worsened";
        const std::string rejectedMessage = formatShadowRefinementSummary(rejected);
        require(
            rejectedMessage.find("not applied: held-out lights worsened") != std::string::npos &&
                rejectedMessage.find("original height retained") != std::string::npos,
            "Rejected shadow-refinement summary omitted its reason or fallback");
        require(formatShadowRefinementSummary({}).empty(),
            "Unrequested shadow refinement produced a completion summary");

        ProgressTiming timing;
        timing.update({"Loading", 0, "Images"}, 0);
        require(timing.remaining(1) < 0, "ETA guessed before any samples");
        timing.update({"Image 2/10", 20, "Images", 2, 10}, 4);
        require(timing.remaining(4) == 16 && timing.remaining(6) == 14, "Measured stage ETA incorrect");
        timing.update({"Validation", 82, "Validation"}, 7);
        require(timing.remaining(7) < 0, "ETA carried into unmeasured stage");
        timing.update({"Iterating", 60, "Inverse", 5, 10, 30}, 8);
        require(timing.remaining(8) == 30 && timing.remaining(42) < 0, "Explicit or overdue ETA incorrect");
        const fs::path root = fs::absolute("gui-lifecycle-test-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(root);
        Options initial;
        initial.guiMode = true;
        initial.calculateHeight = false;
        initial.mitsubaQualityMode = MitsubaQualityMode::Ultra;
        initial.outputDir = (root / "output").string();
        initial.hasSphere = true;
        initial.sphere = {4, 4, 2};
        for (int i = 0; i < 4; ++i) {
            initial.imagePaths.push_back((root / ("image_" + std::to_string(i) + ".png")).string());
            writeImageChecked(initial.imagePaths.back(), cv::Mat(16, 16, CV_8U, cv::Scalar(128)));
        }
        for (bool closing : {false, true}) {
            Options options = initial;
            Driver driver;
            driver.options = &options;
            driver.exitWhileBusy = closing;
            driver.capturePath = root / "completed-window.png";
            active = &driver;
            const UINT_PTR timer = SetTimer(nullptr, 0, 10, drive);
            launchGuiApplication(options, process, argc > 1);
            KillTimer(nullptr, timer);
            require(driver.failure.empty(), driver.failure.c_str());
            require(driver.runs == (closing ? 1 : 3), "Not all lifecycle stages ran");
            require(!IsWindow(driver.window), "Exit left the application window alive");
        }
        std::cout << "Persistent GUI completion, retry, cancellation, new-project, and exit checks passed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
