#include "relight_ui.hpp"

#ifdef PS_NO_GUI

#include <stdexcept>

void launchRelightViewer(const cv::Mat&, const cv::Mat&, const std::string&) {
    throw std::runtime_error("Interactive relighting is disabled in this build.");
}

#else

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace {

struct RelightState {
    cv::Mat previewNormals;
    cv::Mat previewMask;
    cv::Mat fullNormals;
    cv::Mat fullMask;
    std::string outputDir;
    cv::Vec3f light = cv::Vec3f(-0.45f, 0.35f, 0.82f);
    bool dirty = true;
    bool dragging = false;
    bool saved = false;
};

cv::Vec3f normalizeVector(const cv::Vec3f& v) {
    const float length = std::sqrt(v.dot(v));
    if (length <= 1.0e-6f) {
        return cv::Vec3f(0.0f, 0.0f, 1.0f);
    }
    return v / length;
}

float smoothstep(float edge0, float edge1, float x) {
    const float t = std::clamp((x - edge0) / std::max(1.0e-6f, edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

cv::Vec3f mixColor(const cv::Vec3f& a, const cv::Vec3f& b, float t) {
    return a * (1.0f - t) + b * t;
}

cv::Vec3f reflectVector(const cv::Vec3f& incident, const cv::Vec3f& normal) {
    return incident - 2.0f * incident.dot(normal) * normal;
}

cv::Mat renderRelitMetal(const cv::Mat& normalMap, const cv::Mat& mask, const cv::Vec3f& lightInput) {
    const cv::Vec3f light = normalizeVector(lightInput);
    const cv::Vec3f view(0.0f, 0.0f, 1.0f);
    const cv::Vec3f fillLight = normalizeVector(cv::Vec3f(-light[0] * 0.45f, -light[1] * 0.45f, 0.85f));
    const cv::Vec3f darkSteel(0.035f, 0.045f, 0.050f);
    const cv::Vec3f silver(0.72f, 0.76f, 0.78f);
    const cv::Vec3f coolCyan(0.42f, 0.72f, 0.80f);
    const cv::Vec3f warmGold(0.95f, 0.76f, 0.42f);

    cv::Mat out(normalMap.rows, normalMap.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int y = 0; y < normalMap.rows; ++y) {
        const cv::Vec3f* nrow = normalMap.ptr<cv::Vec3f>(y);
        const uchar* mrow = mask.ptr<uchar>(y);
        cv::Vec3b* outRow = out.ptr<cv::Vec3b>(y);
        for (int x = 0; x < normalMap.cols; ++x) {
            if (mrow[x] == 0) {
                continue;
            }

            const cv::Vec3f n = normalizeVector(nrow[x]);
            const cv::Vec3f reflectedView = normalizeVector(reflectVector(-view, n));
            const float ndv = std::clamp(n.dot(view), 0.0f, 1.0f);
            const float key = std::max(0.0f, n.dot(light));
            const float fill = std::max(0.0f, n.dot(fillLight));
            const float stripe = smoothstep(0.28f, 0.88f, 0.5f + 0.5f * (reflectedView[0] * 0.70f + reflectedView[1] * 0.30f));

            cv::Vec3f base = mixColor(darkSteel, silver, stripe);
            base = mixColor(base, coolCyan, 0.20f * smoothstep(0.05f, 0.90f, reflectedView[1] * 0.5f + 0.5f));
            base = mixColor(base, warmGold, 0.14f * smoothstep(0.25f, 1.00f, reflectedView[0] * -0.5f + 0.5f));

            const cv::Vec3f keyReflection = reflectVector(-light, n);
            const cv::Vec3f fillReflection = reflectVector(-fillLight, n);
            const float specKey = std::pow(std::max(0.0f, keyReflection.dot(view)), 58.0f);
            const float specFill = std::pow(std::max(0.0f, fillReflection.dot(view)), 28.0f);
            const float rim = std::pow(1.0f - ndv, 2.2f);

            cv::Vec3f rgb = base * (0.38f + 0.54f * key + 0.16f * fill);
            rgb += cv::Vec3f(1.0f, 0.96f, 0.88f) * (1.10f * specKey + 0.30f * specFill);
            rgb += coolCyan * (0.18f * rim);

            for (int c = 0; c < 3; ++c) {
                rgb[c] = 1.0f - std::exp(-std::max(0.0f, rgb[c]) * 1.35f);
                rgb[c] = std::pow(std::clamp(rgb[c], 0.0f, 1.0f), 1.0f / 2.2f);
            }
            outRow[x] = cv::Vec3b(
                static_cast<uchar>(std::clamp(rgb[2] * 255.0f, 0.0f, 255.0f)),
                static_cast<uchar>(std::clamp(rgb[1] * 255.0f, 0.0f, 255.0f)),
                static_cast<uchar>(std::clamp(rgb[0] * 255.0f, 0.0f, 255.0f)));
        }
    }
    return out;
}

cv::Vec3f lightFromMouse(int x, int y, int width, int height) {
    const float nx = std::clamp((static_cast<float>(x) / std::max(1, width - 1)) * 2.0f - 1.0f, -1.0f, 1.0f);
    const float ny = std::clamp(1.0f - (static_cast<float>(y) / std::max(1, height - 1)) * 2.0f, -1.0f, 1.0f);
    const float r2 = nx * nx + ny * ny;
    const float maxR2 = 0.9996f;
    const float scale = r2 > maxR2 ? std::sqrt(maxR2 / r2) : 1.0f;
    const float lx = nx * scale;
    const float ly = ny * scale;
    const float lz = std::sqrt(std::max(0.0004f, 1.0f - lx * lx - ly * ly));
    return normalizeVector(cv::Vec3f(lx, ly, lz));
}

void drawOverlay(cv::Mat& frame, const RelightState& state) {
    const std::string line1 = "Drag to move light; edges give low raking light. S saves. R resets. Esc closes.";
    const std::string line2 =
        "light=(" +
        std::to_string(static_cast<int>(std::lround(state.light[0] * 100.0f))) + ", " +
        std::to_string(static_cast<int>(std::lround(state.light[1] * 100.0f))) + ", " +
        std::to_string(static_cast<int>(std::lround(state.light[2] * 100.0f))) + ")";
    cv::rectangle(frame, cv::Point(0, 0), cv::Point(frame.cols, 58), cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(frame, line1, cv::Point(12, 23), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    cv::putText(frame, line2, cv::Point(12, 48), cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
    if (state.saved) {
        cv::putText(frame, "Saved", cv::Point(frame.cols - 88, frame.rows - 18), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
    }
}

void mouseCallback(int event, int x, int y, int, void* userdata) {
    auto* state = static_cast<RelightState*>(userdata);
    if (event == cv::EVENT_LBUTTONDOWN) {
        state->dragging = true;
    } else if (event == cv::EVENT_LBUTTONUP) {
        state->dragging = false;
    }
    if (event == cv::EVENT_LBUTTONDOWN || (event == cv::EVENT_MOUSEMOVE && state->dragging)) {
        state->light = lightFromMouse(x, y, state->previewNormals.cols, state->previewNormals.rows);
        state->dirty = true;
        state->saved = false;
    }
}

void saveFullResolution(const RelightState& state) {
    fs::create_directories(state.outputDir);
    const fs::path outPath = fs::path(state.outputDir) / "liquid_metal_custom.png";
    cv::imwrite(outPath.string(), renderRelitMetal(state.fullNormals, state.fullMask, state.light));
}

} // namespace

void launchRelightViewer(const cv::Mat& normalMap, const cv::Mat& validMask, const std::string& outputDir) {
    if (normalMap.empty() || validMask.empty()) {
        throw std::runtime_error("Cannot relight an empty normal map.");
    }

    RelightState state;
    state.fullNormals = normalMap;
    state.fullMask = validMask;
    state.outputDir = outputDir;
    state.light = normalizeVector(state.light);

    const double maxW = 1400.0;
    const double maxH = 900.0;
    const double scale = std::min(1.0, std::min(maxW / normalMap.cols, maxH / normalMap.rows));
    if (scale < 1.0) {
        cv::resize(normalMap, state.previewNormals, cv::Size(), scale, scale, cv::INTER_AREA);
        cv::resize(validMask, state.previewMask, cv::Size(), scale, scale, cv::INTER_NEAREST);
    } else {
        state.previewNormals = normalMap;
        state.previewMask = validMask;
    }

    const std::string windowName = "Interactive specular relight";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(windowName, mouseCallback, &state);

    cv::Mat preview;
    bool done = false;
    while (!done) {
        if (state.dirty || preview.empty()) {
            preview = renderRelitMetal(state.previewNormals, state.previewMask, state.light);
            drawOverlay(preview, state);
            state.dirty = false;
        }
        cv::imshow(windowName, preview);
        const int key = cv::waitKeyEx(20);
        if (cv::getWindowProperty(windowName, cv::WND_PROP_VISIBLE) < 1.0) {
            done = true;
            continue;
        }
        const int ascii = key & 0xff;
        if (ascii == 27 || ascii == 'q' || ascii == 'Q') {
            done = true;
        } else if (ascii == 'r' || ascii == 'R') {
            state.light = normalizeVector(cv::Vec3f(-0.45f, 0.35f, 0.82f));
            state.dirty = true;
            state.saved = false;
        } else if (ascii == 's' || ascii == 'S') {
            saveFullResolution(state);
            state.saved = true;
            state.dirty = true;
        }
    }

    cv::destroyWindow(windowName);
}

#endif
