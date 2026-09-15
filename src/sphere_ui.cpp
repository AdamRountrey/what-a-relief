#include "sphere_ui.hpp"
#include "photometric.hpp"

#ifdef PS_NO_GUI

#include <stdexcept>

SphereSelection chooseSphereInteractive(
    size_t,
    const SphereImageLoader&,
    const std::vector<std::string>&,
    size_t) {
    throw std::runtime_error("Interactive sphere selection is disabled in this build. Use --sphere or rebuild with OpenCV highgui.");
}

Sphere chooseSphereInteractive(const cv::Mat&) {
    throw std::runtime_error("Interactive sphere selection is disabled in this build. Use --sphere or rebuild with OpenCV highgui.");
}

#else

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct PickerState {
    cv::Mat image;
    cv::Size imageSize;
    cv::Size viewSize;
    size_t imageIndex = 0;
    size_t imageCount = 1;
    std::vector<std::string> imageLabels;
    double zoom = 1.0;
    double minZoom = 1.0;
    double maxZoom = 32.0;
    cv::Point2d origin = cv::Point2d(0.0, 0.0);
    std::vector<cv::Point2d> edgePoints;
    SphereCircleFit fit;
    bool panning = false;
    cv::Point panStartMouse;
    cv::Point2d panStartOrigin = cv::Point2d(0.0, 0.0);
    bool leftDown = false;
    cv::Point leftStartMouse;
    bool done = false;
    bool canceled = false;
};

cv::Point2d toImagePoint(const PickerState& state, int x, int y) {
    return state.origin + cv::Point2d(static_cast<double>(x) / state.zoom, static_cast<double>(y) / state.zoom);
}

cv::Point toViewPoint(const PickerState& state, const cv::Point2d& p) {
    return cv::Point(
        static_cast<int>(std::lround((p.x - state.origin.x) * state.zoom)),
        static_cast<int>(std::lround((p.y - state.origin.y) * state.zoom)));
}

void clampOrigin(PickerState& state) {
    const double visibleW = static_cast<double>(state.viewSize.width) / state.zoom;
    const double visibleH = static_cast<double>(state.viewSize.height) / state.zoom;
    const double maxX = std::max(0.0, static_cast<double>(state.image.cols) - visibleW);
    const double maxY = std::max(0.0, static_cast<double>(state.image.rows) - visibleH);
    state.origin.x = std::clamp(state.origin.x, 0.0, maxX);
    state.origin.y = std::clamp(state.origin.y, 0.0, maxY);
}

void resetView(PickerState& state) {
    state.zoom = state.minZoom;
    state.origin = cv::Point2d(0.0, 0.0);
    clampOrigin(state);
}

void zoomAt(PickerState& state, double factor, const cv::Point& anchor) {
    const cv::Point2d before = toImagePoint(state, anchor.x, anchor.y);
    state.zoom = std::clamp(state.zoom * factor, state.minZoom, state.maxZoom);
    state.origin = before - cv::Point2d(static_cast<double>(anchor.x) / state.zoom, static_cast<double>(anchor.y) / state.zoom);
    clampOrigin(state);
}

void panBy(PickerState& state, double dx, double dy) {
    state.origin += cv::Point2d(dx, dy);
    clampOrigin(state);
}

void addEdgePoint(PickerState& state, const cv::Point2d& p) {
    if (state.edgePoints.size() >= 64) return;
    state.edgePoints.push_back(p);
    state.fit = fitSphereCircle(state.edgePoints);
}

void resetSelection(PickerState& state) {
    state.edgePoints.clear();
    state.fit = {};
    state.leftDown = false;
    state.panning = false;
}

bool circleFitsImage(const PickerState& state) {
    if (!state.fit.hasGeometry) return false;
    const Sphere& sphere = state.fit.sphere;
    return sphere.cx - sphere.radius >= -0.5 && sphere.cy - sphere.radius >= -0.5 &&
        sphere.cx + sphere.radius <= static_cast<double>(state.image.cols) - 0.5 &&
        sphere.cy + sphere.radius <= static_cast<double>(state.image.rows) - 0.5;
}

bool selectionAccepted(const PickerState& state) {
    return state.fit.accepted && circleFitsImage(state);
}

void selectImage(PickerState& state, const SphereImageLoader& loader, size_t index) {
    cv::Mat image = loader(index);
    if (image.empty()) throw std::runtime_error("Cannot choose a sphere from an empty image.");
    if (image.size() != state.imageSize) {
        throw std::runtime_error("All sphere-selection images must have identical dimensions.");
    }
    state.image = std::move(image);
    state.imageIndex = index;
    resetSelection(state);
    resetView(state);
}

void mouseCallback(int event, int x, int y, int flags, void* userdata) {
    auto* state = static_cast<PickerState*>(userdata);
    if (event == cv::EVENT_LBUTTONDOWN) {
        state->leftDown = true;
        state->leftStartMouse = cv::Point(x, y);
    } else if (event == cv::EVENT_LBUTTONUP && state->leftDown) {
        const cv::Point mouse(x, y);
        if (cv::norm(mouse - state->leftStartMouse) <= 4.0) {
            addEdgePoint(*state, toImagePoint(*state, x, y));
        }
        state->leftDown = false;
    } else if (event == cv::EVENT_RBUTTONDOWN || event == cv::EVENT_MBUTTONDOWN) {
        state->panning = true;
        state->panStartMouse = cv::Point(x, y);
        state->panStartOrigin = state->origin;
    } else if (event == cv::EVENT_MOUSEMOVE && state->panning) {
        const cv::Point mouse(x, y);
        state->origin = state->panStartOrigin -
            cv::Point2d(static_cast<double>(mouse.x - state->panStartMouse.x) / state->zoom,
                        static_cast<double>(mouse.y - state->panStartMouse.y) / state->zoom);
        clampOrigin(*state);
    } else if (event == cv::EVENT_RBUTTONUP || event == cv::EVENT_MBUTTONUP) {
        state->panning = false;
    } else if (event == cv::EVENT_MOUSEWHEEL) {
        const int delta = cv::getMouseWheelDelta(flags);
        zoomAt(*state, delta > 0 ? 1.25 : 0.80, cv::Point(x, y));
    }
}

cv::Mat render(const PickerState& state) {
    cv::Mat frame(state.viewSize, state.image.type(), cv::Scalar(0, 0, 0));
    const int x0 = std::clamp(static_cast<int>(std::floor(state.origin.x)), 0, state.image.cols - 1);
    const int y0 = std::clamp(static_cast<int>(std::floor(state.origin.y)), 0, state.image.rows - 1);
    const int x1 = std::clamp(static_cast<int>(std::ceil(state.origin.x + static_cast<double>(state.viewSize.width) / state.zoom)) + 1, x0 + 1, state.image.cols);
    const int y1 = std::clamp(static_cast<int>(std::ceil(state.origin.y + static_cast<double>(state.viewSize.height) / state.zoom)) + 1, y0 + 1, state.image.rows);
    const cv::Rect sourceRect(x0, y0, x1 - x0, y1 - y0);
    cv::Mat scaled;
    cv::resize(state.image(sourceRect), scaled, cv::Size(), state.zoom, state.zoom, state.zoom >= 1.0 ? cv::INTER_LINEAR : cv::INTER_AREA);

    const int dstX = static_cast<int>(std::lround((static_cast<double>(x0) - state.origin.x) * state.zoom));
    const int dstY = static_cast<int>(std::lround((static_cast<double>(y0) - state.origin.y) * state.zoom));
    const cv::Rect dstRect(
        std::max(0, dstX),
        std::max(0, dstY),
        std::min(scaled.cols + std::min(0, dstX), frame.cols - std::max(0, dstX)),
        std::min(scaled.rows + std::min(0, dstY), frame.rows - std::max(0, dstY)));
    if (dstRect.width > 0 && dstRect.height > 0) {
        const cv::Rect srcRect(std::max(0, -dstX), std::max(0, -dstY), dstRect.width, dstRect.height);
        scaled(srcRect).copyTo(frame(dstRect));
    }

    std::string label = state.imageIndex < state.imageLabels.size()
        ? state.imageLabels[state.imageIndex]
        : std::string("image");
    if (label.size() > 70) label = "..." + label.substr(label.size() - 67);
    const std::string line1 = "Image " + std::to_string(state.imageIndex + 1) + "/" +
        std::to_string(state.imageCount) + ": " + label + "   PgUp/PgDn changes image and clears points.";
    const std::string line2 = "Click 5+ well-spaced points on this sphere edge. Wheel/+/- zoom; right-drag or WASD pans.";
    const std::string line3 = "Enter/Space accepts a green fit. Backspace removes last. R resets. 0 fits. Esc cancels.";
    cv::rectangle(frame, cv::Point(0, 0), cv::Point(frame.cols, 88), cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(frame, line1, cv::Point(12, 22), cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    cv::putText(frame, line2, cv::Point(12, 48), cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(230, 230, 230), 1, cv::LINE_AA);
    cv::putText(frame, line3, cv::Point(12, 73), cv::FONT_HERSHEY_SIMPLEX, 0.46, cv::Scalar(210, 210, 210), 1, cv::LINE_AA);

    for (size_t i = 0; i < state.edgePoints.size(); ++i) {
        const cv::Point p = toViewPoint(state, state.edgePoints[i]);
        cv::circle(frame, p, 5, cv::Scalar(0, 255, 255), cv::FILLED, cv::LINE_AA);
        cv::circle(frame, p, 9, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        cv::putText(frame, std::to_string(i + 1), p + cv::Point(9, -8), cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
    }

    if (state.fit.hasGeometry) {
        const cv::Scalar color = selectionAccepted(state) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255);
        const cv::Point c = toViewPoint(state, cv::Point2d(state.fit.sphere.cx, state.fit.sphere.cy));
        const int r = static_cast<int>(std::lround(state.fit.sphere.radius * state.zoom));
        cv::circle(frame, c, r, color, 2, cv::LINE_AA);
        cv::circle(frame, c, 4, color, cv::FILLED, cv::LINE_AA);

        std::ostringstream values;
        values << std::fixed << std::setprecision(1)
               << state.edgePoints.size() << " points  cx=" << state.fit.sphere.cx
               << " cy=" << state.fit.sphere.cy << " r=" << state.fit.sphere.radius
               << "  RMS=" << state.fit.rmsResidualPixels << " px  coverage="
               << state.fit.angularCoverageDegrees << " deg";
        cv::putText(frame, values.str(), cv::Point(12, frame.rows - 40), cv::FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv::LINE_AA);
        std::string message = circleFitsImage(state)
            ? state.fit.message
            : "The complete sphere must remain inside the image; it does not need to be centered.";
        cv::putText(frame, message, cv::Point(12, frame.rows - 16), cv::FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv::LINE_AA);
    } else {
        const std::string values =
            std::to_string(state.edgePoints.size()) + "/5 minimum edge points   zoom=" +
            std::to_string(static_cast<int>(std::lround(state.zoom * 100.0))) + "%   " + state.fit.message;
        cv::putText(frame, values, cv::Point(12, frame.rows - 16), cv::FONT_HERSHEY_SIMPLEX, 0.58, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
    }

    return frame;
}

} // namespace

SphereSelection chooseSphereInteractive(
    size_t imageCount,
    const SphereImageLoader& loader,
    const std::vector<std::string>& imageLabels,
    size_t initialImageIndex) {
    if (imageCount == 0 || !loader) throw std::runtime_error("No images are available for sphere selection.");
    if (initialImageIndex >= imageCount) initialImageIndex = 0;
    cv::Mat initialImage = loader(initialImageIndex);
    if (initialImage.empty()) throw std::runtime_error("Cannot choose a sphere from an empty image.");
    PickerState state;
    const double maxW = 1400.0;
    const double maxH = 900.0;
    state.image = std::move(initialImage);
    state.imageSize = state.image.size();
    state.imageIndex = initialImageIndex;
    state.imageCount = imageCount;
    state.imageLabels = imageLabels;
    state.minZoom = std::min(1.0, std::min(maxW / state.image.cols, maxH / state.image.rows));
    state.zoom = state.minZoom;
    state.viewSize = cv::Size(
        std::max(1, static_cast<int>(std::lround(state.image.cols * state.minZoom))),
        std::max(1, static_cast<int>(std::lround(state.image.rows * state.minZoom))));
    state.maxZoom = std::max(32.0, state.minZoom * 32.0);

    const std::string windowName = "Select highlight sphere";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(windowName, mouseCallback, &state);

    while (!state.done && !state.canceled) {
        cv::imshow(windowName, render(state));
        const int key = cv::waitKeyEx(20);
        const int ascii = key & 0xff;
        if (ascii == 27 || ascii == 'q' || ascii == 'Q') {
            state.canceled = true;
        } else if (ascii == 'r' || ascii == 'R') {
            resetSelection(state);
        } else if (ascii == 8 && !state.edgePoints.empty()) {
            state.edgePoints.pop_back();
            state.fit = fitSphereCircle(state.edgePoints);
        } else if (key == 2162688 || key == 65365 || ascii == '[') {
            selectImage(state, loader, (state.imageIndex + imageCount - 1) % imageCount);
        } else if (key == 2228224 || key == 65366 || ascii == ']') {
            selectImage(state, loader, (state.imageIndex + 1) % imageCount);
        } else if (ascii == '0') {
            resetView(state);
        } else if (ascii == '+' || ascii == '=') {
            zoomAt(state, 1.25, cv::Point(state.viewSize.width / 2, state.viewSize.height / 2));
        } else if (ascii == '-' || ascii == '_') {
            zoomAt(state, 0.80, cv::Point(state.viewSize.width / 2, state.viewSize.height / 2));
        } else if (ascii == 'a' || ascii == 'A' || key == 2424832) {
            panBy(state, -static_cast<double>(state.viewSize.width) * 0.15 / state.zoom, 0.0);
        } else if (ascii == 'd' || ascii == 'D' || key == 2555904) {
            panBy(state, static_cast<double>(state.viewSize.width) * 0.15 / state.zoom, 0.0);
        } else if (ascii == 'w' || ascii == 'W' || key == 2490368) {
            panBy(state, 0.0, -static_cast<double>(state.viewSize.height) * 0.15 / state.zoom);
        } else if (ascii == 's' || ascii == 'S' || key == 2621440) {
            panBy(state, 0.0, static_cast<double>(state.viewSize.height) * 0.15 / state.zoom);
        } else if ((ascii == 13 || ascii == 10 || ascii == 32) && selectionAccepted(state)) {
            state.done = true;
        }
    }

    cv::destroyWindow(windowName);
    if (state.canceled) {
        throw std::runtime_error("Interactive sphere selection was canceled.");
    }

    SphereSelection selection;
    selection.sphere = state.fit.sphere;
    selection.imageIndex = state.imageIndex;
    selection.edgePoints = state.edgePoints;
    selection.fitRmsPixels = state.fit.rmsResidualPixels;
    selection.fitMaxResidualPixels = state.fit.maxResidualPixels;
    selection.fitCoverageDegrees = state.fit.angularCoverageDegrees;
    return selection;
}

Sphere chooseSphereInteractive(const cv::Mat& displayImage) {
    const SphereSelection selection = chooseSphereInteractive(
        1,
        [&](size_t) { return displayImage.clone(); },
        {"image"},
        0);
    return selection.sphere;
}

#endif
