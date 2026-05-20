#include "scale_ui.hpp"

#ifdef PS_NO_GUI

#include <stdexcept>

double chooseScaleLineInteractive(const cv::Mat&) {
    throw std::runtime_error("Interactive scale-line selection is disabled in this build. Use --pixel-scale-mm or rebuild with OpenCV highgui.");
}

#else

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct ScaleLineState {
    cv::Mat image;
    cv::Size viewSize;
    double zoom = 1.0;
    double minZoom = 1.0;
    double maxZoom = 32.0;
    cv::Point2d origin = cv::Point2d(0.0, 0.0);
    std::vector<cv::Point2d> points;
    bool panning = false;
    cv::Point panStartMouse;
    cv::Point2d panStartOrigin = cv::Point2d(0.0, 0.0);
    bool leftDown = false;
    cv::Point leftStartMouse;
    bool done = false;
    bool canceled = false;
};

cv::Point2d toImagePoint(const ScaleLineState& state, int x, int y) {
    return state.origin + cv::Point2d(static_cast<double>(x) / state.zoom, static_cast<double>(y) / state.zoom);
}

cv::Point toViewPoint(const ScaleLineState& state, const cv::Point2d& p) {
    return cv::Point(
        static_cast<int>(std::lround((p.x - state.origin.x) * state.zoom)),
        static_cast<int>(std::lround((p.y - state.origin.y) * state.zoom)));
}

void clampOrigin(ScaleLineState& state) {
    const double visibleW = static_cast<double>(state.viewSize.width) / state.zoom;
    const double visibleH = static_cast<double>(state.viewSize.height) / state.zoom;
    const double maxX = std::max(0.0, static_cast<double>(state.image.cols) - visibleW);
    const double maxY = std::max(0.0, static_cast<double>(state.image.rows) - visibleH);
    state.origin.x = std::clamp(state.origin.x, 0.0, maxX);
    state.origin.y = std::clamp(state.origin.y, 0.0, maxY);
}

void resetView(ScaleLineState& state) {
    state.zoom = state.minZoom;
    state.origin = cv::Point2d(0.0, 0.0);
    clampOrigin(state);
}

void zoomAt(ScaleLineState& state, double factor, const cv::Point& anchor) {
    const cv::Point2d before = toImagePoint(state, anchor.x, anchor.y);
    state.zoom = std::clamp(state.zoom * factor, state.minZoom, state.maxZoom);
    state.origin = before - cv::Point2d(static_cast<double>(anchor.x) / state.zoom, static_cast<double>(anchor.y) / state.zoom);
    clampOrigin(state);
}

void panBy(ScaleLineState& state, double dx, double dy) {
    state.origin += cv::Point2d(dx, dy);
    clampOrigin(state);
}

void addPoint(ScaleLineState& state, const cv::Point2d& p) {
    if (state.points.size() >= 2) {
        state.points.clear();
    }
    state.points.push_back(p);
}

void resetSelection(ScaleLineState& state) {
    state.points.clear();
    state.leftDown = false;
    state.panning = false;
}

double pixelDistance(const ScaleLineState& state) {
    if (state.points.size() != 2) {
        return 0.0;
    }
    return cv::norm(state.points[1] - state.points[0]);
}

void mouseCallback(int event, int x, int y, int flags, void* userdata) {
    auto* state = static_cast<ScaleLineState*>(userdata);
    if (event == cv::EVENT_LBUTTONDOWN) {
        state->leftDown = true;
        state->leftStartMouse = cv::Point(x, y);
    } else if (event == cv::EVENT_LBUTTONUP && state->leftDown) {
        const cv::Point mouse(x, y);
        if (cv::norm(mouse - state->leftStartMouse) <= 4.0) {
            addPoint(*state, toImagePoint(*state, x, y));
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

cv::Mat render(const ScaleLineState& state) {
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

    const std::string line1 = "Click two endpoints of a known-length scale line. Wheel/+/- zoom. Right-drag or WASD pans.";
    const std::string line2 = "Enter/Space accepts after 2 points. Backspace removes last. R resets. 0 fits. Esc cancels.";
    cv::rectangle(frame, cv::Point(0, 0), cv::Point(frame.cols, 64), cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(frame, line1, cv::Point(12, 24), cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    cv::putText(frame, line2, cv::Point(12, 50), cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);

    for (size_t i = 0; i < state.points.size(); ++i) {
        const cv::Point p = toViewPoint(state, state.points[i]);
        cv::circle(frame, p, 5, cv::Scalar(0, 255, 255), cv::FILLED, cv::LINE_AA);
        cv::circle(frame, p, 9, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        cv::putText(frame, std::to_string(i + 1), p + cv::Point(9, -8), cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
    }

    if (state.points.size() == 2) {
        const cv::Point p0 = toViewPoint(state, state.points[0]);
        const cv::Point p1 = toViewPoint(state, state.points[1]);
        cv::line(frame, p0, p1, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
    }

    const std::string values =
        std::to_string(state.points.size()) + "/2 points   distance=" +
        std::to_string(static_cast<int>(std::lround(pixelDistance(state)))) + " px   zoom=" +
        std::to_string(static_cast<int>(std::lround(state.zoom * 100.0))) + "%";
    cv::putText(frame, values, cv::Point(12, frame.rows - 16), cv::FONT_HERSHEY_SIMPLEX, 0.58, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);

    return frame;
}

} // namespace

double chooseScaleLineInteractive(const cv::Mat& displayImage) {
    if (displayImage.empty()) {
        throw std::runtime_error("Cannot choose a scale line from an empty image.");
    }

    ScaleLineState state;
    const double maxW = 1400.0;
    const double maxH = 900.0;
    state.image = displayImage.clone();
    state.minZoom = std::min(1.0, std::min(maxW / displayImage.cols, maxH / displayImage.rows));
    state.zoom = state.minZoom;
    state.viewSize = cv::Size(
        std::max(1, static_cast<int>(std::lround(displayImage.cols * state.minZoom))),
        std::max(1, static_cast<int>(std::lround(displayImage.rows * state.minZoom))));
    state.maxZoom = std::max(32.0, state.minZoom * 32.0);

    const std::string windowName = "Select scale line";
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
        } else if (ascii == 8 && !state.points.empty()) {
            state.points.pop_back();
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
        } else if ((ascii == 13 || ascii == 10 || ascii == 32) && pixelDistance(state) > 1.0) {
            state.done = true;
        }
    }

    cv::destroyWindow(windowName);
    if (state.canceled) {
        throw std::runtime_error("Interactive scale-line selection was canceled.");
    }

    const double distance = pixelDistance(state);
    if (distance <= 1.0 || !std::isfinite(distance)) {
        throw std::runtime_error("Scale line must be longer than 1 pixel.");
    }
    return distance;
}

#endif
