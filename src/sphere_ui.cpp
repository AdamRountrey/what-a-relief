#include "sphere_ui.hpp"

#ifdef PS_NO_GUI

#include <stdexcept>

Sphere chooseSphereInteractive(const cv::Mat&) {
    throw std::runtime_error("Interactive sphere selection is disabled in this build. Use --sphere or rebuild with OpenCV highgui.");
}

#else

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace {

struct PickerState {
    cv::Mat base;
    double scale = 1.0;
    cv::Point2d center = cv::Point2d(0.0, 0.0);
    double radius = 0.0;
    bool hasCenter = false;
    bool dragging = false;
    bool done = false;
    bool canceled = false;
};

cv::Point2d toImagePoint(const PickerState& state, int x, int y) {
    return cv::Point2d(static_cast<double>(x) / state.scale, static_cast<double>(y) / state.scale);
}

void mouseCallback(int event, int x, int y, int, void* userdata) {
    auto* state = static_cast<PickerState*>(userdata);
    if (event == cv::EVENT_LBUTTONDOWN) {
        state->center = toImagePoint(*state, x, y);
        state->radius = 0.0;
        state->hasCenter = true;
        state->dragging = true;
    } else if (event == cv::EVENT_MOUSEMOVE && state->dragging) {
        const cv::Point2d p = toImagePoint(*state, x, y);
        state->radius = cv::norm(p - state->center);
    } else if (event == cv::EVENT_LBUTTONUP && state->dragging) {
        const cv::Point2d p = toImagePoint(*state, x, y);
        state->radius = cv::norm(p - state->center);
        state->dragging = false;
    }
}

cv::Mat render(const PickerState& state) {
    cv::Mat frame = state.base.clone();
    const std::string line1 = "Click-drag from highlight sphere center to edge.";
    const std::string line2 = "Enter/Space accepts. R resets. Esc cancels.";
    cv::rectangle(frame, cv::Point(0, 0), cv::Point(frame.cols, 58), cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(frame, line1, cv::Point(12, 23), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    cv::putText(frame, line2, cv::Point(12, 48), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);

    if (state.hasCenter) {
        const cv::Point c(
            static_cast<int>(std::lround(state.center.x * state.scale)),
            static_cast<int>(std::lround(state.center.y * state.scale)));
        const int r = static_cast<int>(std::lround(state.radius * state.scale));
        cv::circle(frame, c, 4, cv::Scalar(0, 255, 255), cv::FILLED, cv::LINE_AA);
        if (r > 0) {
            cv::circle(frame, c, r, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
        }

        const std::string values =
            "cx=" + std::to_string(static_cast<int>(std::lround(state.center.x))) +
            " cy=" + std::to_string(static_cast<int>(std::lround(state.center.y))) +
            " r=" + std::to_string(static_cast<int>(std::lround(state.radius)));
        cv::putText(frame, values, cv::Point(12, frame.rows - 16), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
    }

    return frame;
}

} // namespace

Sphere chooseSphereInteractive(const cv::Mat& displayImage) {
    if (displayImage.empty()) {
        throw std::runtime_error("Cannot choose a sphere from an empty image.");
    }

    PickerState state;
    const double maxW = 1400.0;
    const double maxH = 900.0;
    state.scale = std::min(1.0, std::min(maxW / displayImage.cols, maxH / displayImage.rows));
    if (state.scale < 1.0) {
        cv::resize(displayImage, state.base, cv::Size(), state.scale, state.scale, cv::INTER_AREA);
    } else {
        state.base = displayImage.clone();
    }

    const std::string windowName = "Select highlight sphere";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(windowName, mouseCallback, &state);

    while (!state.done && !state.canceled) {
        cv::imshow(windowName, render(state));
        const int key = cv::waitKey(20);
        if (key == 27 || key == 'q' || key == 'Q') {
            state.canceled = true;
        } else if (key == 'r' || key == 'R') {
            state.hasCenter = false;
            state.dragging = false;
            state.radius = 0.0;
        } else if ((key == 13 || key == 10 || key == 32) && state.hasCenter && state.radius > 1.0) {
            state.done = true;
        }
    }

    cv::destroyWindow(windowName);
    if (state.canceled) {
        throw std::runtime_error("Interactive sphere selection was canceled.");
    }

    Sphere sphere;
    sphere.cx = state.center.x;
    sphere.cy = state.center.y;
    sphere.radius = state.radius;
    return sphere;
}

#endif
