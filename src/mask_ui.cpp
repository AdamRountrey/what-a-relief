#include "mask_ui.hpp"

#ifdef PS_NO_GUI

#include <stdexcept>

cv::Mat chooseHeightMaskInteractive(const cv::Mat&) {
    throw std::runtime_error("Interactive specimen mask selection is disabled in this build. Use --height-mask or rebuild with OpenCV highgui.");
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

struct MaskState {
    cv::Mat image;
    cv::Size viewSize;
    double zoom = 1.0;
    double minZoom = 1.0;
    double maxZoom = 32.0;
    cv::Point2d origin = cv::Point2d(0.0, 0.0);
    std::vector<cv::Point2d> points;
    bool panning = false;
    bool reviewMode = false;
    cv::Point panStartMouse;
    cv::Point2d panStartOrigin = cv::Point2d(0.0, 0.0);
    bool done = false;
    bool canceled = false;
};

enum class PreviewChoice {
    AcceptRefined,
    AcceptRough,
    Reject,
    Canceled
};

cv::Mat render(const MaskState& state);

cv::Point2d toImagePoint(const MaskState& state, int x, int y) {
    const cv::Point2d p = state.origin + cv::Point2d(static_cast<double>(x) / state.zoom, static_cast<double>(y) / state.zoom);
    return cv::Point2d(
        std::clamp(p.x, 0.0, static_cast<double>(state.image.cols - 1)),
        std::clamp(p.y, 0.0, static_cast<double>(state.image.rows - 1)));
}

cv::Point toViewPoint(const MaskState& state, const cv::Point2d& p) {
    return cv::Point(
        static_cast<int>(std::lround((p.x - state.origin.x) * state.zoom)),
        static_cast<int>(std::lround((p.y - state.origin.y) * state.zoom)));
}

void clampOrigin(MaskState& state) {
    const double visibleW = static_cast<double>(state.viewSize.width) / state.zoom;
    const double visibleH = static_cast<double>(state.viewSize.height) / state.zoom;
    const double maxX = std::max(0.0, static_cast<double>(state.image.cols) - visibleW);
    const double maxY = std::max(0.0, static_cast<double>(state.image.rows) - visibleH);
    state.origin.x = std::clamp(state.origin.x, 0.0, maxX);
    state.origin.y = std::clamp(state.origin.y, 0.0, maxY);
}

void resetView(MaskState& state) {
    state.zoom = state.minZoom;
    state.origin = cv::Point2d(0.0, 0.0);
    clampOrigin(state);
}

void zoomAt(MaskState& state, double factor, const cv::Point& anchor) {
    const cv::Point2d before = toImagePoint(state, anchor.x, anchor.y);
    state.zoom = std::clamp(state.zoom * factor, state.minZoom, state.maxZoom);
    state.origin = before - cv::Point2d(static_cast<double>(anchor.x) / state.zoom, static_cast<double>(anchor.y) / state.zoom);
    clampOrigin(state);
}

void panBy(MaskState& state, double dx, double dy) {
    state.origin += cv::Point2d(dx, dy);
    clampOrigin(state);
}

std::vector<cv::Point> roundedPoints(const std::vector<cv::Point2d>& points) {
    std::vector<cv::Point> rounded;
    rounded.reserve(points.size());
    for (const cv::Point2d& p : points) {
        rounded.emplace_back(static_cast<int>(std::lround(p.x)), static_cast<int>(std::lround(p.y)));
    }
    return rounded;
}

cv::Mat polygonMask(const cv::Size& size, const std::vector<cv::Point2d>& points) {
    cv::Mat mask(size, CV_8U, cv::Scalar(0));
    if (points.size() < 3) {
        return mask;
    }
    const std::vector<cv::Point> poly = roundedPoints(points);
    cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{poly}, cv::Scalar(255), cv::LINE_AA);
    cv::threshold(mask, mask, 0, 255, cv::THRESH_BINARY);
    return mask;
}

cv::Mat keepLargestComponent(const cv::Mat& mask) {
    cv::Mat binary;
    cv::threshold(mask, binary, 0, 255, cv::THRESH_BINARY);
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int count = cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8, CV_32S);
    if (count <= 2) {
        return binary;
    }

    int bestLabel = 1;
    int bestArea = stats.at<int>(1, cv::CC_STAT_AREA);
    for (int label = 2; label < count; ++label) {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area > bestArea) {
            bestArea = area;
            bestLabel = label;
        }
    }

    cv::Mat largest(mask.size(), CV_8U, cv::Scalar(0));
    largest.setTo(cv::Scalar(255), labels == bestLabel);
    return largest;
}

cv::Mat renderRefinementProgress(const MaskState& state, const std::string& message, double fraction) {
    cv::Mat frame = render(state);
    fraction = std::clamp(fraction, 0.0, 1.0);
    cv::rectangle(frame, cv::Point(0, 0), cv::Point(frame.cols, 92), cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(frame, message, cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX, 0.62, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    cv::putText(frame, "Finding the nearest strong image boundary. This can take a moment on large TIFFs.",
        cv::Point(12, 56), cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
    const int barX = 12;
    const int barY = 70;
    const int barW = std::max(20, frame.cols - 24);
    const int barH = 12;
    cv::rectangle(frame, cv::Rect(barX, barY, barW, barH), cv::Scalar(70, 70, 70), 1, cv::LINE_AA);
    cv::rectangle(frame, cv::Rect(barX + 1, barY + 1, std::max(1, static_cast<int>((barW - 2) * fraction)), barH - 2),
        cv::Scalar(0, 180, 90), cv::FILLED);
    return frame;
}

void showRefinementProgress(
    const std::string& windowName,
    const MaskState& state,
    const std::string& message,
    double fraction) {
    cv::imshow(windowName, renderRefinementProgress(state, message, fraction));
    cv::waitKey(1);
}

cv::Mat refineMaskWithGrabCut(
    const std::string& windowName,
    const MaskState& state,
    const cv::Mat& image,
    const cv::Mat& roughMask) {
    if (cv::countNonZero(roughMask) < 100) {
        return roughMask;
    }

    showRefinementProgress(windowName, state, "Preparing edge-aware refinement...", 0.05);

    cv::Mat bgr;
    if (image.channels() == 1) {
        cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, bgr, cv::COLOR_BGRA2BGR);
    } else {
        bgr = image;
    }

    const int minDim = std::max(1, std::min(image.cols, image.rows));
    const int sureForegroundRadius = std::clamp(minDim / 55, 8, 120);
    const int likelyForegroundRadius = std::clamp(minDim / 150, 3, 45);
    const int searchRadius = std::clamp(minDim / 22, 24, 180);
    cv::Mat sureForeground;
    cv::Mat likelyForeground;
    cv::Mat possibleArea;
    cv::erode(roughMask, sureForeground, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(sureForegroundRadius * 2 + 1, sureForegroundRadius * 2 + 1)));
    cv::erode(roughMask, likelyForeground, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(likelyForegroundRadius * 2 + 1, likelyForegroundRadius * 2 + 1)));
    cv::dilate(roughMask, possibleArea, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(searchRadius * 2 + 1, searchRadius * 2 + 1)));
    if (cv::countNonZero(sureForeground) < 25) {
        sureForeground = likelyForeground.clone();
    }
    if (cv::countNonZero(likelyForeground) < 100) {
        likelyForeground = roughMask.clone();
    }

    cv::Mat grabMask(image.size(), CV_8U, cv::Scalar(cv::GC_BGD));
    grabMask.setTo(cv::Scalar(cv::GC_PR_BGD), possibleArea);
    grabMask.setTo(cv::Scalar(cv::GC_PR_FGD), likelyForeground);
    grabMask.setTo(cv::Scalar(cv::GC_FGD), sureForeground);

    cv::Mat bgdModel;
    cv::Mat fgdModel;
    try {
        showRefinementProgress(windowName, state, "Refining boundary: pass 1 of 3...", 0.25);
        cv::grabCut(bgr, grabMask, cv::Rect(), bgdModel, fgdModel, 1, cv::GC_INIT_WITH_MASK);
        showRefinementProgress(windowName, state, "Refining boundary: pass 2 of 3...", 0.55);
        cv::grabCut(bgr, grabMask, cv::Rect(), bgdModel, fgdModel, 1, cv::GC_EVAL);
        showRefinementProgress(windowName, state, "Refining boundary: pass 3 of 3...", 0.80);
        cv::grabCut(bgr, grabMask, cv::Rect(), bgdModel, fgdModel, 1, cv::GC_EVAL);
    } catch (const cv::Exception&) {
        return roughMask;
    }

    cv::Mat refined = (grabMask == cv::GC_FGD) | (grabMask == cv::GC_PR_FGD);
    cv::bitwise_and(refined, possibleArea, refined);
    cv::morphologyEx(refined, refined, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
    cv::morphologyEx(refined, refined, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
    cv::threshold(refined, refined, 0, 255, cv::THRESH_BINARY);
    refined = keepLargestComponent(refined);
    showRefinementProgress(windowName, state, "Refinement complete.", 1.0);
    return cv::countNonZero(refined) >= 100 ? refined : roughMask;
}

void mouseCallback(int event, int x, int y, int flags, void* userdata) {
    auto* state = static_cast<MaskState*>(userdata);
    if (event == cv::EVENT_LBUTTONDOWN && !state->reviewMode) {
        state->points.push_back(toImagePoint(*state, x, y));
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

cv::Mat render(const MaskState& state) {
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

    const std::string line1 = "Click around specimen boundary. Wheel/+/- zoom. Right-drag or WASD pans.";
    const std::string line2 = "Enter/Space accepts and refines to edges. U/Backspace undo. R redraws. Esc cancels.";
    cv::rectangle(frame, cv::Point(0, 0), cv::Point(frame.cols, 64), cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(frame, line1, cv::Point(12, 24), cv::FONT_HERSHEY_SIMPLEX, 0.54, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    cv::putText(frame, line2, cv::Point(12, 50), cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);

    std::vector<cv::Point> viewPoints;
    viewPoints.reserve(state.points.size());
    for (const cv::Point2d& p : state.points) {
        viewPoints.push_back(toViewPoint(state, p));
    }
    if (viewPoints.size() >= 2) {
        cv::polylines(frame, std::vector<std::vector<cv::Point>>{viewPoints}, false, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
    }
    if (viewPoints.size() >= 3) {
        std::vector<cv::Point> closed = viewPoints;
        closed.push_back(viewPoints.front());
        cv::polylines(frame, std::vector<std::vector<cv::Point>>{closed}, false, cv::Scalar(0, 190, 255), 1, cv::LINE_AA);
    }
    for (size_t i = 0; i < viewPoints.size(); ++i) {
        cv::circle(frame, viewPoints[i], 4, cv::Scalar(0, 255, 255), cv::FILLED, cv::LINE_AA);
        cv::putText(frame, std::to_string(i + 1), viewPoints[i] + cv::Point(6, -6), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
    }

    const std::string values =
        std::to_string(state.points.size()) + " points   zoom=" +
        std::to_string(static_cast<int>(std::lround(state.zoom * 100.0))) + "%";
    cv::putText(frame, values, cv::Point(12, frame.rows - 16), cv::FONT_HERSHEY_SIMPLEX, 0.58, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
    return frame;
}

cv::Mat renderPreview(const MaskState& state, const cv::Mat& roughMask, const cv::Mat& refinedMask) {
    cv::Mat frame(state.viewSize, state.image.type(), cv::Scalar(0, 0, 0));
    const int x0 = std::clamp(static_cast<int>(std::floor(state.origin.x)), 0, state.image.cols - 1);
    const int y0 = std::clamp(static_cast<int>(std::floor(state.origin.y)), 0, state.image.rows - 1);
    const int x1 = std::clamp(static_cast<int>(std::ceil(state.origin.x + static_cast<double>(state.viewSize.width) / state.zoom)) + 1, x0 + 1, state.image.cols);
    const int y1 = std::clamp(static_cast<int>(std::ceil(state.origin.y + static_cast<double>(state.viewSize.height) / state.zoom)) + 1, y0 + 1, state.image.rows);
    const cv::Rect sourceRect(x0, y0, x1 - x0, y1 - y0);

    cv::Mat scaledImage;
    cv::resize(state.image(sourceRect), scaledImage, cv::Size(), state.zoom, state.zoom, state.zoom >= 1.0 ? cv::INTER_LINEAR : cv::INTER_AREA);

    const int dstX = static_cast<int>(std::lround((static_cast<double>(x0) - state.origin.x) * state.zoom));
    const int dstY = static_cast<int>(std::lround((static_cast<double>(y0) - state.origin.y) * state.zoom));
    const cv::Rect dstRect(
        std::max(0, dstX),
        std::max(0, dstY),
        std::min(scaledImage.cols + std::min(0, dstX), frame.cols - std::max(0, dstX)),
        std::min(scaledImage.rows + std::min(0, dstY), frame.rows - std::max(0, dstY)));
    if (dstRect.width > 0 && dstRect.height > 0) {
        const cv::Rect srcRect(std::max(0, -dstX), std::max(0, -dstY), dstRect.width, dstRect.height);
        scaledImage(srcRect).copyTo(frame(dstRect));
    }

    cv::Mat roughView;
    cv::Mat refinedView;
    cv::resize(roughMask(sourceRect), roughView, cv::Size(), state.zoom, state.zoom, cv::INTER_NEAREST);
    cv::resize(refinedMask(sourceRect), refinedView, cv::Size(), state.zoom, state.zoom, cv::INTER_NEAREST);
    if (dstRect.width > 0 && dstRect.height > 0) {
        const cv::Rect srcRect(std::max(0, -dstX), std::max(0, -dstY), dstRect.width, dstRect.height);
        cv::Mat roughOnFrame(frame.size(), CV_8U, cv::Scalar(0));
        cv::Mat refinedOnFrame(frame.size(), CV_8U, cv::Scalar(0));
        roughView(srcRect).copyTo(roughOnFrame(dstRect));
        refinedView(srcRect).copyTo(refinedOnFrame(dstRect));

        cv::Mat overlay = frame.clone();
        overlay.setTo(cv::Scalar(30, 170, 80), refinedOnFrame);
        cv::addWeighted(overlay, 0.22, frame, 0.78, 0.0, frame);

        std::vector<std::vector<cv::Point>> roughContours;
        std::vector<std::vector<cv::Point>> refinedContours;
        cv::findContours(roughOnFrame, roughContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::findContours(refinedOnFrame, refinedContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::drawContours(frame, roughContours, -1, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
        cv::drawContours(frame, refinedContours, -1, cv::Scalar(0, 255, 80), 2, cv::LINE_AA);
    }

    const std::string line1 = "Review refined specimen boundary: green = refined, yellow = your outline.";
    const std::string line2 = "Enter/Space accepts green. B uses your outline. R returns to editing. Esc cancels.";
    cv::rectangle(frame, cv::Point(0, 0), cv::Point(frame.cols, 64), cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(frame, line1, cv::Point(12, 24), cv::FONT_HERSHEY_SIMPLEX, 0.54, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    cv::putText(frame, line2, cv::Point(12, 50), cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);

    const std::string values =
        "rough=" + std::to_string(cv::countNonZero(roughMask)) +
        " refined=" + std::to_string(cv::countNonZero(refinedMask)) +
        " pixels   zoom=" + std::to_string(static_cast<int>(std::lround(state.zoom * 100.0))) + "%";
    cv::putText(frame, values, cv::Point(12, frame.rows - 16), cv::FONT_HERSHEY_SIMPLEX, 0.58, cv::Scalar(0, 255, 80), 1, cv::LINE_AA);
    return frame;
}

PreviewChoice reviewRefinedMask(
    const std::string& windowName,
    MaskState& state,
    const cv::Mat& roughMask,
    const cv::Mat& refinedMask) {
    state.reviewMode = true;
    for (;;) {
        cv::imshow(windowName, renderPreview(state, roughMask, refinedMask));
        const int key = cv::waitKeyEx(20);
        const int ascii = key & 0xff;
        if (ascii == 27 || ascii == 'q' || ascii == 'Q') {
            state.reviewMode = false;
            return PreviewChoice::Canceled;
        }
        if (ascii == 13 || ascii == 10 || ascii == 32) {
            state.reviewMode = false;
            return PreviewChoice::AcceptRefined;
        }
        if (ascii == 'b' || ascii == 'B') {
            state.reviewMode = false;
            return PreviewChoice::AcceptRough;
        }
        if (ascii == 'r' || ascii == 'R') {
            state.reviewMode = false;
            return PreviewChoice::Reject;
        }
        if (ascii == '0') {
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
        }
    }
}

} // namespace

cv::Mat chooseHeightMaskInteractive(const cv::Mat& displayImage) {
    if (displayImage.empty()) {
        throw std::runtime_error("Cannot choose a specimen mask from an empty image.");
    }

    MaskState state;
    const double maxW = 1400.0;
    const double maxH = 900.0;
    state.image = displayImage.clone();
    state.minZoom = std::min(1.0, std::min(maxW / displayImage.cols, maxH / displayImage.rows));
    state.zoom = state.minZoom;
    state.viewSize = cv::Size(
        std::max(1, static_cast<int>(std::lround(displayImage.cols * state.minZoom))),
        std::max(1, static_cast<int>(std::lround(displayImage.rows * state.minZoom))));
    state.maxZoom = std::max(32.0, state.minZoom * 32.0);

    const std::string windowName = "Select specimen mask for height and PLY";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(windowName, mouseCallback, &state);

    for (;;) {
        state.done = false;
        state.canceled = false;
        while (!state.done && !state.canceled) {
        cv::imshow(windowName, render(state));
        const int key = cv::waitKeyEx(20);
        const int ascii = key & 0xff;
        if (ascii == 27 || ascii == 'q' || ascii == 'Q') {
            state.canceled = true;
        } else if (ascii == 'r' || ascii == 'R') {
            state.points.clear();
        } else if (ascii == 'u' || ascii == 'U' || ascii == 8 || ascii == 127) {
            if (!state.points.empty()) {
                state.points.pop_back();
            }
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
        } else if ((ascii == 13 || ascii == 10 || ascii == 32) && state.points.size() >= 3) {
            state.done = true;
        }
        }

        if (state.canceled) {
            cv::destroyWindow(windowName);
            throw std::runtime_error("Interactive specimen mask selection was canceled.");
        }

        const cv::Mat roughMask = polygonMask(displayImage.size(), state.points);
        const cv::Mat refinedMask = refineMaskWithGrabCut(windowName, state, displayImage, roughMask);
        const PreviewChoice choice = reviewRefinedMask(windowName, state, roughMask, refinedMask);
        if (choice == PreviewChoice::AcceptRefined) {
            cv::destroyWindow(windowName);
            return refinedMask;
        }
        if (choice == PreviewChoice::AcceptRough) {
            cv::destroyWindow(windowName);
            return roughMask;
        }
        if (choice == PreviewChoice::Canceled) {
            cv::destroyWindow(windowName);
            throw std::runtime_error("Interactive specimen mask selection was canceled.");
        }
    }
}

#endif
