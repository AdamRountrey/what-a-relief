#pragma once

#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <string>

struct ProgressUpdate {
    std::string message;
    int percent = 0;
    std::string stage;
    double completed = 0;
    double total = 0;
    double remainingSeconds = -1;
    std::string estimateScope;
    cv::Mat preview;
    int previewIteration = 0;
    int previewTotal = 0;
};

// Estimate only measured work in the current stage, never unmeasured later stages.
class ProgressTiming {
public:
    void update(const ProgressUpdate& event, double seconds) {
        if (event.stage == stage_ && event.completed == completed_ &&
            event.remainingSeconds >= 0 && event.remainingSeconds == reportedRemaining_) return;
        if (event.stage != stage_ || event.completed < completed_) {
            stage_ = event.stage;
            began_ = seconds;
            remaining_ = -1;
        }
        completed_ = event.completed;
        reportedRemaining_ = event.remainingSeconds;
        sampled_ = seconds;
        if (std::isfinite(event.remainingSeconds) && event.remainingSeconds >= 0) {
            remaining_ = event.remainingSeconds;
        } else if (event.total > 0 && completed_ > 0 && completed_ <= event.total && seconds - began_ >= 1) {
            remaining_ = (seconds - began_) * (event.total - completed_) / completed_;
        } else {
            remaining_ = -1;
        }
    }
    double remaining(double seconds) const {
        if (remaining_ < 0) return -1;
        // An overdue estimate is unknown, not a misleading permanent zero.
        const double value = remaining_ - std::max(0.0, seconds - sampled_);
        return value < -2 ? -1 : std::max(0.0, value);
    }
private:
    std::string stage_;
    double began_ = 0, sampled_ = 0, completed_ = 0, remaining_ = -1, reportedRemaining_ = -1;
};
