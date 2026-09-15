#pragma once

#include "types.hpp"
#include "progress.hpp"

#include <string>
#include <functional>

using GuiProgress = std::function<void(const ProgressUpdate&)>;

struct ShadowRefinementSummary {
    bool attempted = false;
    bool applied = false;
    std::string decision;
    double balancedMismatchBefore = -1.0;
    double balancedMismatchAfter = -1.0;
    double correctionRmsPixels = 0.0;
};

struct LightGainSummary {
    bool attempted = false;
    bool applied = false;
    std::string source;
    std::string decision;
    double validationErrorBefore = -1.0;
    double validationErrorAfter = -1.0;
    double stability = -1.0;
    double normalDiversity = -1.0;
};

struct GuiRunResult {
    MitsubaRefinementDiagnostics inverse;
    ShadowRefinementSummary shadow;
    LightGainSummary lightGain;
    cv::Mat relightNormals;
    cv::Mat relightMask;
};
using GuiProcessor = std::function<GuiRunResult(Options&, const GuiProgress&)>;

std::string formatShadowRefinementSummary(const ShadowRefinementSummary& summary);
std::string formatLightGainSummary(const LightGainSummary& summary);
void launchGuiApplication(Options& opt, const GuiProcessor& processor, bool visible = true);
void showGuiInfo(const std::string& title, const std::string& text);
bool askGuiYesNo(const std::string& title, const std::string& text, bool defaultYes);
void openGuiReviewFile(const std::string& path);
bool guiProgressCancellationRequested();
