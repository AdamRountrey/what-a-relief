#pragma once

#include "types.hpp"
#include "progress.hpp"

#include <string>
#include <functional>

using GuiProgress = std::function<void(const ProgressUpdate&)>;
struct GuiRunResult {
    MitsubaRefinementDiagnostics inverse;
    cv::Mat relightNormals;
    cv::Mat relightMask;
};
using GuiProcessor = std::function<GuiRunResult(Options&, const GuiProgress&)>;

void launchGuiApplication(Options& opt, const GuiProcessor& processor, bool visible = true);
void showGuiInfo(const std::string& title, const std::string& text);
bool askGuiYesNo(const std::string& title, const std::string& text, bool defaultYes);
void openGuiReviewFile(const std::string& path);
bool guiProgressCancellationRequested();
