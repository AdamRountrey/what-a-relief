#pragma once

#include "types.hpp"

#include <string>

void launchGuiWorkflow(Options& opt);
void showGuiInfo(const std::string& title, const std::string& text);
bool askGuiYesNo(const std::string& title, const std::string& text, bool defaultYes);
void showGuiProgress(const std::string& title, const std::string& text);
void updateGuiProgress(const std::string& text, int percent);
void closeGuiProgress();
