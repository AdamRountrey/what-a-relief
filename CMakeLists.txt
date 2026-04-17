cmake_minimum_required(VERSION 3.16)

project(WhatARelief
    VERSION 0.1.0
    DESCRIPTION "Relief visualization and photometric stereo from directional image sets"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc highgui)

if(DEFINED OpenCV_DIR AND OpenCV_DIR MATCHES "[Aa]naconda|[.]conda")
    message(FATAL_ERROR
        "OpenCV resolved to an Anaconda path: ${OpenCV_DIR}\n"
        "Use the vcpkg preset or set OpenCV_DIR to a standalone OpenCV install.")
endif()

add_executable(what-a-relief
    src/args.cpp
    src/crop_ui.cpp
    src/gui_workflow.cpp
    src/image_io.cpp
    src/main.cpp
    src/photometric.cpp
    src/relight_ui.cpp
    src/sphere_ui.cpp)

target_link_libraries(what-a-relief PRIVATE ${OpenCV_LIBS})
if(WIN32)
    target_link_libraries(what-a-relief PRIVATE comdlg32 shell32 ole32 user32)
endif()
target_compile_features(what-a-relief PRIVATE cxx_std_17)

if(MSVC)
    target_compile_options(what-a-relief PRIVATE /W4 /permissive-)
    target_link_options(what-a-relief PRIVATE /MANIFEST:NO)
else()
    target_compile_options(what-a-relief PRIVATE -Wall -Wextra -Wpedantic)
endif()
