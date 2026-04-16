cmake_minimum_required(VERSION 3.16)

project(photometric_stereo_spheres
    VERSION 0.1.0
    DESCRIPTION "Photometric stereo from 4 or 8 images using highlight spheres"
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

add_executable(ps_spheres
    src/args.cpp
    src/gui_workflow.cpp
    src/image_io.cpp
    src/main.cpp
    src/photometric.cpp
    src/sphere_ui.cpp)

target_link_libraries(ps_spheres PRIVATE ${OpenCV_LIBS})
if(WIN32)
    target_link_libraries(ps_spheres PRIVATE comdlg32 shell32 ole32 user32)
endif()
target_compile_features(ps_spheres PRIVATE cxx_std_17)

if(MSVC)
    target_compile_options(ps_spheres PRIVATE /W4 /permissive-)
    target_link_options(ps_spheres PRIVATE /MANIFEST:NO)
else()
    target_compile_options(ps_spheres PRIVATE -Wall -Wextra -Wpedantic)
endif()
