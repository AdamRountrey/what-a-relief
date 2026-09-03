cmake_minimum_required(VERSION 3.16)

project(WhatARelief
    VERSION 0.2.1
    DESCRIPTION "Relief visualization and photometric stereo from directional image sets"
    LANGUAGES CXX)

set(WHAT_A_RELIEF_VERSION_STRING "${PROJECT_VERSION}" CACHE STRING
    "User-visible application version recorded in run manifests")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core dnn imgcodecs imgproc highgui)

if(DEFINED OpenCV_DIR AND OpenCV_DIR MATCHES "[Aa]naconda|[.]conda")
    message(FATAL_ERROR
        "OpenCV resolved to an Anaconda path: ${OpenCV_DIR}\n"
        "Use the vcpkg preset or set OpenCV_DIR to a standalone OpenCV install.")
endif()

add_library(what-a-relief-core STATIC
    src/checked_io.cpp
    src/image_io.cpp
    src/photometric.cpp
    src/radiometry.cpp
    src/run_manifest.cpp
    src/rti_export.cpp)
target_include_directories(what-a-relief-core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(what-a-relief-core PUBLIC ${OpenCV_LIBS})
target_compile_features(what-a-relief-core PUBLIC cxx_std_17)
target_compile_definitions(what-a-relief-core PUBLIC
    WHAT_A_RELIEF_VERSION="${WHAT_A_RELIEF_VERSION_STRING}")

add_executable(what-a-relief
    src/args.cpp
    src/crop_ui.cpp
    src/gui_workflow.cpp
    src/main.cpp
    src/mask_ui.cpp
    src/neural_fusion.cpp
    src/relight_ui.cpp
    src/scale_ui.cpp
    src/sphere_ui.cpp)

target_link_libraries(what-a-relief PRIVATE what-a-relief-core ${OpenCV_LIBS})
if(WIN32)
    target_link_libraries(what-a-relief PRIVATE comdlg32 shell32 ole32 user32 gdi32 comctl32)
endif()
target_compile_features(what-a-relief PRIVATE cxx_std_17)

include(CTest)
if(BUILD_TESTING)
    add_executable(what-a-relief-tests
        tests/test_photometric.cpp)
    target_link_libraries(what-a-relief-tests PRIVATE what-a-relief-core)
    target_compile_features(what-a-relief-tests PRIVATE cxx_std_17)
    add_executable(what-a-relief-io-tests
        tests/test_io_exports.cpp)
    target_link_libraries(what-a-relief-io-tests PRIVATE what-a-relief-core)
    target_compile_features(what-a-relief-io-tests PRIVATE cxx_std_17)
    add_executable(what-a-relief-fixture-generator
        tests/generate_workflow_fixture.cpp)
    target_link_libraries(what-a-relief-fixture-generator PRIVATE what-a-relief-core)
    target_compile_features(what-a-relief-fixture-generator PRIVATE cxx_std_17)
    add_executable(what-a-relief-benchmark EXCLUDE_FROM_ALL
        tests/benchmark_photometric.cpp)
    target_link_libraries(what-a-relief-benchmark PRIVATE what-a-relief-core)
    target_compile_features(what-a-relief-benchmark PRIVATE cxx_std_17)
    add_executable(what-a-relief-output-benchmark EXCLUDE_FROM_ALL
        tests/benchmark_output.cpp)
    target_link_libraries(what-a-relief-output-benchmark PRIVATE what-a-relief-core)
    target_compile_features(what-a-relief-output-benchmark PRIVATE cxx_std_17)
    if(WIN32 AND DEFINED VCPKG_INSTALLED_DIR)
        add_test(
            NAME photometric-core
            COMMAND ${CMAKE_COMMAND}
                "-DTEST_EXECUTABLE=$<TARGET_FILE:what-a-relief-tests>"
                "-DRUNTIME_DIRECTORY=${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_with_runtime.cmake")
        add_test(
            NAME io-exports
            COMMAND ${CMAKE_COMMAND}
                "-DTEST_EXECUTABLE=$<TARGET_FILE:what-a-relief-io-tests>"
                "-DRUNTIME_DIRECTORY=${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_with_runtime.cmake")
        add_test(
            NAME end-to-end-workflow
            COMMAND ${CMAKE_COMMAND}
                "-DAPP_EXECUTABLE=$<TARGET_FILE:what-a-relief>"
                "-DFIXTURE_GENERATOR=$<TARGET_FILE:what-a-relief-fixture-generator>"
                "-DFIXTURE_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/workflow-fixture"
                "-DEXPECTED_VERSION=${WHAT_A_RELIEF_VERSION_STRING}"
                "-DRUNTIME_DIRECTORY=${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_workflow.cmake")
        add_test(
            NAME end-to-end-neural-workflow
            COMMAND ${CMAKE_COMMAND}
                "-DAPP_EXECUTABLE=$<TARGET_FILE:what-a-relief>"
                "-DFIXTURE_GENERATOR=$<TARGET_FILE:what-a-relief-fixture-generator>"
                "-DFIXTURE_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/workflow-fixture-neural"
                "-DEXPECTED_VERSION=${WHAT_A_RELIEF_VERSION_STRING}"
                "-DRUNTIME_DIRECTORY=${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin"
                "-DENABLE_NEURAL=ON"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_workflow.cmake")
        add_test(
            NAME end-to-end-uncalibrated-workflow
            COMMAND ${CMAKE_COMMAND}
                "-DAPP_EXECUTABLE=$<TARGET_FILE:what-a-relief>"
                "-DFIXTURE_GENERATOR=$<TARGET_FILE:what-a-relief-fixture-generator>"
                "-DFIXTURE_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/workflow-fixture-uncalibrated"
                "-DEXPECTED_VERSION=${WHAT_A_RELIEF_VERSION_STRING}"
                "-DRUNTIME_DIRECTORY=${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin"
                "-DUNCALIBRATED=ON"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_workflow.cmake")
    else()
        add_test(NAME photometric-core COMMAND what-a-relief-tests)
        add_test(NAME io-exports COMMAND what-a-relief-io-tests)
        add_test(
            NAME end-to-end-workflow
            COMMAND ${CMAKE_COMMAND}
                "-DAPP_EXECUTABLE=$<TARGET_FILE:what-a-relief>"
                "-DFIXTURE_GENERATOR=$<TARGET_FILE:what-a-relief-fixture-generator>"
                "-DFIXTURE_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/workflow-fixture"
                "-DEXPECTED_VERSION=${WHAT_A_RELIEF_VERSION_STRING}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_workflow.cmake")
        add_test(
            NAME end-to-end-neural-workflow
            COMMAND ${CMAKE_COMMAND}
                "-DAPP_EXECUTABLE=$<TARGET_FILE:what-a-relief>"
                "-DFIXTURE_GENERATOR=$<TARGET_FILE:what-a-relief-fixture-generator>"
                "-DFIXTURE_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/workflow-fixture-neural"
                "-DEXPECTED_VERSION=${WHAT_A_RELIEF_VERSION_STRING}"
                "-DENABLE_NEURAL=ON"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_workflow.cmake")
        add_test(
            NAME end-to-end-uncalibrated-workflow
            COMMAND ${CMAKE_COMMAND}
                "-DAPP_EXECUTABLE=$<TARGET_FILE:what-a-relief>"
                "-DFIXTURE_GENERATOR=$<TARGET_FILE:what-a-relief-fixture-generator>"
                "-DFIXTURE_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/workflow-fixture-uncalibrated"
                "-DEXPECTED_VERSION=${WHAT_A_RELIEF_VERSION_STRING}"
                "-DUNCALIBRATED=ON"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_workflow.cmake")
    endif()
endif()

add_custom_command(TARGET what-a-relief POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:what-a-relief>/models"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/assets/models"
        "$<TARGET_FILE_DIR:what-a-relief>/models")

if(MSVC)
    target_compile_options(what-a-relief-core PRIVATE /W4 /permissive-)
    target_compile_options(what-a-relief PRIVATE /W4 /permissive-)
    target_link_options(what-a-relief PRIVATE /MANIFEST:NO)
    if(BUILD_TESTING)
        target_compile_options(what-a-relief-tests PRIVATE /W4 /permissive-)
        target_compile_options(what-a-relief-io-tests PRIVATE /W4 /permissive-)
        target_compile_options(what-a-relief-fixture-generator PRIVATE /W4 /permissive-)
        target_compile_options(what-a-relief-benchmark PRIVATE /W4 /permissive-)
        target_compile_options(what-a-relief-output-benchmark PRIVATE /W4 /permissive-)
    endif()
else()
    target_compile_options(what-a-relief-core PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(what-a-relief PRIVATE -Wall -Wextra -Wpedantic)
    if(BUILD_TESTING)
        target_compile_options(what-a-relief-tests PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(what-a-relief-io-tests PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(what-a-relief-fixture-generator PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(what-a-relief-benchmark PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(what-a-relief-output-benchmark PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()
