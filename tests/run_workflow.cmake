foreach(required APP_EXECUTABLE FIXTURE_GENERATOR FIXTURE_DIRECTORY)
    if(NOT DEFINED ${required})
        message(FATAL_ERROR "${required} was not provided")
    endif()
endforeach()

if(DEFINED RUNTIME_DIRECTORY)
    set(ENV{PATH} "${RUNTIME_DIRECTORY};$ENV{PATH}")
endif()

execute_process(
    COMMAND "${FIXTURE_GENERATOR}" "${FIXTURE_DIRECTORY}"
    RESULT_VARIABLE fixture_result)
if(NOT fixture_result EQUAL 0)
    message(FATAL_ERROR "Synthetic workflow fixture generation failed with exit code ${fixture_result}")
endif()

set(output_dir "${FIXTURE_DIRECTORY}/output")
set(rti_dir "${output_dir}/rti")
set(mesh_path "${output_dir}/surface.ply")
set(printable_path "${output_dir}/printable_surface.ply")
set(app_arguments)
foreach(index RANGE 0 7)
    list(APPEND app_arguments --image "${FIXTURE_DIRECTORY}/inputs/image_${index}.png")
endforeach()
list(APPEND app_arguments
    --out "${output_dir}"
    --no-gui
    --solver robust
    --height-solver fast
    --height-flatten none
    --pixel-scale-mm 0.1
    --mesh "${mesh_path}"
    --printable-mesh "${printable_path}"
    --printable-fill-holes)
if(UNCALIBRATED)
    list(APPEND app_arguments --uncalibrated)
else()
    list(APPEND app_arguments
        --lights-file "${FIXTURE_DIRECTORY}/light_vectors.csv"
        --rti "${rti_dir}"
        --rti-layout image
        --rti-color rgb)
endif()
if(ENABLE_NEURAL)
    list(APPEND app_arguments --neural-fusion --neural-max-side 64)
endif()
if(NOT UNCALIBRATED AND NOT ENABLE_NEURAL)
    list(APPEND app_arguments --specular-diagnostics --shadow-height-refinement)
endif()

execute_process(
    COMMAND "${APP_EXECUTABLE}" ${app_arguments}
    RESULT_VARIABLE app_result
    OUTPUT_VARIABLE app_stdout
    ERROR_VARIABLE app_stderr)
if(NOT app_result EQUAL 0)
    message(FATAL_ERROR
        "End-to-end workflow failed with exit code ${app_result}\nstdout:\n${app_stdout}\nstderr:\n${app_stderr}")
endif()

if(NOT UNCALIBRATED AND NOT ENABLE_NEURAL)
    file(SHA256 "${output_dir}/normal_rgb.png" first_normal_hash)
    set(repeat_arguments ${app_arguments})
    list(FIND repeat_arguments --lights-file calibration_index)
    math(EXPR calibration_index "${calibration_index} + 1")
    list(REMOVE_AT repeat_arguments ${calibration_index})
    list(INSERT repeat_arguments ${calibration_index} "${output_dir}/lights.csv")
    execute_process(
        COMMAND "${APP_EXECUTABLE}" ${repeat_arguments}
        RESULT_VARIABLE repeat_result
        OUTPUT_VARIABLE repeat_stdout
        ERROR_VARIABLE repeat_stderr)
    if(NOT repeat_result EQUAL 0)
        message(FATAL_ERROR "Same-folder calibration reuse failed:\n${repeat_stdout}\n${repeat_stderr}")
    endif()
    file(SHA256 "${output_dir}/normal_rgb.png" repeat_normal_hash)
    if(NOT first_normal_hash STREQUAL repeat_normal_hash)
        message(FATAL_ERROR "Reusing this run's lights.csv changed the normal output")
    endif()
endif()

set(required_outputs
    "${output_dir}/run_manifest.json"
    "${output_dir}/normal_rgb.png"
    "${output_dir}/albedo.png"
    "${output_dir}/height.pfm"
    "${output_dir}/printable_fill_mask.png"
    "${mesh_path}"
    "${printable_path}")
if(NOT UNCALIBRATED)
    list(APPEND required_outputs
        "${output_dir}/robust_weight.png"
        "${output_dir}/robust_unsupported_mask.png"
        "${output_dir}/robust_inlier_count.png"
        "${output_dir}/robust_local_condition.png"
        "${output_dir}/shadow_count.png"
        "${output_dir}/highlight_outlier_count.png"
        "${output_dir}/saturation_count.png"
        "${output_dir}/model_mismatch_count.png"
        "${rti_dir}/info.json"
        "${rti_dir}/rti_manifest.json")
endif()
if(NOT UNCALIBRATED AND NOT ENABLE_NEURAL)
    list(APPEND required_outputs
        "${output_dir}/specular_cue_mask.png"
        "${output_dir}/shadow_height_correction.png"
        "${output_dir}/shadow_height_correction.pfm"
        "${output_dir}/shadow_constraint_count.png"
        "${output_dir}/shadow_mismatch_before.png"
        "${output_dir}/shadow_mismatch_after.png"
        "${output_dir}/robust_observations/light_001_shadow.png"
        "${output_dir}/robust_observations/light_001_highlight.png"
        "${output_dir}/robust_observations/light_001_saturation.png")
endif()
if(ENABLE_NEURAL)
    list(APPEND required_outputs
        "${output_dir}/classical_normal_rgb.png"
        "${output_dir}/neural_normal_rgb.png"
        "${output_dir}/fused_normal_rgb.png"
        "${output_dir}/neural_valid_mask.png"
        "${output_dir}/fused_classical_confidence.png")
endif()
foreach(path IN LISTS required_outputs)
    if(NOT EXISTS "${path}")
        message(FATAL_ERROR "End-to-end workflow did not write ${path}")
    endif()
    file(SIZE "${path}" output_size)
    if(output_size EQUAL 0)
        message(FATAL_ERROR "End-to-end workflow wrote an empty file: ${path}")
    endif()
endforeach()

file(READ "${output_dir}/run_manifest.json" manifest)
string(JSON manifest_status GET "${manifest}" status)
if(NOT manifest_status STREQUAL "complete")
    message(FATAL_ERROR "End-to-end workflow manifest was not marked complete")
endif()
if(DEFINED EXPECTED_VERSION)
    string(JSON manifest_version GET "${manifest}" application version)
    if(NOT manifest_version STREQUAL EXPECTED_VERSION)
        message(FATAL_ERROR
            "End-to-end workflow manifest did not report configured version ${EXPECTED_VERSION}")
    endif()
endif()
string(JSON manifest_coverage GET "${manifest}" diagnostics solved_fraction)
if(manifest_coverage LESS 0.98 OR manifest_coverage GREATER 1.0)
    message(FATAL_ERROR "End-to-end workflow solve coverage was ${manifest_coverage}, expected 0.98 to 1.0")
endif()
if(NOT UNCALIBRATED AND NOT ENABLE_NEURAL)
    string(JSON shadow_refinement_type TYPE
        "${manifest}" diagnostics shadow_height_refinement)
    if(NOT shadow_refinement_type STREQUAL "OBJECT")
        message(FATAL_ERROR "Calibrated workflow manifest omitted shadow-height diagnostics")
    endif()
    string(JSON shadow_refinement_parameter GET
        "${manifest}" parameters shadow_height_refinement)
    if(NOT shadow_refinement_parameter)
        message(FATAL_ERROR "Calibrated workflow manifest did not record shadow-height refinement")
    endif()
endif()
if(UNCALIBRATED)
    string(JSON condition_type TYPE "${manifest}" diagnostics lighting_condition_number)
    if(NOT condition_type STREQUAL "NULL")
        message(FATAL_ERROR "Uncalibrated workflow manifest must mark lighting condition as not applicable")
    endif()
endif()

if(UNCALIBRATED)
    message(STATUS "End-to-end uncalibrated workflow completed with checked normal, geometry, and manifest outputs")
elseif(ENABLE_NEURAL)
    message(STATUS "End-to-end neural workflow completed with checked classical, neural, fused, geometry, RTI, and manifest outputs")
else()
    message(STATUS "End-to-end workflow completed with checked normal, height, mesh, RTI, and manifest outputs")
endif()
