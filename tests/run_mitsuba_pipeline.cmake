foreach(required PYTHON_EXECUTABLE APP_EXECUTABLE SOURCE_DIRECTORY OUTPUT_DIRECTORY)
    if(NOT DEFINED ${required})
        message(FATAL_ERROR "${required} was not provided")
    endif()
endforeach()
if(WIN32 AND IS_DIRECTORY "${RUNTIME_DIRECTORY}")
    set(ENV{PATH} "${RUNTIME_DIRECTORY};$ENV{PATH}")
endif()
execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -B "${SOURCE_DIRECTORY}/tests/test_mitsuba_pipeline.py"
        --app "${APP_EXECUTABLE}" --worker "${SOURCE_DIRECTORY}/tools/mitsuba_backend/worker.py"
        --output "${OUTPUT_DIRECTORY}"
    RESULT_VARIABLE result)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Independent inverse-improvement acceptance test failed: ${result}")
endif()
