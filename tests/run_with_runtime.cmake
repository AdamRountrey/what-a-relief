if(NOT DEFINED TEST_EXECUTABLE)
    message(FATAL_ERROR "TEST_EXECUTABLE was not provided")
endif()

if(DEFINED RUNTIME_DIRECTORY)
    set(ENV{PATH} "${RUNTIME_DIRECTORY};$ENV{PATH}")
endif()

execute_process(
    COMMAND "${TEST_EXECUTABLE}"
    RESULT_VARIABLE test_result)

if(NOT test_result EQUAL 0)
    message(FATAL_ERROR "Test executable failed with exit code ${test_result}")
endif()
