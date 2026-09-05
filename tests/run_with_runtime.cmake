if(NOT DEFINED TEST_EXECUTABLE)
    message(FATAL_ERROR "TEST_EXECUTABLE was not provided")
endif()

if(DEFINED RUNTIME_DIRECTORY)
    set(ENV{PATH} "${RUNTIME_DIRECTORY};$ENV{PATH}")
endif()

set(test_arguments)
if(DEFINED TEST_ARGUMENT_1)
    list(APPEND test_arguments "${TEST_ARGUMENT_1}")
endif()
if(DEFINED TEST_ARGUMENT_2)
    list(APPEND test_arguments "${TEST_ARGUMENT_2}")
endif()

execute_process(
    COMMAND "${TEST_EXECUTABLE}" ${test_arguments}
    RESULT_VARIABLE test_result)

if(NOT test_result EQUAL 0)
    message(FATAL_ERROR "Test executable failed with exit code ${test_result}")
endif()
