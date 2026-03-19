set(SKL_TEST_LOG_LEVEL SKL_TEST_LOG_INFO CACHE STRING "Test log level")

add_library(skl-test-driver OBJECT skl-test-driver.c)
target_compile_definitions(
    skl-test-driver
    PRIVATE SKL_TEST_LOG_LEVEL=${SKL_TEST_LOG_LEVEL}
)
target_compile_options(
    skl-test-driver
    PRIVATE ${SKL_COMPILE_OPTIONS} ${SKL_TEST_COMPILE_OPTIONS}
)

function(skl_add_test TEST_NAME HARNESS_SRC TEST_SRC OPTS)
    add_executable(test-${TEST_NAME} ${HARNESS_SRC} ${TEST_SRC})
    target_link_libraries(
        test-${TEST_NAME}
        PRIVATE ${SKL_LIBRARY} ${SKL_TEST_LIBS} skl-test-driver
    )
    target_compile_options(
        test-${TEST_NAME}
        PRIVATE ${SKL_COMPILE_OPTIONS} ${SKL_TEST_COMPILE_OPTIONS} ${OPTS}
    )
    target_compile_definitions(
        test-${TEST_NAME}
        PRIVATE
            SKL_TEST_NAME=${TEST_NAME}
            SKL_TEST_LOG_LEVEL=${SKL_TEST_LOG_LEVEL}
    )
    target_include_directories(test-${TEST_NAME} PRIVATE . ../src)
    add_test(
        NAME test-${TEST_NAME}
        COMMAND
            ${SKL_TESTER} ${SKL_TESTER_OPTIONS} $<TARGET_FILE:test-${TEST_NAME}>
    )
endfunction()

if(RISCV_ZVE32F)
    skl_add_test(
        skl_gemm_f32_f32_f32_zve32f_x390
        gemm/gemm_f32rc_f32rc_f32rc.c
        gemm/rvv/skl_gemm_f32_f32_f32_zve32f_x390.c
        ""
    )
endif()
if(RISCV_XSFMM32A32F)
    skl_add_test(
        skl_gemm_a1b01_f32c_f32_f32_xsfmm32a32f
        gemm/gemm_f32rc_f32rc_f32rc.c
        gemm/xsfmm/skl_gemm_a1b01_f32c_f32_f32_xsfmm32a32f.c
        ""
    )
endif()
