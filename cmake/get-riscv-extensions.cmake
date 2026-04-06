
execute_process(COMMAND ${CMAKE_C_COMPILER} -xc -dM -E ${SKL_COMPILE_OPTIONS} /dev/null
  COMMAND grep riscv
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE SKL_PREPROCESSOR_OUTPUT
)

set(SKL_QEMU_CPU_OPTIONS
  rv64
  v=true
  rvv_vl_half_avl=true
  rvv_ma_all_1s=true
  rvv_ta_all_1s=true
)

if(${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_v_min_vlen)
  string(REGEX REPLACE
      ".*__riscv_v_min_vlen ([0-9]*).*"
      "\\1"
      RISCV_VLEN
      ${SKL_PREPROCESSOR_OUTPUT}
  )
  list(APPEND SKL_QEMU_CPU_OPTIONS "vlen=${RISCV_VLEN}")
endif()

set(VALID_RISCV_EXTENSIONS "")
set(RISCV_EXTENSIONS "")

macro(detect_riscv_extension EXT_NAME EXT_PREPROCESSOR_STR QEMU_CPU_OPTION)
  # Determine whether the given RISC-V extension is present in the current
  # environment by checking EXT_PREPROCESSOR_STR against C preprocessor output
  # and populate QEMU options.

  # *** This is a macro because it modifies lists in the parent scope. ***

  list(APPEND VALID_RISCV_EXTENSIONS "${EXT_NAME}")
  if(${SKL_PREPROCESSOR_OUTPUT} MATCHES "__riscv_${EXT_PREPROCESSOR_STR}")
    list(APPEND RISCV_EXTENSIONS "${EXT_NAME}")
    list(APPEND SKL_QEMU_CPU_OPTIONS "${QEMU_CPU_OPTION}")
    message(STATUS "Extension ${EXT_NAME} is enabled.")
  endif()
endmacro()

function(assert_extension_valid EXT)
  if(NOT ${EXT} IN_LIST VALID_RISCV_EXTENSIONS)
    message(FATAL_ERROR "Provided RISC-V extension name is not valid: " ${EXT})
  endif()
endfunction()


detect_riscv_extension("zfh" "zfh " "zfh=true") # The space is necessary to distinguish from zfhmin
detect_riscv_extension("zvfh" "zvfh" "zvfh=true")
detect_riscv_extension("zvfbfwma" "zvfbfwma" "zvfbfwma=true")
detect_riscv_extension("zvfbfmin" "zvfbfmin" "zvfbfmin=true")
detect_riscv_extension("zve32f" "zve32f" "zve32f=true")
detect_riscv_extension("zve64d" "zve64d" "zve64d=true")
detect_riscv_extension("zve32x" "zve32x" "zve32x=true")
detect_riscv_extension("zvfofp4min" "zvfofp4min" "x-zvfofp4min=true")
detect_riscv_extension("zvfofp8min" "zvfofp8min" "x-zvfofp8min=true")

detect_riscv_extension("xsfmmbase" "xsfmmbase" "x-xsfmmbase=true")
detect_riscv_extension("xsfmm32a8f" "xsfmm32a8f" "x-xsfmm32a8f=true")
detect_riscv_extension("xsfmm32a8i" "xsfmm32a8i" "x-xsfmm32a8i=true")
detect_riscv_extension("xsfmm32a16f" "xsfmm32a16f" "x-xsfmm32a16f=true")
detect_riscv_extension("xsfmm32a32f" "xsfmm32a32f" "x-xsfmm32a32f=true")

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES "__riscv_xsfmm[0-9]+t")
  string(REGEX REPLACE
      ".*__riscv_xsfmm([0-9]+)t.*"
      "\\1"
      XSFMM_TE
      ${SKL_PREPROCESSOR_OUTPUT}
  )
  list(APPEND SKL_QEMU_CPU_OPTIONS "x-xsfmm${XSFMM_TE}t=true")
endif()

detect_riscv_extension("xsfvfbfa" "xsfvfbfa" "x-xsfvfbfa=true")
detect_riscv_extension("xsfvfexpa" "xsfvfexpa" "x-xsfvfexpa=true")
detect_riscv_extension("xsfvfexp16e" "xsfvfexp16e" "x-xsfvfexp16e=true")
detect_riscv_extension("xsfvfbfexp16e" "xsfvfbfexp16e" "x-xsfvfbfexp16e=true")
detect_riscv_extension("xsfvfexp32e" "xsfvfexp32e" "x-xsfvfexp32e=true")
detect_riscv_extension("xsfvqdotq" "xsfvqdotq" "x-xsfvqdotq=true")

string(JOIN "," SKL_QEMU_CPU_OPTIONS ${SKL_QEMU_CPU_OPTIONS})
set(SKL_QEMU_OPTIONS -cpu ${SKL_QEMU_CPU_OPTIONS})
