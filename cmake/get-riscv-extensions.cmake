
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

set(RISCV_EXTENSIONS "")

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES "__riscv_zfh ")
  # The above space is necessary to distinguish from zfhmin
  list(APPEND SKL_QEMU_CPU_OPTIONS "zfh=true")
  list(APPEND RISCV_EXTENSIONS "ZFH")
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES "__riscv_zvfh ")
  list(APPEND SKL_QEMU_CPU_OPTIONS "zvfh=true")
  list(APPEND RISCV_EXTENSIONS "ZVFH")
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES "__riscv_zvfbfwma")
  list(APPEND SKL_QEMU_CPU_OPTIONS "zvfbfwma=true")
  list(APPEND RISCV_EXTENSIONS "ZVFBFWMA")
endif()

if(${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_zvfbfmin)
  list(APPEND SKL_QEMU_CPU_OPTIONS "zvfbfmin=true")
  list(APPEND RISCV_EXTENSIONS "ZVFBFMIN")
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_zve32f)
  list(APPEND SKL_QEMU_CPU_OPTIONS "zve32f=true")
  list(APPEND RISCV_EXTENSIONS "ZVE32F")
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_zve64d)
  list(APPEND SKL_QEMU_CPU_OPTIONS "zve64d=true")
  list(APPEND RISCV_EXTENSIONS "ZVE64D")
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_zve32x)
  list(APPEND SKL_QEMU_CPU_OPTIONS "zve32x=true")
  list(APPEND RISCV_EXTENSIONS "ZVE32X")
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_xsfmmbase)
  list(APPEND SKL_QEMU_CPU_OPTIONS "x-xsfmmbase=true")
  list(APPEND RISCV_EXTENSIONS "XSFMM")
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_xsfmm32a8f)
  list(APPEND SKL_QEMU_CPU_OPTIONS "x-xsfmm32a8f=true")
  list(APPEND RISCV_EXTENSIONS "XSFMM32A8F")
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_xsfmm32a8i)
  list(APPEND SKL_QEMU_CPU_OPTIONS "x-xsfmm32a8i=true")
  list(APPEND RISCV_EXTENSIONS "XSFMM32A8I")
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_xsfmm32a16f)
  list(APPEND SKL_QEMU_CPU_OPTIONS "x-xsfmm32a16f=true")
  list(APPEND RISCV_EXTENSIONS "XSFMM32A16F")
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_xsfmm32a32f)
  list(APPEND SKL_QEMU_CPU_OPTIONS "x-xsfmm32a32f=true")
  list(APPEND RISCV_EXTENSIONS "XSFNN32A32F")
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES "__riscv_xsfmm[0-9]+t")
  string(REGEX REPLACE
      ".*__riscv_xsfmm([0-9]+)t.*"
      "\\1"
      XSFMM_TE
      ${SKL_PREPROCESSOR_OUTPUT}
  )
  list(APPEND SKL_QEMU_CPU_OPTIONS "x-xsfmm${XSFMM_TE}t=true")
endif()

if(${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_xsfvfbfa)
  list(APPEND SKL_QEMU_CPU_OPTIONS "x-xsfvfbfa=true")
  list(APPEND RISCV_EXTENSIONS "XSFVFBFA")
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_xsfvfexpa)
  list(APPEND SKL_QEMU_CPU_OPTIONS "x-xsfvfexpa=true")
  list(APPEND RISCV_EXTENSIONS "XSFVFEXPA")
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_xsfvfexp16e)
  list(APPEND SKL_QEMU_CPU_OPTIONS "x-xsfvfexp16e=true")
  list(APPEND RISCV_EXTENSIONS "XSFVFEXP16E")
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_xsfvfbfexp16e)
  list(APPEND SKL_QEMU_CPU_OPTIONS "x-xsfvfbfexp16e=true")
  list(APPEND RISCV_EXTENSIONS "XSFVFBFEXP16E")
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_xsfvfexp32e)
  list(APPEND SKL_QEMU_CPU_OPTIONS "x-xsfvfexp32e=true")
  list(APPEND RISCV_EXTENSIONS "XSFVFEXP32E")
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_xsfvqdotq)
  list(APPEND SKL_QEMU_CPU_OPTIONS "x-xsfvqdotq=true")
  list(APPEND RISCV_EXTENSIONS "XSFVQDOTQ")
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_zvfofp8min)
  list(APPEND SKL_QEMU_CPU_OPTIONS "x-zvfofp8min=true")
  list(APPEND RISCV_EXTENSIONS "ZVFOFP8MIN")
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_zvfofp4min)
  list(APPEND SKL_QEMU_CPU_OPTIONS "x-zvfofp4min=true")
  list(APPEND RISCV_EXTENSIONS "ZVFOFP4MIN")
endif()

string(JOIN "," SKL_QEMU_CPU_OPTIONS ${SKL_QEMU_CPU_OPTIONS})
set(SKL_QEMU_OPTIONS -cpu ${SKL_QEMU_CPU_OPTIONS})
