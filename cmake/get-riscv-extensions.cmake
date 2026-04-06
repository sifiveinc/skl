
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

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES "__riscv_zfhmin")
  list(APPEND SKL_QEMU_CPU_OPTIONS "zfhmin=true")
  list(APPEND RISCV_EXTENSIONS "zfhmin")
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES "__riscv_zfh ")
  # The above space is necessary to distinguish from zfhmin
  list(APPEND SKL_QEMU_CPU_OPTIONS "zfh=true")
  list(APPEND RISCV_EXTENSIONS "zfh")
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES "__riscv_zvfhmin")
  list(APPEND SKL_QEMU_CPU_OPTIONS "zvfhmin=true")
  list(APPEND RISCV_EXTENSIONS "zvfhmin")
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES "__riscv_zvfh ")
  list(APPEND SKL_QEMU_CPU_OPTIONS "zvfh=true")
  list(APPEND RISCV_EXTENSIONS "zvfh")
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES "__riscv_zvfbfwma")
  list(APPEND SKL_QEMU_CPU_OPTIONS "zvfbfwma=true")
  list(APPEND RISCV_EXTENSIONS "zvfbfwma")
endif()

if(${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_zvfbfmin)
  list(APPEND SKL_QEMU_CPU_OPTIONS "zvfbfmin=true")
  list(APPEND RISCV_EXTENSIONS "zvfbfmin")
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_zve32f)
  list(APPEND SKL_QEMU_CPU_OPTIONS "zve32f=true")
  list(APPEND RISCV_EXTENSIONS "zve32f")
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_zve64d)
  list(APPEND SKL_QEMU_CPU_OPTIONS "zve64d=true")
  list(APPEND RISCV_EXTENSIONS "zve64d")
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_zve32x)
  list(APPEND SKL_QEMU_CPU_OPTIONS "zve32x=true")
  list(APPEND RISCV_EXTENSIONS "zve32x")
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_xsfmmbase)
  list(APPEND SKL_QEMU_CPU_OPTIONS "x-xsfmmbase=true")
  list(APPEND RISCV_EXTENSIONS "xsfmm")
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_xsfmm32a8f)
  list(APPEND SKL_QEMU_CPU_OPTIONS "x-xsfmm32a8f=true")
  list(APPEND RISCV_EXTENSIONS "xsfmm32a8f")
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_xsfmm32a8i)
  list(APPEND SKL_QEMU_CPU_OPTIONS "x-xsfmm32a8i=true")
  list(APPEND RISCV_EXTENSIONS "xsfmm32a8i")
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_xsfmm32a16f)
  list(APPEND SKL_QEMU_CPU_OPTIONS "x-xsfmm32a16f=true")
  list(APPEND RISCV_EXTENSIONS "xsfmm32a16f")
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_xsfmm32a32f)
  list(APPEND SKL_QEMU_CPU_OPTIONS "x-xsfmm32a32f=true")
  list(APPEND RISCV_EXTENSIONS "xsfmm32a32f")
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
  list(APPEND RISCV_EXTENSIONS "xsfvfbfa")
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_xsfvfexpa)
  list(APPEND SKL_QEMU_CPU_OPTIONS "x-xsfvfexpa=true")
  list(APPEND RISCV_EXTENSIONS "xsfvfexpa")
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_xsfvfexp16e)
  list(APPEND SKL_QEMU_CPU_OPTIONS "x-xsfvfexp16e=true")
  list(APPEND RISCV_EXTENSIONS "xsfvfexp16e")
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_xsfvfbfexp16e)
  list(APPEND SKL_QEMU_CPU_OPTIONS "x-xsfvfbfexp16e=true")
  list(APPEND RISCV_EXTENSIONS "xsfvfbfexp16e")
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_xsfvfexp32e)
  list(APPEND SKL_QEMU_CPU_OPTIONS "x-xsfvfexp32e=true")
  list(APPEND RISCV_EXTENSIONS "xsfvfexp32e")
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_xsfvqdotq)
  list(APPEND SKL_QEMU_CPU_OPTIONS "x-xsfvqdotq=true")
  list(APPEND RISCV_EXTENSIONS "xsfvqdotq")
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_zvfofp8min)
  list(APPEND SKL_QEMU_CPU_OPTIONS "x-zvfofp8min=true")
  list(APPEND RISCV_EXTENSIONS "zvfofp8min")
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_zvfofp4min)
  list(APPEND SKL_QEMU_CPU_OPTIONS "x-zvfofp4min=true")
  list(APPEND RISCV_EXTENSIONS "zvfofp4min")
endif()

string(JOIN "," SKL_QEMU_CPU_OPTIONS ${SKL_QEMU_CPU_OPTIONS})
set(SKL_QEMU_OPTIONS -cpu ${SKL_QEMU_CPU_OPTIONS})
