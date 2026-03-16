
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

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES "__riscv_zfh ")
  # The above space is necessary to distinguish from zfhmin
  set(RISCV_ZFH ON)
  list(APPEND SKL_QEMU_CPU_OPTIONS "zfh=true")
else()
  set(RISCV_ZFH OFF)
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES "__riscv_zvfh ")
  set(RISCV_ZVFH ON)
  list(APPEND SKL_QEMU_CPU_OPTIONS "zvfh=true")
else()
  set(RISCV_ZVFH OFF)
endif()

if(${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_zvfbfmin)
  list(APPEND SKL_QEMU_CPU_OPTIONS "zvfbfmin=true")
  set(RISCV_ZVFBFMIN ON)
else()
  set(RISCV_ZVFBFMIN OFF)
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_zve32f)
  list(APPEND SKL_QEMU_CPU_OPTIONS "zve32f=true")
  set(RISCV_ZVE32F ON)
else()
  set(RISCV_ZVE32F OFF)
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_zve64d)
  list(APPEND SKL_QEMU_CPU_OPTIONS "zve64d=true")
  set(RISCV_ZVE64D ON)
else()
  set(RISCV_ZVE64D OFF)
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_zve32x)
  list(APPEND SKL_QEMU_CPU_OPTIONS "zve32x=true")
  set(RISCV_ZVE32X ON)
else()
  set(RISCV_ZVE32X OFF)
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_xsfmmbase)
  set(RISCV_XSFMM ON)
  list(APPEND SKL_QEMU_CPU_OPTIONS "x-xsfmmbase=true")
else()
  set(RISCV_XSFMM OFF)
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_xsfmm32a8f)
  set(RISCV_XSFMM32A8F ON)
  list(APPEND SKL_QEMU_CPU_OPTIONS "x-xsfmm32a8f=true")
else()
  set(RISCV_XSFMM32A8F OFF)
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_xsfmm32a8i)
  set(RISCV_XSFMM32A8I ON)
  list(APPEND SKL_QEMU_CPU_OPTIONS "x-xsfmm32a8i=true")
else()
  set(RISCV_XSFMM32A8I OFF)
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_xsfmm32a16f)
  set(RISCV_XSFMM32A16F ON)
  list(APPEND SKL_QEMU_CPU_OPTIONS "x-xsfmm32a16f=true")
else()
  set(RISCV_XSFMM32A16F OFF)
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_xsfmm32a32f)
  set(RISCV_XSFMM32A32F ON)
  list(APPEND SKL_QEMU_CPU_OPTIONS "x-xsfmm32a32f=true")
else()
  set(RISCV_XSFMM32A32F OFF)
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
  set(RISCV_XSFVFBFA ON)
else()
  set(RISCV_XSFVFBFA OFF)
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_xsfvfexpa)
  set(RISCV_XSFVFEXPA ON)
  list(APPEND SKL_QEMU_CPU_OPTIONS "x-xsfvfexpa=true")
else()
  set(RISCV_XSFVFEXPA OFF)
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_xsfvfexp16e)
  set(RISCV_XSFVFEXP16E ON)
  list(APPEND SKL_QEMU_CPU_OPTIONS "x-xsfvfexp16e=true")
else()
  set(RISCV_XSFVFEXP16E OFF)
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_xsfvfbfexp16e)
  set(RISCV_XSFVFBFEXP16E ON)
  list(APPEND SKL_QEMU_CPU_OPTIONS "x-xsfvfbfexp16e=true")
else()
  set(RISCV_XSFVFBFEXP16E OFF)
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_xsfvfexp32e)
  set(RISCV_XSFVFEXP32E ON)
  list(APPEND SKL_QEMU_CPU_OPTIONS "x-xsfvfexp32e=true")
else()
  set(RISCV_XSFVFEXP32E OFF)
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_xsfvqdotq)
  set(RISCV_XSFVQDOTQ ON)
  list(APPEND SKL_QEMU_CPU_OPTIONS "x-xsfvqdotq=true")
else()
  set(RISCV_XSFVQDOTQ OFF)
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_zvfofp8min)
  set(RISCV_ZVFOFP8MIN ON)
  list(APPEND SKL_QEMU_CPU_OPTIONS "x-zvfofp8min=true")
else()
  set(RISCV_ZVFOFP8MIN OFF)
endif()

if (${SKL_PREPROCESSOR_OUTPUT} MATCHES __riscv_zvfofp4min)
  set(RISCV_ZVFOFP4MIN ON)
  list(APPEND SKL_QEMU_CPU_OPTIONS "x-zvfofp4min=true")
else()
  set(RISCV_ZVFOFP4MIN OFF)
endif()

string(JOIN "," SKL_QEMU_CPU_OPTIONS ${SKL_QEMU_CPU_OPTIONS})
set(SKL_QEMU_OPTIONS -cpu ${SKL_QEMU_CPU_OPTIONS})
