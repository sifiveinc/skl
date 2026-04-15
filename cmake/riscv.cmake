set(CMAKE_SYSTEM_PROCESSOR riscv)

set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")

set(CMAKE_C_COMPILER riscv64-unknown-elf-clang)
option(SKL_ENABLE_CLANG_TIDY "Enable clang-tidy command" ON)
if(SKL_ENABLE_CLANG_TIDY)
  set(CMAKE_C_CLANG_TIDY riscv64-unknown-elf-clang-tidy)
endif()

set(SKL_ARCH_EXTENSIONS
  rv64gcv
  zba
  zbb
  # xsfmmbase
  # xsfmm32a8f
  # xsfmm32a8i
  # xsfmm32a16f
  # xsfmm32a32f
  # xsfmm64t
  zfh
  zvfh
  # xsfvfbfa
  # xsfvfbfexp16e
  # xsfvfexp16e
  # xsfvfexp32e
  # xsfvfexpa
  # zvfofp8min0p2
  # zvfofp4min0p1
  # xsfvqdotq
  zvfbfmin
  zvfbfwma
  zihintntl
)

string(JOIN "_" SKL_ARCH_STR ${SKL_ARCH_EXTENSIONS})
set(COMPILE_OPTIONS -march=${SKL_ARCH_STR} -menable-experimental-extensions)

set(SKL_COMPILE_OPTIONS ${COMPILE_OPTIONS})
set(SKL_TEST_COMPILE_OPTIONS ${COMPILE_OPTIONS})

set(SKL_TESTER "qemu-riscv64")
set(SKL_TESTER_OPTIONS ${SKL_QEMU_OPTIONS})
