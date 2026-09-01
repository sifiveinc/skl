# Copyright (c) 2025-2026 SiFive, Inc. All rights reserved.
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.
# SPDX-License-Identifier: MIT

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR riscv)

set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")

set(CMAKE_C_COMPILER riscv64-unknown-elf-clang)

set(SKL_ARCH_EXTENSIONS
  rv64gcv
  zba
  zbb
  zvl512b
  # xsfmmbase
  # xsfmm32a8f
  # xsfmm32a8i
  # xsfmm32a16f
  # xsfmm32a32f
  # xsfmm64t
  # zfh
  # zvfh
  # xsfvfbfa
  # xsfvfbfexp16e
  # xsfvfexp16e
  # xsfvfexp32e
  # xsfvfexpa
  # zvfofp8min0p2
  # zvfofp4min0p1
  # xsfvqdotq
  # zvfbfmin
  # zvfbfwma
  zvqwbdota8i0p2
  # zihintntl
)

string(JOIN "_" SKL_ARCH_STR ${SKL_ARCH_EXTENSIONS})
set(COMPILE_OPTIONS -march=${SKL_ARCH_STR} -menable-experimental-extensions)

set(SKL_COMPILE_OPTIONS ${COMPILE_OPTIONS})
set(SKL_TEST_COMPILE_OPTIONS ${COMPILE_OPTIONS})

set(SKL_TESTER "qemu-riscv64")
set(SKL_TESTER_OPTIONS ${SKL_QEMU_OPTIONS})
