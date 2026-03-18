# SKL Test & Benchmark Framework Redesign


This PR introduces a new test driver framework that replaces the legacy monolithic test files with a three-layer architecture:

1. **Driver** (`skl-test-driver.{c,h}`) - Environment-specific test execution engine (provides memory allocation, performance counters, I/O)
2. **Harness** (e.g., [gemm/gemm_f32rc_f32rc_f32rc.{c,h}](gemm/gemm_f32rc_f32rc_f32rc.h)) - Operation-specific test logic shared across all kernel variants in a family (init, verify, report, cleanup callbacks)
3. **Test Suite** (e.g., [gemm/xsfmm/skl_gemm_*.c](gemm/xsfmm/skl_gemm_a1b01_f32c_f32_f32_xsfmm32a32f.c)) - Kernel-specific test definitions (array of test configs + execute callback)

See [README.md](README.md) for detailed usage guide, examples, and buffer management.
See [skl-test-driver.h](skl-test-driver.h) for complete API reference.

## Status

This PR only converts FP32 GEMM tests to the new framework as an example.
I would like to hear everyone's feedback and suggestions now based on this initial example.
Once we all agree on the design, I will ask individual owners of each kernel family to migrate their tests to this new framework.

Ultimately, we want to eliminate the legacy `skl-test.h` framework, and possibly that file itself unless we keep some utility functions in there.
