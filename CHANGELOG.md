# SKL Changelog

### v3.0.0

Initial release of SKL. Features include:
* Supported kernels:
    * Matrix multiply (GEMM):
        * RVV: `float16`, widening `float16`, `float32`, `float64`, quad-widening `int8`.
        * Xsfmm: quad-widening OFP8, widening `float16`, widening `bfloat16`, `float32`, quad-widening `int8`.
        * Xsfvqdotq: quad-widening packed `int8`, packing kernel for B matrix.
    * Matrix transposition:
        * RVV: format-agnostic implementations for SEW 8, 16, 32 types.
        * Xsfmm: format-agnostic implementations for SEW 8, 16, 32 types.
    * Exponential:
        * RVV: `float16`, `bfloat16`, `float32`.
        * Xsfvfexp32e: `float32`.
        * Xsfvfexp16e: `float16`.
        * Xsfvfbfexp16e: `bfloat16`.
        * Xsfvfexpa: `float16`, `bfloat16`, `float32`.
        * Xsfvfbfa: `bfloat16`.
        * Zvfbfmin: `bfloat16`.
    * Softmax:
        * RVV: `float16`, `bfloat16`, `float32`.
        * Xsfvfexp32e: `float32`.
        * Xsfvfexp16e: `float16`.
        * Xsfvfbfexp16e: `bfloat16`.
        * Xsfvfexpa: `float16`, `bfloat16`, `float32`.
        * Xsfvfbfa: `bfloat16`.
        * Zvfbfmin: `bfloat16`.
    * Logistic:
        * RVV: `float16`, `float32`.
        * Xsfvfexp16e: `float16`.
        * Xsfvfexpa: `float32`.
    * Depthwise Convolution 2D:
        * RVV: `float16`, `float32`, quad-widening `int8`.
    * OFP Conversion:
        * Zvfofp8min: narrowing `float32` to OFP8, widening OFP8 to `bfloat16`.
        * Zvfofp8min + Zvfbfmin: narrowing `bfloat16` to OFP8.
        * Zvfofp4min: widening OFP4 to OFP8.
* Test suite: tests included for every kernel which assess functional correctness and measure performance.
* Documentation: each public function documented with a Doxygen documentation string, as well as detailed Markdown documents in most directories.
