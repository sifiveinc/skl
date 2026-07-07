# Sigmoid Linear Unit (SiLU) Kernels

The Sigmoid Linear Unit (SiLU) function is a widely-used activation function in large language models (LLMs), computer vision, and stable diffusion neural networks, as a smooth alternative to the Rectified Linear Unit (ReLU).
SiLU multiplies its input with a sigmoid function.
SKL's SiLU uses the logistic as its sigmoid, computing the following elementwise function:

```
silu(x) = x / (1 + e^(-x))
```

## Kernel Naming Convention

The kernel names follow the pattern: `skl_silu_<accuracy>_<type>_<isa>`

where:

- `<accuracy>`: denotes the maximum absolute error in ULP (e.g., `52u` for 52 ULP, `9u` for 9 ULP).
- `<type>`: denotes the type of the input and output arrays.
- `<isa>`: denotes the instruction set architecture extension(s) the kernel requires.

## Constraints

- Input and output arrays may overlap only for in-place computation.
- For maximum performance, input and output arrays should be naturally aligned.

## Kernel List

- FP32 APIs
```c
void skl_silu_52u_f32_zve32f(float *out, const float *in, size_t n);
void skl_silu_52u_f32_xsfvfexpa(float *out, const float *in, size_t n);
void skl_silu_52u_f32_xsfvfexp32e(float *out, const float *in, size_t n);
```

- FP16 APIs

```c
void skl_silu_9u_f16_zvfh(_Float16 *out, const _Float16 *in, size_t n);
void skl_silu_9u_f16_xsfvfexpa_zvfh(_Float16 *out, const _Float16 *in, size_t n);
void skl_silu_31u_f16_xsfvfexp16e(_Float16 *out, const _Float16 *in, size_t n);
```

## Notes

For x = -∞, implementations return NaN.
