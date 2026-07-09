# Sigmoid Kernels

The sigmoid (logistic) function is a widely-used activation function primitive in large language models (LLMs), computer vision, and stable diffusion neural networks.
SKL's sigmoid computes the logistic function with an optional scaling factor `beta` and optional fused elementwise multiplications:

```
out[i] = logistic(beta * x[i])
if (y)  out[i] *= y[i]
if (up) out[i] *= (up[i] + delta)
```

where

```
logistic(z) = 1 / (1 + e^(-z))
```

The `x`, `y`, and `up` parameters are pointers to input arrays; when `x` is `NULL` the kernel computes nothing.
`y` and `up` are optional: when non-`NULL` they are applied elementwise.
This makes it possible to fuse SiLU (`y == x`, `beta == 1`, `up == NULL`), GLU (`y != x`, `up == NULL`) and SwiGLU (via `up`/`delta`) into a single kernel.

## Kernel Naming Convention

The kernel names follow the pattern: `skl_sigmoid_<type>_<isa>`

where:

- `<type>`: denotes the type of the input and output arrays.
- `<isa>`: denotes the instruction set architecture extension(s) the kernel requires.

## Constraints

- Input and output arrays may overlap only for in-place computation.
- For maximum performance, input and output arrays should be naturally aligned.

## Kernel List

- FP32 APIs
```c
void skl_sigmoid_f32_zve32f(float *out, float beta, const float *x, const float *y, const float *up, float delta, size_t n);
void skl_sigmoid_f32_xsfvfexpa(float *out, float beta, const float *x, const float *y, const float *up, float delta, size_t n);
void skl_sigmoid_f32_xsfvfexp32e(float *out, float beta, const float *x, const float *y, const float *up, float delta, size_t n);
```

- FP16 APIs

```c
void skl_sigmoid_f16_zvfh(_Float16 *out, _Float16 beta, const _Float16 *x, const _Float16 *y, const _Float16 *up, _Float16 delta, size_t n);
void skl_sigmoid_f16_xsfvfexpa_zvfh(_Float16 *out, _Float16 beta, const _Float16 *x, const _Float16 *y, const _Float16 *up, _Float16 delta, size_t n);
void skl_sigmoid_f16_xsfvfexp16e(_Float16 *out, _Float16 beta, const _Float16 *x, const _Float16 *y, const _Float16 *up, _Float16 delta, size_t n);
```

- BF16 APIs

```c
void skl_sigmoid_bf16_zve32f(__bf16 *out, __bf16 beta, const __bf16 *x, const __bf16 *y, const __bf16 *up, __bf16 delta, size_t n);
void skl_sigmoid_bf16_xsfvfexpa(__bf16 *out, __bf16 beta, const __bf16 *x, const __bf16 *y, const __bf16 *up, __bf16 delta, size_t n);
void skl_sigmoid_bf16_xsfvfexp32e(__bf16 *out, __bf16 beta, const __bf16 *x, const __bf16 *y, const __bf16 *up, __bf16 delta, size_t n);
void skl_sigmoid_bf16_xsfvfbfa(__bf16 *out, __bf16 beta, const __bf16 *x, const __bf16 *y, const __bf16 *up, __bf16 delta, size_t n);
void skl_sigmoid_bf16_xsfvfbfexp16e_xsfvfbfa(__bf16 *out, __bf16 beta, const __bf16 *x, const __bf16 *y, const __bf16 *up, __bf16 delta, size_t n);
```

## Notes

- If `x` is `NULL`, the kernel computes nothing.
- For `beta * x = -∞`, implementations return NaN.

## Wrapper APIs

For convenience and optimization, each `skl_sigmoid_<type>_<isa>` kernel ships with a set of thin wrappers that specialize the core sigmoid into common activation functions.
They follow the pattern `skl_<op>_<type>_<isa>` and are defined in terms of the core sigmoid as follows:

| Wrapper | Computes | In terms of `skl_sigmoid` |
| --- | --- | --- |
| `skl_logistic_<type>_<isa>(out, x, n)` | `sigmoid(x)` | `sigmoid(out, 1, x, NULL, NULL, 0, n)` |
| `skl_silu_<type>_<isa>(out, x, n)` | `x * sigmoid(x)` | `sigmoid(out, 1, x, x, NULL, 0, n)` |
| `skl_swish_<type>_<isa>(out, beta, x, n)` | `x * sigmoid(beta * x)` | `sigmoid(out, beta, x, x, NULL, 0, n)` |
| `skl_glu_<type>_<isa>(out, x, y, n)` | `x * sigmoid(y)` | `sigmoid(out, 1, y, x, NULL, 0, n)` |
| `skl_swiglu_<type>_<isa>(out, gate, up, delta, n)` | `silu(gate) * (up + delta)` | `sigmoid(out, 1, gate, gate, up, delta, n)` |

These wrappers are available for every `<type>_<isa>` combination listed above.
For example, the FP32 Zve32f wrappers are:

```c
void skl_logistic_f32_zve32f(float *out, const float *x, size_t n);
void skl_silu_f32_zve32f(float *out, const float *x, size_t n);
void skl_swish_f32_zve32f(float *out, float beta, const float *x, size_t n);
void skl_glu_f32_zve32f(float *out, const float *x, const float *y, size_t n);
void skl_swiglu_f32_zve32f(float *out, const float *gate, const float *up, float delta, size_t n);
```
