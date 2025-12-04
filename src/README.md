# SKL Kernels Overview

## SKL Kernel Naming Convention

Kernels exposed in the SKL API are defined with the `SKL_FUNC` attribute and use the following naming convention using snake casing:

```C
skl_<op>_<spec>_<types>_<isa>[_<cpu>]
```

where:
- `<op>`: denotes the high level operation that the kernel computes (e.g., `gemm`, `softmax`, `depthwise_conv2d`, etc.).
- `<spec>`: denotes any specialization of the parameters that normally occur in `<op>` kernels.
This is highly dependent on the specific operation and users should refer to the documentation for specific kernels for the meaning of the specializations.
- `<types>`: denotes the datatype(s) used in the kernel.
Most kernels operate on a single datatype, but some may use multiple.
Only the most important buffers are notated, such as the input and output matrices for a matrix-matrix multiply.
See the [type specifier convention](#type-specifier-convention) section below for details on specific types.
- `<isa>`: denotes the most relevant instruction set architecture extension the
kernel requires or is targeting. This is not the full `-march` field passed to the compiler but rather the most salient RISC-V extension used, such as `zve32f` or `xsfvqmaccqoq`.
- `[<cpu>]`: denotes the specific microarchitecture the kernel is designed for and for which high performance is intended.
Most generic kernels are not tuned for a specific microarchitecture and so omit this field.

### Type Specifier Convention

Each datatype is represented in the `<types>` field with a string generally following the RISC-V intrinsics format:

| `<type>` | Description | C Type |
|----------|-------------|---------|
| `f8e4m3` | 8-bit OFP8 E4M3 floating point | `uint8_t` |
| `bf16` | 16-bit bfloat16 floating point | `__bf16` |
| `f16`  | 16-bit IEEE half-precision floating point | `_Float16` |
| `f32`  | 32-bit IEEE floating point | `float` |
| `f64`  | 64-bit IEEE floating point | `double` |
| `e8` | Un-typed 8-bit elements | `uint8_t` |
| `i8`  | 8-bit signed integer | `int8_t` |
| `u8`  | 8-bit unsigned integer | `uint8_t` |
| `e16` | Un-typed 16-bit elements | `uint16_t` |
| `i16`  | 16-bit signed integer | `int16_t` |
| `u16`  | 16-bit unsigned integer | `uint16_t` |
| `e32` | Un-typed 32-bit elements | `uint32_t` |
| `i32`  | 32-bit signed integer | `int32_t` |
| `u32`  | 32-bit unsigned integer | `uint32_t` |
| `e64` | Un-typed 64-bit elements | `uint64_t` |
| `i64`  | 64-bit signed integer | `int64_t` |
| `u64`  | 64-bit unsigned integer | `uint64_t` |

That is, the specifier consists of one or two characters to represent the format (floating-point, signed integer, unsigned integer), followed by the number of bits in the type.

When a kernel depends only on the element width, but not the format, the type specifier `e` is used, as in `e32` for 32-bit elements of any format.
This is expressed in the C language as `uint32_t` or the equivalent for other widths.

Currently, there is no built-in type for OFP8 E4M3 floating point numbers, so these are type-punned as 8-bit unsigned integers.

A type specifier must consist of **at least** one of the values in the first column of the table above, but some kernels--especially GEMM--may extend the type specifier with additional characters to indicate other properties of the data such as memory layout.

### ISA Specifier Convention

Most kernels are compatible with a subset of the full RISC-V vector ("V") extension (denoted by `rvv`), and so are marked with the minimal required ISA, such as `zve64d` or `zve32x`.
Others require an extension of RVV, and so are marked with the ISA they require in addition, such as the standard `zvfh` (half-precision floating point) or SiFive custom `xsfvfexp32e` (32-bit exponential).
