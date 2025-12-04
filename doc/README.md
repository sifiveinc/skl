# Project Documentation Style Guide

The primary documentation for individual SKL functions is provided in the form of Doxygen-style block comments.
Aside from this, project-level documentation that applies to more than one kernel will be supplied in markdown files throughout the source tree.

## Markdown Style

Markdown files should use [semantic linebreaks](https://sembr.org/) to facilitate version control and editing.

## Doxygen Style

Each public function (i.e., those declared with `SKL_FUNC`) should have a prototype in its accompanying header file with a Doxygen-style block comment immediately above it.
Each line of the block comment except the first should begin with `* `.
The Doxygen block should have the following format:
1. The first line should begin the block comment using a double-asterisk `/**`, followed by a newline.
2. The next line should begin with `@brief`, followed by a 1-line synopsis. This is the text that will be below the function name in the table of contents page of the compiled Doxygen document.
3. The next line should be empty.
4. The next line(s) should describe each parameter of the function, with each line consisting of `@param <param-name> - <description>`.
That is: the bare parameter name (no `[in]`/`[out]` clutter), followed by a dash to separate it from the description, and then a description.
Do not attempt to align the starting positions of the descriptions across all the parameters, but if one parameter's description spills onto multiple lines, then do align all of its lines.
If the function uses multiple parameters, they should be described in the same order that they are used in the function prototype.
The `<param-name>` should match the name of the parameter in the prototype (including casing).
5. The next line should be empty.
6. The next line(s) should express any additional information needed to describe the function or its usage.
If a kernel specializes a generic function (like the GEMM kernels specialize their scalar references), show the equivalent function call using markdown (```) blocks.
Do not use `@details` -- this is implied.
7. [Optional] The next line(s) may contain one or more `@note` blocks, with each `@note` block beginning on its own line and an empty line between successive notes.
This is good for notes about input ranges for which the kernel is most suitable, or for emphasizing an important API constraint.
8. The next line should be the final line containing exactly ` */` to end the block comment.
The function prototype should follow immediately on the next line (i.e., with no empty lines in between the documentation string and the prototype).

An example documentation string following the above standard is as follows:
```c
/**
 * @brief 1-ULP float32 exponential function.
 *
 * @param out - Output array where results are stored.
 * @param in - Input array of float32 values.
 * @param n - Number of elements to process.
 *
 * Computes `out[i] = exp(in[i])` for 0 ≤ i < n.
 *
 * Results are accurate to less than 0.927 ulp.
 */
void skl_exp_1u_f32_zve32f(float *out, const float *in, size_t n);
```