# Im2Row Kernels

The `im2row` directory contains functions for converting HWC layout convolution patches into matrix rows.

## Kernel List

### `skl_im2row_generic_hwc()`
A straightforward implementation that processes elements individually:
- **Element-by-element processing**: Iterates through each element in the patch
- **HWC layout optimization**: Designed specifically for Height-Width-Channels tensor layout
- **Flexible patch extraction**: Supports arbitrary starting coordinates and patch sizes
- **Bounds checking**: Handles zero-padding for out-of-bounds accesses beyond tensor boundaries
- **Dilation support**: Implements dilated convolution patterns
- **Generic type support**: Works with any primitive data type via `void*` and `element_size`

### `skl_im2row_hwc_zve32x()`
An optimized implementation that leverages bulk memory operations:
- **Bulk memory operations**: Uses `memcpy` for efficient channel-wise data movement
- **HWC layout optimization**: Exploits channel-contiguous memory layout in HWC tensors
- **Three-phase processing**: Handles head, middle (full channels), and tail portions separately
- **Memory efficiency**: Minimizes function call overhead through bulk operations
- **Channel optimization**: Optimized for contiguous channel data in HWC layout
- **Generic type support**: Same as generic version, works with any primitive data type via `void*` and `element_size`

> **Note**: See `im2row_hwc.h` for detailed API documentation with complete function signatures, parameter descriptions, and usage examples.

