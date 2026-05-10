#include <limits.h>
#include <stdint.h>
#include <stdlib.h>

#include "tensor_runtime.h"

// Returns the byte width for one element of the given tensor base type.
// Returns 0 for unsupported/unknown types so callers can fail early.
static size_t elem_byte_size(enum base_type type)
{
    switch (type) {
    case TYPE_FP32:
        return 4;
    case TYPE_FP16:
    case TYPE_BF16:
        return 2;
    case TYPE_FP8:
    case TYPE_I8:
        return 1;
    case TYPE_I16:
        return 2;
    case TYPE_I32:
        return 4;
    case TYPE_I64:
        return 8;
    case TYPE_F64:
        return 8;
    default:
        return 0;
    }
}

// Multiplies all dimensions to produce total element count.
// Fails when dimensions are invalid (NULL, non-positive, or overflow).
static int multiply_dims(const int64_t *dims, int num_dims, int64_t *out_count)
{
    int64_t n = 1;

    if (!dims || num_dims <= 0 || !out_count)
        return -1;

    for (int i = 0; i < num_dims; i++) {
        if (dims[i] <= 0)
            return -1;
        if (n > INT64_MAX / dims[i])
            return -1;
        n *= dims[i];
    }
    *out_count = n;
    return 0;
}

// Allocates raw tensor storage.
// The context argument is currently reserved for future custom allocators.
void *create_buffer(struct context *ctx, size_t num_of_bytes)
{
    (void)ctx;

    if (num_of_bytes == 0)
        return NULL;

    return malloc(num_of_bytes);
}

// Creates a tensor for arbitrary rank (num_dims).
// Validates type and shape, computes total bytes, and allocates metadata+data.
struct base_tensor *create_tensor(
    struct context *ctx,
    enum base_type type,
    const int64_t *dims,
    int num_dims)
{
    int64_t count = 0;
    size_t esz = elem_byte_size(type);

    if (esz == 0)
        return NULL;
    if (multiply_dims(dims, num_dims, &count) != 0)
        return NULL;

    size_t nbytes = (size_t)count * esz;
    if ((int64_t)nbytes != count * (int64_t)esz)
        return NULL;

    struct base_tensor *t =
        calloc(1, sizeof(struct base_tensor) + 1);
    if (!t)
        return NULL;

    t->dtype = type;
    t->num_of_elem = count;
    t->num_of_bytes = nbytes;

    if (nbytes > 0) {
        t->data = create_buffer(ctx, nbytes);
        if (!t->data) {
            free(t);
            return NULL;
        }
    }

    t->name[0] = '\0';
    return t;
}

// Matrix (2D tensor) with shape [rows, cols].
struct base_tensor *create_matrix(
    struct context *ctx,
    enum base_type type,
    int64_t rows,
    int64_t cols)
{
    int64_t dims[2] = { rows, cols };
    return create_tensor(ctx, type, dims, 2);
}

// Flattens each [a,b] pair into consecutive axes: ... a, b, a, b, ...
struct base_tensor *create_tensor_from_dim_pairs(
    struct context *ctx,
    enum base_type type,
    const int64_t (*dim_pairs)[2],
    int num_pairs)
{
    if (!dim_pairs || num_pairs <= 0)
        return NULL;
    if (num_pairs > INT_MAX / 2)
        return NULL;

    int num_dims = num_pairs * 2;
    int64_t *dims = malloc((size_t)num_dims * sizeof(int64_t));
    if (!dims)
        return NULL;

    for (int i = 0; i < num_pairs; i++) {
        dims[2 * i] = dim_pairs[i][0];
        dims[2 * i + 1] = dim_pairs[i][1];
    }

    struct base_tensor *t = create_tensor(ctx, type, dims, num_dims);
    free(dims);
    return t;
}

// Convenience wrapper to create a 1D tensor from one dimension size.
struct base_tensor *create_tensor_1d(
    struct context *ctx,
    enum base_type type,
    int64_t *dim0)
{
    if (!dim0)
        return NULL;
    int64_t dims[1] = { *dim0 };
    return create_tensor(ctx, type, dims, 1);
}

// Convenience wrapper to create a 2D tensor from two dimension sizes.
struct base_tensor *create_tensor_2d(
    struct context *ctx,
    enum base_type type,
    int64_t *dim0,
    int64_t *dim1)
{
    if (!dim0 || !dim1)
        return NULL;
    int64_t dims[2] = { *dim0, *dim1 };
    return create_tensor(ctx, type, dims, 2);
}

// Convenience wrapper to create a 3D tensor from three dimension sizes.
struct base_tensor *create_tensor_3d(
    struct context *ctx,
    enum base_type type,
    int64_t *dim0,
    int64_t *dim1,
    int64_t *dim2)
{
    if (!dim0 || !dim1 || !dim2)
        return NULL;
    int64_t dims[3] = {
        *dim0,
        *dim1,
        *dim2,
    };
    return create_tensor(ctx, type, dims, 3);
}

// Releases a tensor and its owned data buffer.
// View tensors do not own data and therefore do not free shared storage.
void tensor_destroy(struct base_tensor *t)
{
    if (!t)
        return;
    if (!t->view_src)
        free(t->data);
    free(t);
}
