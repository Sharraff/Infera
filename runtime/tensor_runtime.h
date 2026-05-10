#ifndef TENSOR_RUNTIME_H
#define TENSOR_RUNTIME_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "../operators/opps.h"

struct tensor;
struct graph;
struct context;

enum base_type {
    TYPE_FP32 = 0,
    TYPE_FP16 = 1,
    TYPE_BF16 = 2,
    TYPE_FP8 = 3,
    TYPE_I8 = 4,
    TYPE_I16 = 5,
    TYPE_I32 = 6,
    TYPE_I64 = 7,
    TYPE_F64 = 8,
};

struct tensor_allocator {
    size_t mem_size;
    void *mem_buffer;
};

struct base_tensor {
    enum base_type dtype;
    enum base_opps op;

    int64_t num_of_elem;
    size_t num_of_bytes;

    struct base_tensor *src;

    struct base_tensor *view_src;
    size_t view_offs;

    void *data;

    char name[];
};

static const size_t tensor_size = sizeof(struct base_tensor);

/* dims[i] are axis lengths; num_dims is the tensor rank (any >= 1). */
struct base_tensor *create_tensor(
    struct context *ctx,
    enum base_type type,
    const int64_t *dims,
    int num_dims
);

/* 2D matrix: shape [rows, cols]. */
struct base_tensor *create_matrix(
    struct context *ctx,
    enum base_type type,
    int64_t rows,
    int64_t cols
);

/*
 * Shape given as a list of pairs, e.g. { {2,8}, {5,1}, {4,2}, {8,6} }
 * becomes a tensor of rank 2*num_pairs with dimensions in order:
 * 2, 8, 5, 1, 4, 2, 8, 6.
 */
struct base_tensor *create_tensor_from_dim_pairs(
    struct context *ctx,
    enum base_type type,
    const int64_t (*dim_pairs)[2],
    int num_pairs
);

struct base_tensor *create_tensor_1d(
    struct context *ctx,
    enum base_type type,
    int64_t *dim0
);

struct base_tensor *create_tensor_2d(
    struct context *ctx,
    enum base_type type,
    int64_t *dim0,
    int64_t *dim1
);

struct base_tensor *create_tensor_3d(
    struct context *ctx,
    enum base_type type,
    int64_t *dim0,
    int64_t *dim1,
    int64_t *dim2
);

void *create_buffer(struct context *ctx, size_t num_of_bytes);

void tensor_destroy(struct base_tensor *t);

#endif
