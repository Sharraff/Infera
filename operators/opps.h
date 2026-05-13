#ifndef OPPS_H
#define OPPS_H

#include "stdio.h"
#include "stdlib.h"

struct context;
struct base_tensor;

#ifdef __cplusplus
extern "C" {
#endif
enum base_opps {
    OPP_ADD = 1,
    OPP_SUB = 2,
    OPP_DIV = 3,
    OPP_MUL = 4,
    OPP_TRANSPOSE = 5,
    OPP_SIGMOID = 6,
    OPP_SOFTMAX = 7,
    OPP_GELU = 8,
    OPP_CONCAT = 9
};

struct base_tensor *tensor_add(
    struct context *ctx,
    struct base_tensor *a,
    struct base_tensor *b
);

struct base_tensor *tensor_sub(
    struct context *ctx,
    struct base_tensor *a,
    struct base_tensor *b
);

struct base_tensor *tensor_div(
    struct context *ctx,
    struct base_tensor *a,
    struct base_tensor *b
);

struct base_tensor *tensor_mul(
    struct context *ctx,
    struct base_tensor *a,
    struct base_tensor *b
);

#ifdef __cplusplus
}
#endif

#endif /* OPPS_H */