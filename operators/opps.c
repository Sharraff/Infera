#include "opps.h"

#include <stdlib.h>

#include "dispatch.h"
#include "tensor_runtime.h"

static int tensor_same_shape(
    const struct base_tensor *a,
    const struct base_tensor *b
)
{
    if (!a || !b || !a->shape || !b->shape)
        return 0;
    if (a->num_dims != b->num_dims)
        return 0;
    for (int i = 0; i < a->num_dims; i++) {
        if (a->shape[i] != b->shape[i])
            return 0;
    }
    return 1;
}

enum tensor_binary_kind {
    TENSOR_BINARY_ADD,
    TENSOR_BINARY_SUB,
    TENSOR_BINARY_MUL,
    TENSOR_BINARY_DIV,
};

static struct base_tensor *tensor_binary_dispatch(
    struct context *ctx,
    struct base_tensor *a,
    struct base_tensor *b,
    enum tensor_binary_kind kind
)
{
    if (!a || !b)
        return NULL;

    if (a->dtype != b->dtype)
        return NULL;

    if (!tensor_same_shape(a, b))
        return NULL;

    if ((int)a->dtype < 0 || (int)a->dtype >= DISPATCH_BINARY_SLOTS)
        return NULL;

    struct binary_dispatch_table *table = dispatch_binary_tables[a->dtype];
    if (!table)
        return NULL;

    binary_kernel_fn fn = NULL;
    switch (kind) {
    case TENSOR_BINARY_ADD:
        fn = table->add;
        break;
    case TENSOR_BINARY_SUB:
        fn = table->sub;
        break;
    case TENSOR_BINARY_MUL:
        fn = table->mul;
        break;
    case TENSOR_BINARY_DIV:
        fn = table->div;
        break;
    default:
        return NULL;
    }

    if (!fn)
        return NULL;

    struct base_tensor *out =
        create_tensor(ctx, a->dtype, a->shape, a->num_dims);

    if (!out)
        return NULL;

    fn(out->data, a->data, b->data, a->num_of_elem);

    return out;
}

struct base_tensor *tensor_add(
    struct context *ctx,
    struct base_tensor *a,
    struct base_tensor *b
)
{
    return tensor_binary_dispatch(ctx, a, b, TENSOR_BINARY_ADD);
}

struct base_tensor *tensor_sub(
    struct context *ctx,
    struct base_tensor *a,
    struct base_tensor *b
)
{
    return tensor_binary_dispatch(ctx, a, b, TENSOR_BINARY_SUB);
}

struct base_tensor *tensor_div(
    struct context *ctx,
    struct base_tensor *a,
    struct base_tensor *b
)
{
    return tensor_binary_dispatch(ctx, a, b, TENSOR_BINARY_DIV);
}

struct base_tensor *tensor_mul(
    struct context *ctx,
    struct base_tensor *a,
    struct base_tensor *b
)
{
    return tensor_binary_dispatch(ctx, a, b, TENSOR_BINARY_MUL);
}
