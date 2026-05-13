#ifndef DISPATCH_H
#define DISPATCH_H

#include <stdint.h>

#include "../runtime/tensor_runtime.h"

typedef void (*binary_kernel_fn)(
    void *out,
    const void *a,
    const void *b,
    int64_t n
);

struct binary_dispatch_table {
    binary_kernel_fn add;
    binary_kernel_fn sub;
    binary_kernel_fn mul;
    binary_kernel_fn div;
};

/* One slot per enum base_type value (0 .. TYPE_F64); entry may be NULL. */
enum { DISPATCH_BINARY_SLOTS = 9 };

extern struct binary_dispatch_table *dispatch_binary_tables[DISPATCH_BINARY_SLOTS];

#endif
