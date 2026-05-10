#include "opps.h"
#include <stdio.h>
#include <stdlib.h>
#include  "tensor_runtime.h"


#ifdef __cplusplus
extern "C" {
#endif
}
#endif


struct base_tensor add(struct base_tensor *a, struct base_tensor *b) {
    if (a->dtype != b->dtype) {
        printf("Error: add: dtype missing");
        return NULL;
    }
}

static void sub() {}

static void div() {}

static void mul() {}

static void transpose() {}

static void sigmoid() {}

static void gelu() {}

static void concat() {}