#include "opps.h"
#include <stdio.h>
#include <stdlib.h>
#include  "tensor_runtime.h"


struct base_tensor add(struct base_tensor *a, struct base_tensor *b) {
    if (a->dtype != b->dtype) {
        printf("Error: add: dtype missing");
        return NULL;
    }
}

struct base_tensor sub(struct base_tensor *a, struct base_tensor *b) {}

struct base_tensor div(struct base_tensor *a) {};

struct base_tensor mul(struct base_tensor *a, struct base_tensor *b) {};

struct base_tensor transpose(struct base_tensor *a) {};

struct base_tensor sigmoid(struct base_tensor *a) {};

struct base_tensor gelu(struct base_tensor *a) {};

static void concat() {}