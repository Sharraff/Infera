#include "stdio.h"
#include "stdlib.h"



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

#ifdef __cplusplus
}
#endif