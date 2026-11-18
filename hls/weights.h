#ifndef WEIGHTS_H
#define WEIGHTS_H
#include <stdint.h>
#define NUM_CLASSES 2
#define FRAC_BITS 12
const int8_t  conv1_weight[144]    = {0};
const int32_t conv1_bias[16]       = {0};
const int8_t  conv2_weight[4608]   = {0};
const int32_t conv2_bias[32]       = {0};
const int8_t  conv3_weight[18432]  = {0};
const int32_t conv3_bias[64]       = {0};
const int8_t  conv4_weight[36864]  = {0};
const int32_t conv4_bias[64]       = {0};
const int8_t  fc1_weight[131072]   = {0};
const int32_t fc1_bias[128]        = {0};
const int8_t  fc2_weight[256]      = {0};
const int32_t fc2_bias[2]          = {0};
#endif
