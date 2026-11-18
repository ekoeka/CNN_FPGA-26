#include <stdint.h>
#include "weights.h"

#ifdef __INTELFPGA_COMPILER__
#include "HLS/hls.h"
#define HLS_COMPONENT component
#define HLS_SLAVE_MEM hls_avalon_slave_memory_argument(16384)
#define HLS_SLAVE_REG hls_avalon_slave_register_argument
#else
#define HLS_COMPONENT
#define HLS_SLAVE_MEM
#define HLS_SLAVE_REG
#endif

static inline int32_t pot_mac(int32_t acc, int32_t val, int8_t w_code) {
    if (w_code != 0 && val != 0) {
        int8_t abs_code = (w_code > 0) ? w_code : -w_code;
        int8_t k = abs_code - 16;
        int32_t prod = (k >= 0) ? (val << k) : (val >> (-k));
        acc += (w_code > 0) ? prod : -prod;
    }
    return acc;
}

void conv2d_relu(
    const int32_t* in_buf, int32_t* out_buf,
    const int8_t* weights, const int32_t* bias,
    int in_c, int in_h, int in_w,
    int out_c, int stride, int pad
) {
    int out_h = (in_h + 2 * pad - 3) / stride + 1;
    int out_w = (in_w + 2 * pad - 3) / stride + 1;

    for (int oc = 0; oc < out_c; oc++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                int32_t acc = bias[oc];
                for (int ic = 0; ic < in_c; ic++) {
                    for (int kh = 0; kh < 3; kh++) {
                        #pragma unroll
                        for (int kw = 0; kw < 3; kw++) {
                            int ih = oh * stride + kh - pad;
                            int iw = ow * stride + kw - pad;
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                int in_idx = (ic * in_h + ih) * in_w + iw;
                                int w_idx  = (((oc * in_c + ic) * 3) + kh) * 3 + kw;
                                acc = pot_mac(acc, in_buf[in_idx], weights[w_idx]);
                            }
                        }
                    }
                }
                out_buf[(oc * out_h + oh) * out_w + ow] = (acc > 0) ? acc : 0;
            }
        }
    }
}

void maxpool2x2(const int32_t* in_buf, int32_t* out_buf, int channels, int in_h, int in_w) {
    int out_h = in_h >> 1;
    int out_w = in_w >> 1;
    for (int c = 0; c < channels; c++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                int ih = oh << 1;
                int iw = ow << 1;
                int32_t v00 = in_buf[(c * in_h + ih)     * in_w + iw];
                int32_t v01 = in_buf[(c * in_h + ih)     * in_w + (iw + 1)];
                int32_t v10 = in_buf[(c * in_h + ih + 1) * in_w + iw];
                int32_t v11 = in_buf[(c * in_h + ih + 1) * in_w + (iw + 1)];
                int32_t m0 = (v00 > v01) ? v00 : v01;
                int32_t m1 = (v10 > v11) ? v10 : v11;
                out_buf[(c * out_h + oh) * out_w + ow] = (m0 > m1) ? m0 : m1;
            }
        }
    }
}

void linear_layer(
    const int32_t* in_buf, int32_t* out_buf,
    const int8_t* weights, const int32_t* bias,
    int in_features, int out_features, bool apply_relu
) {
    for (int o = 0; o < out_features; o++) {
        int32_t acc = bias[o];
        for (int i = 0; i < in_features; i++) {
            acc = pot_mac(acc, in_buf[i], weights[o * in_features + i]);
        }
        if (apply_relu && acc < 0) acc = 0;
        out_buf[o] = acc;
    }
}

HLS_COMPONENT int lsi_cnn_128x128_hls(
    HLS_SLAVE_MEM const uint8_t img[1][128][128],
    HLS_SLAVE_REG int32_t logits_out[NUM_CLASSES]
) {
    static int32_t buf_a[65536];
    static int32_t buf_b[65536];

    for (int h = 0; h < 128; h++) {
        for (int w = 0; w < 128; w++) {
            buf_a[h * 128 + w] = ((int32_t)img[0][h][w] << FRAC_BITS) / 255;
        }
    }

    conv2d_relu(buf_a, buf_b, conv1_weight, conv1_bias, 1, 128, 128, 16, 2, 1);
    maxpool2x2(buf_b, buf_a, 16, 64, 64);

    conv2d_relu(buf_a, buf_b, conv2_weight, conv2_bias, 16, 32, 32, 32, 1, 1);
    maxpool2x2(buf_b, buf_a, 32, 32, 32);

    conv2d_relu(buf_a, buf_b, conv3_weight, conv3_bias, 32, 16, 16, 64, 1, 1);
    maxpool2x2(buf_b, buf_a, 64, 16, 16);

    conv2d_relu(buf_a, buf_b, conv4_weight, conv4_bias, 64, 8, 8, 64, 1, 1);
    maxpool2x2(buf_b, buf_a, 64, 8, 8);

    linear_layer(buf_a, buf_b, fc1_weight, fc1_bias, 1024, 128, true);
    linear_layer(buf_b, buf_a, fc2_weight, fc2_bias, 128, NUM_CLASSES, false);

    int best_class = 0;
    int32_t max_val = buf_a[0];
    logits_out[0] = buf_a[0];

    for (int c = 1; c < NUM_CLASSES; c++) {
        logits_out[c] = buf_a[c];
        if (buf_a[c] > max_val) {
            max_val = buf_a[c];
            best_class = c;
        }
    }
    return best_class;
}
