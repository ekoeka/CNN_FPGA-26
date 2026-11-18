#include <stdio.h>
#include <stdint.h>
#include "weights.h"
int lsi_cnn_128x128_hls(const uint8_t img[1][128][128], int32_t logits_out[NUM_CLASSES]);
int main() {
    uint8_t dummy_img[1][128][128];
    int32_t logits[NUM_CLASSES];
    for (int h = 0; h < 128; h++)
        for (int w = 0; w < 128; w++)
            dummy_img[0][h][w] = (uint8_t)((h * 2 + w) & 0xFF);
    int pred = lsi_cnn_128x128_hls(dummy_img, logits);
    const char* labels[2] = {"Real (Asli)", "Altered (Dimanipulasi)"};
    printf("=== SIMULASI INFERENSI HLS LSI_CNN_128x128 ===\n");
    printf("Prediksi Kelas : %d (%s)\n", pred, labels[pred]);
    for (int c = 0; c < NUM_CLASSES; c++)
        printf("Logit Kelas %d  : %.4f (raw Q4.12: %d)\n", c, (float)logits[c] / (1 << FRAC_BITS), logits[c]);
    return 0;
}
