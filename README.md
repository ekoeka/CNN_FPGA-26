# Klasifikasi Keaslian Sidik Jari Menggunakan Akselerator CNN Berbasis FPGA

**Nama:** Eka Hernanda Fulca  
**NIM:** 101022330137  
**Mata Kuliah:** Desain SoC dan FPGA  

Repositori ini berisi implementasi *end-to-end* perancangan akselerator perangkat keras Convolutional Neural Network (CNN) berdaya rendah untuk verifikasi keaslian citra biometrik sidik jari (**Real** vs **Altered**) menggunakan **Quantization-Aware Training (QAT) Power-of-Two ($\pm 2^k$)** dan **High-Level Synthesis (HLS)**.

---

## 1. Latar Belakang & Dataset
Sistem keamanan biometrik *offline* membutuhkan pemrosesan lokal (*on-device*) dengan latensi rendah tanpa bergantung pada komputasi *cloud*. Citra sidik jari memiliki garis alur (*minutiae*) halus yang kehilangan detail jika dikompresi ke resolusi rendah (32×32 atau 64×64), sehingga proyek ini menggunakan resolusi **128×128 piksel (1-channel Grayscale)**.

- **Dataset:** Sokoto Coventry Fingerprint Dataset (**SOCOFing** — `ruizgara/socofing`)
- **Total Sampel Latih & Uji:** 3.000 citra seimbang (1.500 `Real` dan 1.500 `Altered` meliputi *obliteration*, *central rotation*, dan *Z-cut*).
- **Pembagian Data:** 80% Training (2.400 citra) dan 20% Validation/Test (600 citra) menggunakan *stratified split*.

---

## 2. Arsitektur Jaringan (`LSI_CNN_128x128`)
Arsitektur dirancang dengan 4 blok konvolusi agar dimensi *feature map* sebelum *fully connected layer* menyusut menjadi $64 \times 4 \times 4 = 1024$, menjaga kebutuhan memori tetap muat di dalam *on-chip* BRAM/M10K FPGA:

| Layer | Konfigurasi | Dimensi Output | Jumlah Parameter (Bobot + Bias) | Memori Bobot (PoT 8-bit) |
| :--- | :--- | :--- | :--- | :--- |
| **Input** | Grayscale 1-Channel | `1 × 128 × 128` | - | - |
| **Conv1 + Pool1** | `3×3, s=2, p=1` + `MaxPool(2×2)` | `16 × 32 × 32` | 160 | 144 B |
| **Conv2 + Pool2** | `3×3, s=1, p=1` + `MaxPool(2×2)` | `32 × 16 × 16` | 4.640 | 4,5 KB |
| **Conv3 + Pool3** | `3×3, s=1, p=1` + `MaxPool(2×2)` | `64 × 8 × 8` | 18.496 | 18,0 KB |
| **Conv4 + Pool4** | `3×3, s=1, p=1` + `MaxPool(2×2)` | `64 × 4 × 4` | 36.928 | 36,0 KB |
| **FC1 + ReLU** | `Linear(1024 → 128)` | `128` | 131.200 | 128,0 KB |
| **FC2 (Output)** | `Linear(128 → 2)` | `2` | 258 | 256 B |
| **Total** | - | - | **191.682** | **~187 KB** |

---

## 3. Optimasi Perangkat Keras (QAT PoT & HLS)
1. **Multiplier-less MAC (`pot_mac`):** Melalui *Quantization-Aware Training* (QAT) dengan *Straight-Through Estimator* (STE), seluruh bobot dikuantisasi menjadi nilai pangkat dua ($\pm 2^k$). Pada kode C++ HLS (`hls/cnn_hls.cpp`), operasi perkalian digantikan sepenuhnya oleh operasi *bit-shift* (`<<` dan `>>`) serta penjumlahan/pengurangan tanpa menggunakan blok DSP.
2. **Aritmetika Fixed-Point `Q4.12`:** Aktivasi antar-layer direpresentasikan menggunakan integer 32-bit (`int32_t`) dengan faktor skala $2^{12} = 4096$.
3. **Arsitektur Memori Ping-Pong:** Menggunakan dua array statis (`buf_a[65536]` dan `buf_b[65536]`) yang dipakai secara bergantian di seluruh *layer* untuk meminimalkan penggunaan *on-chip* BRAM.

---

## 4. Struktur Repositori
```text
CNN_FPGA-26/
├── dataset/                    # Sampel data uji & skrip penarik dataset Kaggle
│   ├── samples/                # Contoh citra sidik jari Real & Altered
│   └── download_dataset.py     # Skrip otomatis unduh dataset SOCOFing
├── python/                     # Tahap Software: Training Float32, QAT PoT, & Export
│   ├── train_lsi_cnn.ipynb     # Notebook lengkap Google Colab
│   ├── model_sidik_jari_128x128_pot.pth  # Bobot terlatih PyTorch
│   └── requirements.txt        # Daftar library Python
├── hls/                        # Tahap High-Level Synthesis (C++ ke RTL)
│   ├── cnn_hls.cpp             # Desain arsitektur CNN (Shift-Add MAC & Ping-Pong Buffer)
│   ├── weights.h               # Header bobot PoT 8-bit & bias Q4.12 hasil Colab
│   ├── tb_cnn.cpp              # Testbench C++ untuk verifikasi akurasi & fungsi
│   └── Makefile                # Perintah kompilasi cepat (g++ / i++)
├── verilog/                    # Tahap RTL & Hardware (Hasil sintesis HLS)
│   └── README.md               # Berkas RTL Verilog (.v / .sv)
├── report/                     # Dokumen laporan PDF/DOCX, blok diagram, & screenshot hasil
│   └── README.md               # Aset laporan proyek
├── .gitignore                  # Mencegah puluhan ribu file .BMP mentah ikut ter-push
└── README.md                   # Laporan utama proyek
```

---

## 5. Hasil Pengujian & Evaluasi
Pengujian dilakukan pada 600 citra validasi (300 `Real` dan 300 `Altered`) yang tidak pernah dilihat model selama proses pelatihan:

| Tahap Pengujian | Presisi Bobot | Akurasi Validasi (600 Citra) | Loss Akhir | Keterangan |
| :--- | :--- | :--- | :--- | :--- |
| **Tahap 1: Standard Training** | Float32 (32-bit) | **95,83%** | 0,1658 | Baseline sebelum kuantisasi |
| **Tahap 2: QAT Fine-Tuning** | Power-of-Two ($\pm 2^k$) | **95,67%** | 0,1290 | Penurunan hanya 0,16%, siap HLS tanpa multiplier |

---

## 6. Cara Menjalankan

### A. Mengunduh Dataset & Sampel Uji (Python)
```bash
pip install -r python/requirements.txt
python dataset/download_dataset.py
```

### B. Kompilasi & Simulasi Inferensi C++ HLS
```bash
cd hls
make run
```