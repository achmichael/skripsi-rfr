# Prediksi Masa Pakai Token dan Estimasi Biaya Listrik Menggunakan Random Forest Regression

Project ini membangun model Random Forest Regression untuk keperluan skripsi dari dasar untuk dua skenario prediksi konsumsi listrik rumah tangga:

- Prabayar: memprediksi masa pakai token listrik dalam satuan hari.
- Pascabayar: memprediksi rata-rata tagihan listrik bulanan dalam Rupiah.

Implementasi model tidak menggunakan library machine learning seperti scikit-learn. Library eksternal hanya digunakan untuk manipulasi data, operasi numerik, penyimpanan hasil, dan visualisasi.

## Ringkasan Sistem

Sistem membaca data survei konsumsi listrik, membersihkan data, melakukan feature engineering, encoding fitur kategorikal, membagi data menjadi train-test, melatih Random Forest Regression dari scratch, lalu menyimpan model dan hasil evaluasi.

Target prediksi:

| Dataset | Target | Satuan |
| --- | --- | --- |
| Prabayar | `Token_Habis_Dalam_Hari` | Hari |
| Pascabayar | `Rata_Rata_Tagihan_Per_Bulan_Rp` | Rupiah |

Input utama:

- Data demografi sederhana, seperti kota/kabupaten dan jumlah anggota keluarga.
- Informasi listrik rumah, seperti daya listrik dan jenis pembayaran.
- Informasi perangkat listrik, seperti jumlah alat, estimasi watt, jam pemakaian, dan energi harian.
- Informasi pembayaran, seperti nominal token terakhir, frekuensi isi token, bulan tagihan, dan stabilitas tagihan.

## System Design

### Arsitektur High-Level

```text
data/raw/*.csv
    |
    v
src/train.py
    |
    v
Dataset-specific pipeline
src/models/prabayar.py
src/models/pascabayar.py
    |
    v
Preprocessing
src/core/data_cleaner.py
src/core/feature_engineer.py
src/core/encoder.py
    |
    v
Train-test split
src/core/splitter.py
    |
    v
Random Forest Regression
src/forest/random_forest_regressor.py
src/forest/bootstrap.py
src/tree/decision_tree_regressor.py
src/tree/decision_tree_node.py
    |
    v
Evaluation and persistence
src/core/metrics.py
src/utils/file_writer.py
    |
    v
results/*.pkl
results/evaluation_*.json
```

### Komponen Utama

| Komponen | File | Tanggung Jawab |
| --- | --- | --- |
| Training entrypoint | `src/train.py` | Menjalankan pipeline training berdasarkan argumen dataset. |
| Model pipeline prabayar | `src/models/prabayar.py` | Menjalankan preprocessing untuk dataset prabayar. |
| Model pipeline pascabayar | `src/models/pascabayar.py` | Menjalankan preprocessing untuk dataset pascabayar. |
| Data cleaner | `src/core/data_cleaner.py` | Missing value handling, konversi tipe data, drop kolom tidak dipakai, dan clipping outlier. |
| Feature engineering | `src/core/feature_engineer.py` | Membuat fitur energi harian, total energi, dan fitur turunan pembayaran. |
| Encoder | `src/core/encoder.py` | Mengubah fitur kategorikal menjadi angka. |
| Splitter | `src/core/splitter.py` | Memisahkan fitur `X` dan target `y`, lalu membagi train-test. |
| Random forest | `src/forest/random_forest_regressor.py` | Mengelola training banyak decision tree dan agregasi prediksi. |
| Bootstrap | `src/forest/bootstrap.py` | Membuat bootstrap sample dengan replacement untuk setiap tree. |
| Decision tree | `src/tree/decision_tree_regressor.py` | Membangun tree regression berdasarkan split yang meminimalkan MSE. |
| Tree node | `src/tree/decision_tree_node.py` | Struktur node untuk tree. |
| Metrics | `src/core/metrics.py` | Fungsi evaluasi MAE, MSE, RMSE, R2, dan MAPE. |
| File writer | `src/utils/file_writer.py` | Menyimpan model dan hasil evaluasi. |
| Config | `src/utils/config.py` | Konfigurasi target, fitur, path, dan hyperparameter. |

## Desain Model Random Forest

Random Forest Regression pada project ini dibangun dari komponen berikut:

1. Bootstrap sampling

Setiap tree dilatih menggunakan sample acak dengan replacement dari data training. Posisi baris bootstrap digunakan untuk menjaga `X_sample` dan `y_sample` tetap sinkron.

2. Decision tree regression

Setiap tree memilih split berdasarkan penurunan error. Threshold dibuat dari midpoint nilai unik pada fitur numerik.

3. Random feature selection

Pada setiap proses split, tree hanya mempertimbangkan subset fitur. Nilai default `max_features="sqrt"` berarti jumlah fitur yang dipertimbangkan adalah akar kuadrat dari total fitur.

4. Aggregation

Prediksi akhir random forest adalah rata-rata prediksi semua tree.

```text
y_pred = mean(tree_1(X), tree_2(X), ..., tree_n(X))
```

## Struktur Direktori

```text
.
|-- data/
|   |-- raw/
|   |   |-- prabayar.csv
|   |   `-- pascabayar.csv
|   |-- processed/
|   `-- split/
|-- docs/
|-- experiments/
|-- outputs/
|   |-- models/
|   |-- predictions/
|   `-- reports/
|-- results/
|   |-- *.pkl
|   `-- evaluation_*.json
|-- src/
|   |-- core/
|   |-- forest/
|   |-- models/
|   |-- tree/
|   |-- utils/
|   `-- train.py
|-- requirements.txt
`-- README.md
```

## Instalasi

Gunakan Python 3.11 atau versi kompatibel.

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependency:

```bash
pip install -r requirements.txt
```

Dependency utama:

- `pandas`
- `numpy`
- `matplotlib`

## Cara Menjalankan Training

Training dataset prabayar:

```bash
python src/train.py --dataset prabayar
```

Training dataset pascabayar:

```bash
python src/train.py --dataset pascabayar
```

Jika menggunakan virtual environment Windows yang sudah tersedia di project:

```powershell
.\venv\Scripts\python.exe src\train.py --dataset prabayar
.\venv\Scripts\python.exe src\train.py --dataset pascabayar
```

## Output

Setelah training selesai, sistem menyimpan:

- Model pickle di `results/<dataset>_<timestamp>.pkl`
- Hasil evaluasi di `results/evaluation_<dataset>_<timestamp>.json`

Contoh hasil evaluasi:

```json
{
    "dataset": "prabayar",
    "mse": 26.777185549574543,
    "timestamp": "20260503_125902"
}
```

## Konfigurasi

Konfigurasi utama berada di `src/utils/config.py`.

Parameter random forest:

| Parameter | Default | Keterangan |
| --- | --- | --- |
| `n_estimators` | `100` | Jumlah tree dalam forest. |
| `max_depth` | `10` | Kedalaman maksimum setiap tree. |
| `min_samples_split` | `2` | Minimum sample untuk melakukan split. |
| `min_samples_leaf` | `1` | Minimum sample pada leaf node. |
| `max_features` | `"sqrt"` | Jumlah fitur acak yang dipertimbangkan saat split. |
| `bootstrap` | `True` | Menggunakan bootstrap sampling. |
| `random_state` | `42` | Seed agar hasil lebih reproducible. |

Parameter preprocessing:

| Parameter | Default | Keterangan |
| --- | --- | --- |
| `test_size` | `0.2` | Proporsi data test. |
| `shuffle` | `True` | Mengacak data sebelum split. |
| `numeric_missing_strategy` | `"median"` | Strategi imputasi numerik. |
| `categorical_missing_strategy` | `"most_frequent"` | Strategi imputasi kategorikal. |
| `handle_outliers` | `True` | Mengaktifkan clipping outlier. |
| `outlier_method` | `"iqr"` | Metode outlier berbasis IQR. |
| `scaling` | `False` | Scaling tidak wajib untuk random forest. |

## Alur Data

### 1. Load Data

`src/train.py` membaca file CSV dari:

- `data/raw/prabayar.csv`
- `data/raw/pascabayar.csv`

Path ditentukan melalui `config["paths"]["raw_data"]`.

### 2. Data Cleaning

`DataCleaner` melakukan:

- Mengisi missing value numerik dengan median atau mean.
- Mengisi missing value kategorikal dengan modus.
- Mengubah fitur numerik yang terbaca sebagai string menjadi angka.
- Menghapus kolom yang tidak digunakan.
- Melakukan clipping outlier dengan metode IQR.

### 3. Feature Engineering

`FeatureEngineer` berisi fungsi untuk:

- Menghitung energi harian perangkat listrik.
- Menghitung energi mesin cuci berdasarkan frekuensi mingguan.
- Menghitung total energi utama dan alat lain.
- Membuat fitur turunan untuk prabayar dan pascabayar.

Catatan: pada pipeline saat ini, model memanggil sebagian fungsi feature engineering secara eksplisit. Jika ingin memakai seluruh pipeline fitur turunan, gunakan `engineer_features()`.

### 4. Encoding

`Encoder` mengubah fitur kategorikal menjadi kode numerik. Ini diperlukan karena decision tree dari scratch saat ini hanya menerima nilai numerik untuk membentuk threshold split.

### 5. Split Dataset

`Splitter` memisahkan:

- `X`: semua fitur selain target.
- `y`: target prediksi.

Target tidak ikut masuk ke fitur model. Hal ini penting untuk mencegah data leakage.

### 6. Training Random Forest

Untuk setiap estimator:

- Ambil bootstrap sample dari `X_train`.
- Ambil `y_train` dengan posisi bootstrap yang sama.
- Train decision tree.
- Simpan tree ke dalam list forest.

### 7. Prediction dan Evaluation

Prediksi dibuat dengan merata-ratakan output semua tree. Evaluasi utama yang saat ini dicetak oleh `train.py` adalah MSE.

## Best Practice yang Dipakai

- Target dipisahkan dari fitur sebelum training.
- Bootstrap sampling menjaga sinkronisasi antara `X` dan `y`.
- Random feature selection digunakan untuk membangun diversity antar tree.
- Split train-test dilakukan setelah target dipisahkan.
- Model tidak menggunakan library machine learning siap pakai.
- Konfigurasi target, fitur, path, dan parameter model dipusatkan di `config.py`.

## Catatan Kualitas dan Batasan Saat Ini

Beberapa hal masih perlu diperhatikan jika project ini digunakan untuk eksperimen akademik atau produksi:

- Preprocessing masih dilakukan sebelum train-test split, sehingga ada potensi data leakage pada imputasi, clipping outlier, dan encoding.
- Label encoding pada fitur nominal dapat memberi urutan palsu, misalnya kota A dianggap lebih kecil dari kota B. One-hot encoding lebih aman untuk fitur nominal.
- Outlier clipping saat ini juga dapat memengaruhi target. Untuk evaluasi yang objektif, target sebaiknya tidak diubah sembarangan.
- `FeatureEngineer.engineer_features()` belum sepenuhnya dipakai oleh pipeline model.
- Metrik yang disimpan saat ini hanya MSE, walaupun fungsi MAE, RMSE, R2, dan MAPE sudah tersedia.
- Belum ada unit test untuk bootstrap, splitter, preprocessing, dan tree.
- Belum ada OOB score, padahal bootstrap sudah menyediakan dasar untuk menghitung out-of-bag evaluation.

## Rekomendasi Pengembangan Lanjutan

1. Buat preprocessing berbasis `fit` dan `transform`.

Preprocessing harus fit di data train saja, lalu transform ke data test. Ini mengurangi risiko leakage.

2. Gunakan one-hot encoding untuk fitur nominal.

Fitur seperti `Kota/Kabupaten`, kategori alat, dan jenis alat lebih cocok one-hot daripada label encoding.

3. Aktifkan evaluasi lengkap.

Simpan `MAE`, `MSE`, `RMSE`, dan `R2` agar interpretasi performa lebih jelas.

4. Tambahkan OOB evaluation.

Out-of-bag score dapat menjadi validasi internal random forest tanpa validasi tambahan.

5. Tambahkan unit test.

Minimal test:

- Target tidak ada di `X_train`.
- Semua fitur setelah preprocessing numeric.
- Bootstrap sample memiliki panjang sama dengan data train.
- `X_sample` dan `y_sample` selalu sinkron.
- Prediksi memiliki jumlah baris sama dengan `X_test`.

6. Rapikan output directory.

Saat ini config mendefinisikan path di `outputs/`, tetapi `FileWriter` menyimpan ke `results/`. Pilih satu standar agar struktur project konsisten.

## Troubleshooting

### KeyError target saat bootstrap

Error:

```text
KeyError: "['Token_Habis_Dalam_Hari'] not found in axis"
```

Penyebab:

- Target sudah benar dipisahkan dari `X`.
- Kode lama masih mencoba melakukan `sample.drop(columns=[y.name])` pada bootstrap sample dari `X`.

Solusi:

- Bootstrap harus mengambil sample dari `X`.
- Posisi bootstrap digunakan untuk mengambil `y` yang sesuai.

### TypeError saat membuat threshold split

Error:

```text
TypeError: unsupported operand type(s) for /: 'str' and 'int'
```

Penyebab:

- Masih ada fitur string yang belum dikonversi atau diencoding.

Solusi:

- Pastikan fitur numerik dikonversi ke numeric.
- Pastikan fitur kategorikal diencoding sebelum masuk ke model.

## Status Verifikasi Terakhir

Perintah berikut berhasil dijalankan:

```powershell
.\venv\Scripts\python.exe src\train.py --dataset prabayar
.\venv\Scripts\python.exe src\train.py --dataset pascabayar
```

Hasil terakhir:

| Dataset | Status | MSE |
| --- | --- | --- |
| Prabayar | Berhasil | `26.777185549574543` |
| Pascabayar | Berhasil | `14421218286.188038` |

## Lisensi

Belum ditentukan.
