# Replikasi Parsial: Apache Spark untuk Peramalan Beban pada Low Carbon London

Tugas Eksplorasi Mandiri Mahadata (IF25-40402), Institut Teknologi Sumatera.

Artikel acuan: Ali, H. A. E.-S., Alham, M. H., & Ibrahim, D. K. (2024). Big data resolving using Apache Spark
for load forecasting and demand response in smart grid: A case study of Low Carbon London Project.
Journal of Big Data, 11, Article 59. https://doi.org/10.1186/s40537-024-00909-6

## Tingkat replikasi
Replikasi parsial dengan perluasan. Bagian inti artikel (pembersihan, agregasi, analitik deskriptif, seleksi fitur,
model MLlib) diulang pada satu berkas data (30 rumah tangga) dengan Spark mode lokal di Google Colab.
Perluasan: pengaruh jumlah partition dan ukuran kelompok rumah.

## Perbedaan terhadap artikel
- Skala: 30 rumah tangga (954.433 rekaman valid), bukan 5.567 rumah dan 167 juta rekaman.
- Unit "blok": seluruh 30 rumah di satu berkas, karena berkas resmi tidak memuat ID blok.
- Cuaca: Open-Meteo menggantikan Darksky yang sudah tidak aktif.
- Lingkungan: Spark lokal di Colab, bukan klaster Dataproc 1 master + 3 worker.
- Split 80/20 acak (seed 42), 3-fold CV, dan grid hiperparameter ditetapkan sendiri karena artikel tidak merinci.
- Daftar hari libur disusun sendiri dan belum diverifikasi penuh terhadap sumber resmi.

## Struktur
- src/01_prepare_data.py : pembersihan dan agregasi (Spark SQL)
- src/02_descriptive.py : analisis deskriptif dan korelasi Pearson
- src/03_train_models.py : regresi polinomial (plain, Lasso, Ridge, Elastic Net) dan decision tree (MLlib)
- src/04_partition_experiment.py : pengaruh jumlah partition
- src/05_group_size_experiment.py : pengaruh ukuran kelompok rumah
- src/06_make_figures.py : grafik hasil
- results/ dan figures/ : keluaran
- data/raw/ : sumber data kecil (cuaca, hari libur); CSV smart meter tidak disertakan karena ukuran

## Menyiapkan lingkungan
Dijalankan di Google Colab dengan Java 17 (versi yang dipakai Spark) dan PySpark. Versi persis ada di requirements-lock.txt.

    pip install -r requirements.txt
    java -version        # butuh Java 17 atau lebih baru

## Data
1. Smart meter: London Datastore, "SmartMeter Energy Consumption Data in London Households" (UK Power Networks).
   Berkas yang dipakai: LCL-June2015v2_0.csv. Letakkan di data/raw/halfhourly/ (folder ini tidak masuk repositori).
   https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d
2. Cuaca: Open-Meteo Historical Weather API, tersedia di data/raw/weather_london_2013.json.
3. Hari libur Inggris 2011-2014: data/raw/uk_bank_holidays.csv (disusun sendiri, belum diverifikasi penuh).
Rincian ada di data/raw/README_data.md.

## Menjalankan ulang

    bash run_all.sh

Angka hasil dapat sedikit berbeda antar mesin karena pembagian data acak bergantung pada partisi;
waktu eksekusi bergantung pada mesin.

## Penggunaan generative AI
[Isi sesuai izin dosen: bagian kode dan laporan yang dibantu AI, dan pernyataan bahwa penulis telah
memeriksa dan memahami isinya.]
