#!/usr/bin/env bash
# Menjalankan seluruh eksperimen berurutan. Pakai: bash run_all.sh
set -e
mkdir -p data/processed results figures
python src/01_prepare_data.py
python src/02_descriptive.py
python src/03_train_models.py
python src/04_partition_experiment.py
python src/05_group_size_experiment.py
python src/06_make_figures.py
