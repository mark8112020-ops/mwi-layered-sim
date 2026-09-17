#!/usr/bin/env bash
# Full pipeline, in the order the spec requires: validate the forward model, generate,
# train, then the headline phase-error sweep. Stops at the first failure -- in
# particular it will not reach the ML stage if the forward-model checks fail.
set -euo pipefail

PY=${PY:-.venv/bin/python}
N=${N:-12000}
N_SWEEP=${N_SWEEP:-6000}

echo "### 0. dielectric tables vs published anchors"
$PY tissue_params.py --plot

echo
echo "### 1. forward-model validation (spec section 7)"
$PY validate.py -v

echo
echo "### 2. dataset"
$PY generate.py --n "$N" --out data/dataset.npz

echo
echo "### 3. feature-set comparison at nominal noise"
$PY train.py --data data/dataset.npz --out results/metrics_main.json | tee results/train_log.txt

echo
echo "### 4. phase-error sweep (the deliverable)"
$PY sweep_phase_error.py --data data/dataset.npz --n "$N_SWEEP" | tee results/sweep_log.txt

echo
echo "### 5. contrast sensitivity -- does the conclusion survive a weaker tumour contrast?"
for c in 0.0 0.5 1.5; do
  $PY generate.py --n 6000 --contrast "$c" --seed 424242 --out "data/dataset_c${c}.npz" --skip-validation
  $PY train.py --data "data/dataset_c${c}.npz" --out "results/metrics_contrast_${c}.json" \
      --models rf --sets A B D --skip-importance --skip-leakage-audit
done

echo
echo "### 6. true null -- identical classes, must score exactly 0.5"
$PY generate.py --n 6000 --contrast 0.0 --benign-lesion-prob 1.0 --seed 999 \
    --out data/dataset_null.npz --skip-validation
$PY train.py --data data/dataset_null.npz --out results/metrics_true_null.json \
    --models rf --sets A B D --skip-importance

echo
echo "### 7. band ablation -- retrain without each band (NOT importance) to price the antennas"
$PY train.py --data data/dataset.npz --band low --out results/metrics_band_low.json \
    --models rf --sets A B D --skip-importance --skip-leakage-audit
$PY train.py --data data/dataset.npz --band mid --out results/metrics_band_mid.json \
    --models rf --sets A B D --skip-importance --skip-leakage-audit
$PY train.py --data data/dataset.npz --fmax 7e9 --out results/metrics_band_2to7.json \
    --models rf --sets A B D --skip-importance --skip-leakage-audit

echo
echo "done. figures and metrics are in results/"
