#!/usr/bin/env bash
# Full pipeline, in the order the spec requires: validate the forward model, generate,
# train, then the headline phase-error sweep. Stops at the first failure -- in
# particular it will not reach the ML stage if the forward-model checks fail.
set -euo pipefail
cd "$(dirname "$0")/.."   # run from the repo root so data/ and results/ land there

PY=${PY:-.venv/bin/python}
N=${N:-12000}
N_SWEEP=${N_SWEEP:-6000}

echo "### 0. dielectric tables vs published anchors"
$PY python/tissue_params.py --plot

echo
echo "### 1. forward-model validation (spec section 7)"
$PY python/validate.py -v

echo
echo "### 2. dataset"
$PY python/generate.py --n "$N" --out data/dataset.npz

echo
echo "### 3. feature-set comparison at nominal noise"
$PY python/train.py --data data/dataset.npz --out results/metrics_main.json

echo
echo "### 4. phase-error sweep (the deliverable)"
$PY python/sweep_phase_error.py --data data/dataset.npz --n "$N_SWEEP"

echo
echo "### 5. contrast sensitivity -- does the conclusion survive a weaker tumour contrast?"
for c in 0.0 0.5 1.5; do
  $PY python/generate.py --n 6000 --contrast "$c" --seed 424242 --out "data/dataset_c${c}.npz" --skip-validation
  $PY python/train.py --data "data/dataset_c${c}.npz" --out "results/metrics_contrast_${c}.json" \
      --models rf --sets A B D --skip-importance --skip-leakage-audit
done

echo
echo "### 6. true null -- identical classes, must score exactly 0.5"
$PY python/generate.py --n 6000 --contrast 0.0 --benign-lesion-prob 1.0 --seed 999 \
    --out data/dataset_null.npz --skip-validation
$PY python/train.py --data data/dataset_null.npz --out results/metrics_true_null.json \
    --models rf --sets A B D --skip-importance

echo
echo "### 7. band ablation -- retrain without each band (NOT importance) to price the antennas"
$PY python/train.py --data data/dataset.npz --band low --out results/metrics_band_low.json \
    --models rf --sets A B D --skip-importance --skip-leakage-audit
$PY python/train.py --data data/dataset.npz --band mid --out results/metrics_band_mid.json \
    --models rf --sets A B D --skip-importance --skip-leakage-audit
$PY python/train.py --data data/dataset.npz --fmax 7e9 --out results/metrics_band_2to7.json \
    --models rf --sets A B D --skip-importance --skip-leakage-audit

echo
echo "done. figures and metrics are in results/"
