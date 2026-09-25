# mwi-layered-sim: full details

The short version is in the [README](../README.md). This is the complete record: model, sourcing, methods, every result table, limitations and open questions.

Layered-media S11 simulation for the complex-vs-magnitude classification question.

**The question:** does the MWI probe's front end need to recover complex S11, or is
|S11| enough? Magnitude-only allows a scalar detector front end. Complex S11 requires a
phase-coherent receiver, which is the difference between a six-port reflectometer with
an ADL5961 and something closer to a real VNA in both parts count and calibration
burden. The entire commercial argument is that the device is cheap enough to sit in a
dermatology office, so the answer here decides whether that argument survives.

**The deliverable** is `results/fig_phase_sweep.png`: classification performance of each
feature set against receiver phase-error magnitude. The crossing point, if there is one,
is the engineering spec for how good the phase reference has to be.

This is a 1D analytical model, not full-wave. CST is unavailable (license) and would be
the wrong tool anyway: full-wave gives tens of samples, a tree ensemble needs thousands
before the accuracy difference between two feature sets means anything, and this gives
unlimited samples in seconds (~840 samples/s on a laptop).

---

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r python/requirements.txt
./python/run_all.sh
```

Or step by step:

```bash
.venv/bin/python python/tissue_params.py           # dielectric tables vs published anchors
.venv/bin/python python/validate.py -v             # forward-model checks (spec section 7)
.venv/bin/python python/generate.py --n 12000      # dataset -> data/dataset.npz
.venv/bin/python python/train.py                   # feature-set comparison at nominal noise
.venv/bin/python python/sweep_phase_error.py --n 6000   # the headline experiment
```

`validate.py` runs automatically inside `generate.py` and the generator refuses to write
a dataset if any check fails. That is deliberate: do not run the ML stage on a broken
forward model.

---

## Repo layout

| path | what it is |
|---|---|
| `python/tissue_params.py` | dielectric models + sources, single source of truth |
| `python/layered.py` | `eps_c()`, impedance recursion, `s11()` |
| `python/validate.py` | the five forward-model checks from spec section 7 |
| `python/generate.py` | sampling, noise injection, dataset to npz |
| `python/features.py` | feature set A/B/C/D extraction |
| `python/train.py` | CV, metrics, permutation importance, leakage audit, band ablation |
| `python/sweep_phase_error.py` | the headline experiment |
| `python/tests/` | pytest wrapper around the same checks |
| `python/notebooks/` | exploration only, nothing load-bearing |
| `matlab/` | the physics half ported to base MATLAB (no toolboxes); see `matlab/README.md` |
| `results/` | metrics json + figures, committed |
| `data/` | generated datasets, gitignored |

Run every Python command from the repository root, so `data/` and `results/` land at
the top level. A dataset regenerates from its seed in about 15 seconds, which is why
`data/` is not versioned.

---|---|
| `tissue_params.py` | dielectric models + sources, single source of truth |
| `layered.py` | `eps_c()`, impedance recursion, `s11()` |
| `validate.py` | the five forward-model checks from spec section 7 |
| `generate.py` | sampling, noise injection, dataset to npz |
| `features.py` | feature set A/B/C/D extraction |
| `train.py` | CV, metrics, permutation importance, leakage audit |
| `sweep_phase_error.py` | the headline experiment |
| `tests/` | pytest wrapper around the same checks |
| `notebooks/` | exploration only, nothing load-bearing |
| `results/` | metrics json + figures, committed |

Fold this into the existing MIS GitHub repo rather than keeping it standalone, so the
handoff markdown files stay in one place. `data/` is gitignored -- a dataset is
regenerable from its seed in about 15 seconds, so there is no reason to version 50 MB.

---

## The model

Skin as a stack of homogeneous planar layers, plane wave at normal incidence, S11 by
transmission-line impedance recursion from the bottom layer up.

```
eta_i   = eta_0 / sqrt(eps_c_i)
gamma_i = 1j * (w/c) * sqrt(eps_c_i)
Z_N     = eta_N                                              (semi-infinite)
Z_i     = eta_i * (Z_{i+1} + eta_i*tanh(g_i*d_i)) / (eta_i + Z_{i+1}*tanh(g_i*d_i))
S11     = (Z_1 - eta_ref) / (Z_1 + eta_ref)
```

Layer stack, top to bottom:

| layer | thickness | notes |
|---|---|---|
| coupling medium / antenna substrate | n/a (reference impedance) | RT/duroid 5880, eps_r 2.2 |
| coupling gap (air) | 0-150 um | probe standoff / contact-pressure proxy |
| stratum corneum | 10-20 um | low water, low loss |
| epidermis | 50-150 um | |
| dermis | 1.0-3.0 mm | dominant layer in our band |
| hypodermis (fat) | 4-10 mm | effectively semi-infinite at 2 GHz+ |

**Branch convention.** With time dependence `exp(+jwt)` and `eps_c = eps' - j*eps''`,
numpy's principal `sqrt` gives a root with positive real part and negative imaginary
part, so `gamma` has positive real part and `exp(-gamma*z)` decays into the tissue.
`layered.assert_decaying()` checks this explicitly and validation case 2 checks it end
to end on a lossy half-space. This is the one thing here that silently ruins everything
downstream if it is wrong.

### Dielectric data

Every dielectric number lives in `tissue_params.py` with a per-entry `source` string.
Anything not traceable to a published table carries `provisional=True` and shows up in

```bash
.venv/bin/python python/tissue_params.py --todo
```

The GHz-band tissues are Gabriel's 4-Cole-Cole parametric models, and
`python python/tissue_params.py` checks them against published anchor values before you trust
anything:

```
tissue                  f GHz  eps model  eps pub  sig model  sig pub   ok
skin_dry                 2.45      38.01    38.00      1.464    1.460   OK
skin_dry                10.00      31.30    31.30      8.012    8.010   OK
skin_wet                 2.45      42.85    42.90      1.593    1.590   OK
fat_not_infiltrated      2.45       5.28     5.28      0.098    0.105   OK
fat_infiltrated          2.45      10.82    10.80      0.268    0.268   OK
```

Currently provisional and needing a real source:

- **stratum corneum** -- Gabriel has no SC entry; the values are our low-water Debye fit.
  At 10-20 um the SC is electrically thin over 2-26 GHz, so the outcome should be
  insensitive to it, which validation case 4 lets you confirm.
- **epidermis / dermis** -- mapped to Gabriel's Skin (Dry) / Skin (Wet). That mapping is
  convention, not a measurement of separated layers.
- **malignant and benign lesion** -- built as an explicit, auditable contrast on top of
  dermis (`+16%` on the first Cole-Cole pole, `+0.35 S/m` ionic), chosen to land near
  the normal-vs-malignant contrast reported by Mirbeik-Sabzevari et al. 2018 rather than
  transcribed pointwise from their figures. Achieved contrast:

  ```
    f GHz  dermis eps  malig eps     d%  dermis eps"  malig eps"     d%  benign d% eps
      2.0       43.52      49.61   14.0        12.01       15.76   31.2            7.0
     10.0       33.53      38.57   15.0        16.09       18.77   16.6            7.5
     26.0       19.79      22.79   15.2        17.11       20.05   17.2            7.5
  ```

  `--contrast` rescales the whole departure from dermis, so `--contrast 0` is a null
  experiment (tumour dielectrically identical to dermis; accuracy must collapse to
  chance) and `0.5` / `1.5` are the sensitivity analysis. **A conclusion that only
  survives at `--contrast 1.0` is not a conclusion**, which is why `run_all.sh` runs all
  three.

- **coupling gap range** -- our assumption; measure the real probe's contact
  repeatability and replace it.

### Physiological variation

Each sample draws its permittivities from a distribution around nominal, not the nominal
itself, in two parts:

- a per-sample **hydration factor** (`HYDRATION_STD = 0.08`) shared across all skin
  layers, standing in for hydration, body site and patient age
- independent per-layer **multiplicative jitter** (`JITTER_STD = 0.10`) on every
  Cole-Cole parameter, log-normal so a 10% spread cannot produce a negative permittivity

Both are label-independent, and the lesion material inherits the same hydration factor
(otherwise hydration itself becomes a label proxy). This matters: the shared component
moves eps' by more than the tumour contrast does in parts of the band, which is the
physical situation. A model without it produces a fake 99%.

---

## Leakage

Any nuisance parameter that correlates with the label will be found by the trees and
will inflate the score. Two mechanisms keep that from happening:

1. **The lesion is carved out of the stack, not appended to it.** `carve_inclusion()`
   replaces the material in a depth window and splits the affected layers, so every
   interface below the lesion keeps its position and the total stack thickness is
   bit-for-bit identical to what it would have been without a lesion. `generate.py`
   asserts this per sample and raises if it ever drifts.
2. **Benign lesions.** A configurable fraction of benign samples (default 0.5) gets a
   benign-lesion inclusion with intermediate properties drawn from the *same* geometry
   distribution, so the classifier cannot win by detecting "any inclusion".

`train.py` then audits it directly, training on the nuisance parameters alone:

- `stack_geometry` (thicknesses, hydration, gap, total thickness) -- must be ~0.5
- `lesion_geometry_given_inclusion` (depth, thickness among samples with an inclusion)
  -- must be ~0.5
- `has_inclusion` -- predictive **by design** with expected AUC `1 - benign_lesion_prob/2`,
  reported as a generator check rather than a leak, because the instrument cannot
  measure it. Set `--benign-lesion-prob 1.0` to drive it to 0.5.

---

## Measurement realism

A noiseless comparison favours complex S11 automatically -- complex data is a strict
superset of magnitude data, so that result is trivially true and tells us nothing. The
experiment only has meaning if both feature sets are corrupted the way real hardware
corrupts them. `generate.apply_measurement_noise()` applies, before feature extraction:

| effect | default | why |
|---|---|---|
| additive receiver noise | 30 dB SNR in band | thermal / detector floor |
| magnitude gain error | 2% per sweep | scalar gain uncertainty |
| magnitude drift | 1% smooth in-band ripple | slow, small |
| phase noise | 0.4 x total, iid per point | |
| phase drift | 0.92 x total, smooth per band | cable flex, temperature, imperfect cal |
| -- of which residual electrical delay | 70% of the smooth part | the dominant leftover after a one-port calibration, and what cable flex actually produces |

Phase is the fragile quantity in a cheap front end, and the residual after calibration
is not white noise -- it is mostly a linear-in-frequency delay term, so it is modelled as
one. That is exactly the effect that could make magnitude-only competitive in practice
despite being information-poor in theory.

S11 is stored **clean** in the dataset; corruption is applied downstream so the sweep can
re-corrupt the same physics at every phase-error level without regenerating stacks.

Frequency grids match the existing antennas, not a math ideal: **2-12 GHz** and
**8-26 GHz**, 301 points each.

---

## Feature sets

| ID | features | count (both bands) |
|---|---|---|
| A | Re(S11), Im(S11) per point | 1204 |
| B | \|S11\| only | 602 |
| C | \|S11\| and unwrapped phase | 1204 |
| D | \|S11\| plus engineered scalars: resonance frequencies, notch depths, band-averaged loss, slope | 628 |

A and C are the same information in different coordinates and should score the same.
`train.py` checks `|AUC(A) - AUC(C)| <= 0.02` and says so loudly if it fails -- a gap
means the phase unwrapping or the scaling is broken and the rest of the run is not
trustworthy. Phase is unwrapped per band; unwrapping across the band gap would be
meaningless.

## ML protocol

Random forest first for speed and interpretability, XGBoost second for the headline
number. Stratified 5-fold, fixed seed recorded in every results file. Reported per
feature set:

- accuracy, and per-fold mean/std
- **sensitivity and specificity separately** -- a missed melanoma and a false alarm are
  not the same error
- **specificity at a fixed 95% sensitivity** operating point, which is what a triage
  device is actually judged on
- ROC AUC and the full confusion matrix
- permutation importance aggregated by frequency, plotted as importance vs GHz -- flat
  regions are candidates for narrowing the hardware band, which is another cost lever

---

## Results

Run of 2026-08-01: 12,000 samples, seed 20260801, stratified 5-fold. Full numbers in
`results/metrics_main.json` and `results/phase_sweep.json`.

### The headline: how good does the phase reference have to be?

`results/fig_phase_sweep.png`. Random forest, n = 6000, everything except the phase
error held fixed across the sweep.

| phase error, deg rms | A (Re/Im) | B (\|S11\|) | C (\|S11\|+phase) | D (\|S11\|+scalars) |
|---|---|---|---|---|
| 0 | **0.689** | 0.664 | 0.685 | 0.663 |
| 1 | 0.672 | 0.663 | 0.671 | 0.665 |
| 2 | 0.656 | 0.664 | 0.667 | 0.664 |
| 5 | 0.636 | **0.666** | 0.656 | **0.666** |
| 10 | 0.627 | 0.663 | 0.651 | 0.666 |
| 45 | 0.563 | 0.662 | 0.647 | 0.664 |
| 90 | 0.554 | **0.668** | 0.650 | 0.667 |

**A phase-coherent receiver pays for itself only below ~1.45 deg rms phase error**
(1.21 deg on accuracy, 2.40 deg comparing C against B). Above that, magnitude-only
matches or beats it, and by 45 deg the complex feature set is *worse than useless*
relative to a scalar detector.

Two things make this a decision rather than a curve:

- The advantage complex S11 buys at its very best -- perfect phase -- is **+0.025 AUC**.
  That is the entire prize for a phase-coherent front end.
- The assumed six-port post-calibration range (2-10 deg rms) sits **entirely to the right
  of the crossing**. If that assumption is anywhere near right, the scalar detector is
  the correct choice and the cheap-device argument survives.

The whole conclusion rests on that 2-10 deg assumption, which is open question 1 and is
currently a guess. **Measuring it is the highest-value next step in the project** -- it
is the difference between "build the scalar front end" and "we need to think again".

B and D are flat across the entire sweep, which is the built-in control: phase error
cannot touch magnitude except through additive noise, and it doesn't.

### Feature sets at nominal noise (5 deg rms, 30 dB SNR)

| set | RF AUC | XGB AUC | RF sens | RF spec | spec @ 95% sens |
|---|---|---|---|---|---|
| A Re/Im | 0.646 | 0.642 | 0.605 | 0.603 | 0.115 |
| B \|S11\| | 0.664 | 0.644 | 0.602 | 0.638 | 0.122 |
| C \|S11\|+phase | 0.657 | 0.646 | 0.606 | 0.621 | 0.124 |
| D \|S11\|+scalars | **0.668** | **0.652** | 0.603 | 0.642 | 0.130 |

A/C equivalence holds in both models (delta 0.0116 RF, 0.0043 XGB, tolerance 0.02), so
the phase unwrapping and scaling are sound. Specificity at 95% sensitivity is poor
(~0.12) everywhere -- at this contrast and this much physiological variation, the model
cannot be pushed to a triage-grade operating point. That is a statement about the
simulated task, not about the hardware question, and the *comparison between feature
sets* is what this repo is for.

### Band ablation: do not cut either antenna

`results/fig_importance_vs_ghz.png` shows block-permutation importance concentrated
below ~7 GHz and indistinguishable from zero above ~12 GHz. **Acting on that alone would
have been a mistake.** Retraining on each band separately tells a different story:

| configuration | points | A | B | D |
|---|---|---|---|---|
| 2-7 GHz only | 151 | 0.599 | 0.581 | 0.581 |
| low band, 2-12 GHz | 301 | 0.624 | 0.614 | 0.614 |
| mid band, 8-26 GHz | 301 | 0.579 | 0.613 | 0.618 |
| **both bands, 2-26 GHz** | 602 | **0.646** | **0.664** | **0.668** |

Each band alone reaches only ~0.61; together they reach ~0.66. The two bands are
**complementary, not redundant**, and every narrowing costs AUC. So the answer to open
question 2 is no in both directions: the low band is not disposable, and neither is the
sub-5 GHz portion of it.

The methodological lesson is worth keeping: permutation importance measures what a
*fixed trained model* leans on, and with correlated inputs it will happily report zero
for a band whose removal costs five AUC points, because the model can re-route through
neighbouring frequencies. Retrain-and-ablate (`--band`, `--fmin`, `--fmax`) is the
decision-grade evidence. Do not cut hardware on an importance plot.

### Does the conclusion survive a different tumour contrast?

The contrast between normal and malignant skin is the most provisional number in the
repo, so the answer must not depend on getting it exactly right. `--contrast` scales the
whole departure from dermis; n = 6000, RF, nominal 5 deg phase error:

| contrast | A (Re/Im) | B (\|S11\|) | D (\|S11\|+scalars) |
|---|---|---|---|
| 0.0 (null) | 0.517 | 0.529 | 0.534 |
| 0.5 | 0.555 | 0.573 | 0.576 |
| 1.0 (nominal, n=12000) | 0.646 | 0.664 | 0.668 |
| 1.5 | 0.690 | 0.704 | 0.709 |

Performance is monotone in the physical parameter, and **B beats A at every level**. The
conclusion is a property of the phase-error regime, not of the assumed contrast.

### The null experiment

At `--contrast 0` the score is 0.517-0.534, slightly above chance. That is not leakage,
it is the model being honest: the lesion still *displaces epidermis* with dermis-like
material, and since every malignant sample has an inclusion but only
`benign_lesion_prob` of the benign ones do, a faint geometric signal survives wherever
the lesion crosses the epidermis.

Removing that asymmetry gives the definitive null -- `--contrast 0
--benign-lesion-prob 1.0`, so both classes are drawn from identical distributions
(`results/metrics_true_null.json`):

| A | B | D | `has_inclusion` |
|---|---|---|---|
| 0.4997 | 0.5014 | 0.4970 | 0.5000 (expected 0.5000) |

Exactly chance, to three decimals, on 6000 samples. Nothing in the generator, the
feature extraction, or the CV protocol can manufacture signal from nothing. Re-run this
after any change to `generate.py` -- it is the cheapest way to catch an accidental leak.

### Validity checks

| check | result |
|---|---|
| Forward model, 5 cases (spec section 7) | all pass |
| Dielectric anchors vs IFAC tool | 4 skin anchors within 0.07%; fat sigma WARN at 6.4%, documented |
| A/C information equivalence | pass, both models |
| Leakage: stack geometry only | AUC 0.4993 (chance) |
| Leakage: lesion geometry given inclusion | AUC 0.4909 (chance) |
| `has_inclusion` design invariant | AUC 0.7547 vs 0.7500 expected |
| **True null** (`--contrast 0 --benign-lesion-prob 1.0`) | **A 0.4997, B 0.5014, D 0.4970** |
| pytest suite | 9/9 pass |


---

## Known limitations, to state out loud in any writeup

- **Plane wave at normal incidence.** Ignores the spiral's near-field coupling and the
  fact that our aperture is comparable to a wavelength at the low end of the low band.
- **Infinite planar layers.** A real lesion has finite lateral extent, and a 4 mm lesion
  under a larger aperture is a partial-fill problem this model cannot represent. Expect
  the real contrast to be *diluted* relative to what this model predicts.
- **No antenna resonance structure in S11.** Real measured S11 contains the antenna's own
  match, which will dominate the raw magnitude and must be calibrated out or included as
  a known reference. Feature set D's notch features in particular would be measuring the
  antenna, not the tissue, if that calibration is not done.
- **Homogeneous layers.** Real dermis has vasculature and appendages.

**This model cannot validate the spiral.** It answers the information-content question,
which is what we need it for. Say so explicitly rather than letting a reviewer point it
out.

---

## Context to carry into any future session on this

- Device scope is a **classification label only**. Not localization, not depth, not
  margins. The reflectometry critique in the survey paper applies to us and we accept it.
- Competing prior art is Tavassolian's group at Stevens. Their 2022 *Scientific Reports*
  paper did in-vivo real-time HR-MMWI on 136 lesions from 71 patients and reported 97%
  sensitivity / 98% specificity with PCA into an MLP. Their 2019 *TMI* paper (the ex-vivo
  Mohs one) is the histopathology-facing system. **Our differentiator is not accuracy and
  we should not pretend it is** -- it is hardware cost and the absence of a scanning
  stage. Do not compare the numbers this repo produces to theirs; different model,
  different data, no clinical validation here at all.
- Existing antennas: self-complementary Archimedean spirals on RT/duroid 5880, low band
  2-12 GHz and mid band 8-26 GHz, simulated in CST Learning Edition under the 100k mesh
  cell cap.
- Front-end candidates: six-port reflectometer, ADL5961, ADF4371.
- CST full license pending through the faculty advisor. openEMS is the free FDTD fallback
  if full-wave becomes necessary. Ansys HFSS Student has a different limit structure than
  CST Learning and sometimes accepts a model CST rejects.

## Open questions

1. **What phase accuracy does a six-port at 26 GHz actually achieve after calibration?**
   This sets the realistic value for the phase-error sweep and right now we are guessing.
   The shaded band in `fig_phase_sweep.png` is that guess (2-10 deg rms) and it is drawn
   as a guess on purpose. Answering this turns the sweep from a curve into a decision.
2. **Is the sub-5 GHz portion of the low band contributing anything?** If permutation
   importance says no, the low-band antenna may be unnecessary and the BOM gets cheaper.
   See `results/fig_importance_vs_ghz.png`.
3. **Should benign lesions be a distinct third class rather than folded into the negative
   class?** Clinically that is the harder discrimination and the one that matters. The
   generator already labels them (`has_inclusion` in the metadata), so this is a
   three-class relabel in `train.py`, not a regeneration.
