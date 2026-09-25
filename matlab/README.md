# matlab/

The physics half of the experiment, ported from `../python` to **base MATLAB, with no
toolboxes required**.

**Start here:** open `demo_physics.m` and press Run. It checks the model, plots the
tissue properties, builds virtual patients, compares benign and malignant spectra, and
shows why phase error scrambles Re/Im but leaves |S11| untouched.

| file | what it does |
|---|---|
| `demo_physics.m` | guided walkthrough, run section by section |
| `tissue_params.m` | every dielectric number and its source (single source of truth) |
| `eps_c.m`, `sigma_eff.m` | complex permittivity and effective conductivity vs frequency |
| `layered_s11.m` | the forward model: impedance recursion, returns complex S11 |
| `carve_inclusion.m` | inserts a lesion without changing total thickness (anti-leakage) |
| `sample_stack.m` | one random virtual patient |
| `apply_measurement_noise.m` | receiver noise, gain error, phase noise and drift |
| `validate_model.m` | the physics checks; must print ALL CHECKS PASSED |
| `generate_dataset.m` | builds a labelled dataset and saves it to `../data/dataset_matlab.mat` |

## What's not here, and why

The machine-learning half (random forest, 5-fold cross-validation, ROC/AUC, the
phase-error sweep) needs the **Statistics and Machine Learning Toolbox**, which isn't
installed on this machine. Those parts stay in `../python`, which is also where all
the committed results come from. If you add the toolbox (UCSB's campus license usually
includes it, via Home > Add-Ons), the ML half can be ported next.

## Status

Verified in MATLAB R2025b. `validate_model` prints `ALL CHECKS PASSED`, and the numbers
match the Python version: identical anchor values (skin within 0.1%, documented WARN on
fat conductivity), the same 26.45 mm dermis penetration depth at 2 GHz, and the same
thin-layer convergence. `demo_physics` and `generate_dataset` run end to end, and the
noise model hits its target phase error.

Random numbers come from MATLAB's generator, so individual samples differ from Python's
even with the same seed. Distributions and conclusions should match, but not
sample-for-sample values.
