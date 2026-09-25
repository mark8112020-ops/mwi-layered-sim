"""
pytest wrapper around the section-7 validation checks, so they can run in CI as well as
by hand. `python validate.py` is the canonical entry point and prints the numbers;
this file just makes the same checks fail a test runner.

    python -m pytest tests -q          (needs pytest; validate.py does not)
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import generate  # noqa: E402
import tissue_params as tp  # noqa: E402
import validate  # noqa: E402
from layered import Layer, carve_inclusion, total_thickness  # noqa: E402


@pytest.mark.parametrize("name,fn", validate.CHECKS)
def test_forward_model(name, fn):
    ok, msg = fn(False)
    assert ok, f"{name}: {msg}"


def test_dielectric_anchors():
    """The dielectric tables must still reproduce the published anchor values."""
    failures = tp.check_anchors()
    assert not failures, "; ".join(failures)


def test_carving_preserves_total_thickness():
    """The anti-leakage invariant: a carved lesion must not move any interface below it."""
    f = tp.band_frequencies("low")
    rng = np.random.default_rng(0)
    base = [
        Layer("gap", tp.AIR.eps_c(f), 50e-6),
        Layer("sc", tp.STRATUM_CORNEUM.eps_c(f), 15e-6),
        Layer("epi", tp.EPIDERMIS.eps_c(f), 100e-6),
        Layer("derm", tp.DERMIS.eps_c(f), 2e-3),
        Layer("hypo", tp.HYPODERMIS.eps_c(f), 0.0),
    ]
    t0 = total_thickness(base)
    for _ in range(200):
        d = float(rng.uniform(0, 1.5e-3))
        t = float(rng.uniform(0.1e-3, 1.0e-3))
        out = carve_inclusion(base, d, t, "lesion", tp.MALIGNANT.eps_c(f), depth_origin_index=1)
        assert abs(total_thickness(out) - t0) < 1e-15
        assert any(l.name == "lesion" for l in out)


def test_generator_label_independence_of_geometry():
    """Thickness distributions must be identical in both classes."""
    d = generate.build_dataset(n_samples=1200, seed=3, progress_every=0)
    y = d["y"].astype(int)
    keys = [str(k) for k in d["meta_keys"]]
    meta = d["meta"]
    for k in ("t_total", "t_dermis", "t_epidermis", "t_coupling_gap", "hydration"):
        i = keys.index(k)
        a, b = meta[y == 1, i], meta[y == 0, i]
        # means within a quarter of a standard deviation of each other
        pooled = np.sqrt(0.5 * (a.var() + b.var()))
        assert abs(a.mean() - b.mean()) < 0.25 * pooled, f"{k} differs by class"


def test_phase_error_does_not_touch_magnitude():
    """Feature set B must be blind to the swept parameter except through additive noise."""
    f = tp.band_frequencies("low")
    d = generate.build_dataset(n_samples=40, bands=("low",), seed=5, progress_every=0)
    kw = dict(generate.DEFAULT_NOISE)
    kw.update(snr_db=np.inf)  # isolate the phase path
    a = generate.apply_measurement_noise(d["s11"], f, np.random.default_rng(1), **{**kw, "phase_err_deg": 0.0})
    b = generate.apply_measurement_noise(d["s11"], f, np.random.default_rng(1), **{**kw, "phase_err_deg": 45.0})
    assert np.allclose(np.abs(a), np.abs(b), rtol=1e-9), "phase error leaked into |S11|"
