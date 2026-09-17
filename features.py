"""
features.py -- the four feature sets under comparison (spec section 4).

    A   Re(S11), Im(S11) at each frequency point
    B   |S11| only                                   <- what a scalar detector can measure
    C   |S11| and unwrapped phase                    <- information-equivalent to A
    D   |S11| plus engineered scalars: resonance frequencies, notch depths,
        band-averaged loss

A and C should score the same. They are the same information in different coordinates,
so a gap between them means the phase unwrapping or the scaling is broken and nothing
else in the run is trustworthy. train.py enforces that as an explicit check with a
tolerance (tree ensembles are not exactly invariant to a nonlinear change of
coordinates, so the check is "within tolerance", not "bit-identical").
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import find_peaks

FEATURE_SETS = ("A", "B", "C", "D")

FEATURE_SET_DESCRIPTIONS = {
    "A": "Re(S11), Im(S11) per frequency point",
    "B": "|S11| only (scalar detector front end)",
    "C": "|S11| and unwrapped phase (information-equivalent to A)",
    "D": "|S11| plus engineered scalars (resonances, notch depths, band-averaged loss)",
}

# Short labels for plot legends. The long descriptions above are for logs and JSON.
FEATURE_SET_SHORT = {
    "A": "A: Re/Im  (coherent)",
    "B": "B: |S11| only  (scalar)",
    "C": "C: |S11| + phase  (coherent)",
    "D": "D: |S11| + scalars  (scalar)",
}

N_NOTCHES = 3  # deepest notches kept per band


def _db(x: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-12))


def _unwrapped_phase(x: np.ndarray, band_slices: Sequence[slice]) -> np.ndarray:
    """Unwrap per band. Unwrapping across a band gap would be meaningless."""
    out = np.empty(x.shape, dtype=float)
    for sl in band_slices:
        out[:, sl] = np.unwrap(np.angle(x[:, sl]), axis=1)
    return out


def _engineered(
    x: np.ndarray, f: np.ndarray, band_slices: Sequence[slice]
) -> Tuple[np.ndarray, List[str]]:
    """Resonance frequencies, notch depths, band-averaged loss, per band.

    Notches are local minima of |S11| in dB, ranked by prominence. Missing notches are
    padded with sentinels (band centre frequency, 0 dB prominence) rather than NaN so
    tree models see a consistent, finite column.
    """
    n = x.shape[0]
    cols: List[str] = []
    blocks: List[np.ndarray] = []

    for bi, sl in enumerate(band_slices):
        fb = f[sl]
        mag_db = _db(x[:, sl])
        nb = fb.size
        f_ctr = 0.5 * (fb[0] + fb[-1])

        feats = np.zeros((n, 4 + 3 * N_NOTCHES), dtype=float)
        # Band-level scalars: these are what a power detector plus an integrator gives you.
        feats[:, 0] = mag_db.mean(axis=1)          # band-averaged loss
        feats[:, 1] = mag_db.std(axis=1)           # in-band variation
        feats[:, 2] = mag_db.min(axis=1)           # deepest point
        # linear slope of |S11| in dB across the band
        u = (fb - fb.mean()) / (fb[-1] - fb[0])
        feats[:, 3] = (mag_db * u[None, :]).sum(axis=1) / np.sum(u ** 2)

        for i in range(n):
            row = mag_db[i]
            pk, props = find_peaks(-row, prominence=1e-6)
            if pk.size:
                order = np.argsort(props["prominences"])[::-1][:N_NOTCHES]
                pk = pk[order]
                prom = props["prominences"][order]
            else:
                prom = np.zeros(0)
            for j in range(N_NOTCHES):
                base = 4 + 3 * j
                if j < pk.size:
                    k = int(pk[j])
                    feats[i, base + 0] = fb[k] / 1e9      # resonance frequency, GHz
                    feats[i, base + 1] = row[k]           # notch depth, dB
                    feats[i, base + 2] = prom[j]          # prominence, dB
                else:
                    feats[i, base + 0] = f_ctr / 1e9
                    feats[i, base + 1] = row.min()
                    feats[i, base + 2] = 0.0

        blocks.append(feats)
        b = f"b{bi}"
        cols += [f"{b}_mean_db", f"{b}_std_db", f"{b}_min_db", f"{b}_slope_db"]
        for j in range(N_NOTCHES):
            cols += [f"{b}_notch{j}_fghz", f"{b}_notch{j}_depth_db", f"{b}_notch{j}_prom_db"]

    return np.hstack(blocks), cols


def extract(
    s11: np.ndarray,
    f: np.ndarray,
    feature_set: str,
    band_slices: Optional[Sequence[slice]] = None,
    mag_in_db: bool = True,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Build a feature matrix.

    Returns (X, column_names, freq_of_column). freq_of_column is NaN for columns that
    are not tied to a single frequency; train.py uses it for the importance-vs-GHz plot.

    mag_in_db applies a log to the magnitude. It changes nothing for tree models (a
    monotone map of a single feature), but it keeps the columns on a sane scale for
    anything linear that gets bolted on later.
    """
    x = np.atleast_2d(np.asarray(s11))
    f = np.asarray(f, dtype=float)
    if band_slices is None:
        band_slices = [slice(0, f.size)]
    fs = feature_set.upper()

    if fs == "A":
        X = np.hstack([np.real(x), np.imag(x)]).astype(np.float32)
        cols = [f"re_{v/1e9:.4f}GHz" for v in f] + [f"im_{v/1e9:.4f}GHz" for v in f]
        fcol = np.concatenate([f, f])
    elif fs == "B":
        m = _db(x) if mag_in_db else np.abs(x)
        X = m.astype(np.float32)
        cols = [f"mag_{v/1e9:.4f}GHz" for v in f]
        fcol = f.copy()
    elif fs == "C":
        m = _db(x) if mag_in_db else np.abs(x)
        p = _unwrapped_phase(x, band_slices)
        X = np.hstack([m, p]).astype(np.float32)
        cols = [f"mag_{v/1e9:.4f}GHz" for v in f] + [f"phase_{v/1e9:.4f}GHz" for v in f]
        fcol = np.concatenate([f, f])
    elif fs == "D":
        m = _db(x) if mag_in_db else np.abs(x)
        eng, eng_cols = _engineered(x, f, band_slices)
        X = np.hstack([m, eng]).astype(np.float32)
        cols = [f"mag_{v/1e9:.4f}GHz" for v in f] + eng_cols
        # Engineered scalars are band-level, not point-level: no single frequency.
        fcol = np.concatenate([f, np.full(eng.shape[1], np.nan)])
    else:
        raise ValueError(f"unknown feature set {feature_set!r}; expected one of {FEATURE_SETS}")

    return X, cols, fcol


def uses_phase(feature_set: str) -> bool:
    """True if the feature set can see phase at all -- i.e. needs a coherent receiver."""
    return feature_set.upper() in ("A", "C")
