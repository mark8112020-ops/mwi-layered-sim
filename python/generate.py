"""
generate.py -- dataset generation: sampling, measurement realism, dataset to npz.

Per sample (spec section 3):
  - layer thicknesses drawn within the table ranges
  - per-layer Cole-Cole parameters jittered multiplicatively around nominal (10% std)
    plus a shared per-sample hydration factor across the skin layers
  - lesion presence (the label), lesion top depth, lesion thickness
  - probe standoff / contact-pressure proxy as a thin variable air gap

The lesion is carved out of the existing stack (see layered.carve_inclusion), so total
stack thickness is identical in both classes and cannot leak the label. A configurable
fraction of BENIGN samples receives a benign-lesion inclusion with intermediate
properties, so the classifier cannot win by detecting "any inclusion".

S11 is stored CLEAN. Measurement corruption (spec section 3.1) is applied downstream by
apply_measurement_noise(), because sweep_phase_error.py needs to re-corrupt the same
physics thousands of times and regenerating the stacks each time would be wasteful.

    python generate.py --n 12000 --out data/dataset.npz
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

import tissue_params as tp
from layered import Layer, carve_inclusion, s11, total_thickness

# --------------------------------------------------------------------------------------
# Default measurement-realism parameters (spec section 3.1)
# --------------------------------------------------------------------------------------
# A noiseless comparison favours complex S11 automatically -- complex data is a strict
# superset of magnitude data, so that result is trivially true and tells us nothing.
# These defaults are what make the comparison mean something.
DEFAULT_NOISE = dict(
    snr_db=30.0,             # additive receiver noise, referenced to mean |S11|^2 in band
    mag_gain_std=0.02,       # per-sweep scalar gain error, 2%
    mag_drift_std=0.01,      # smooth in-band magnitude ripple, 1%
    phase_err_deg=5.0,       # TOTAL rms phase error; split below
    phase_noise_frac=0.40,   # share of phase error that is iid point-to-point
    phase_drift_frac=0.917,  # share that is smooth/systematic (0.4^2+0.917^2 = 1.0)
    delay_err_frac=0.70,     # share of the smooth part that is a residual electrical delay
)
# Phase is the fragile quantity in a cheap front end: cable flex, temperature and
# imperfect calibration hit it far harder than magnitude, and the dominant residual
# after a one-port calibration is an electrical-delay (linear-in-frequency) term. That
# is modelled explicitly by delay_err_frac rather than as white phase noise.


@dataclass
class Sample:
    s11: np.ndarray            # complex, shape (n_freq,) -- clean
    label: int                 # 1 = malignant, 0 = benign
    meta: Dict[str, float] = field(default_factory=dict)
    layers: Optional[List[Layer]] = None


# --------------------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------------------
def sample_stack(
    rng: np.random.Generator,
    f_hz: np.ndarray,
    label: Optional[int] = None,
    benign_lesion_prob: float = 0.5,
    jitter_std: float = tp.JITTER_STD,
    hydration_std: float = tp.HYDRATION_STD,
    contrast_scale: float = 1.0,
    keep_layers: bool = False,
) -> Sample:
    """Draw one physiological realisation and compute its clean S11."""
    if label is None:
        label = int(rng.integers(0, 2))

    # Shared hydration / body-site / age factor across the skin layers. Label-independent
    # by construction. Section 2.2: this moves eps' by more than the tumour contrast in
    # parts of the band, and a model without it produces a fake 99%.
    hyd = 1.0
    if hydration_std > 0:
        s = np.sqrt(np.log(1.0 + hydration_std ** 2))
        hyd = float(rng.lognormal(-0.5 * s * s, s))

    materials: List[Tuple[str, np.ndarray]] = []
    thicknesses: List[float] = []
    meta: Dict[str, float] = {"hydration": hyd}
    for spec in tp.STACK:
        # The coupling gap is air: no physiological jitter, and no hydration coupling.
        is_tissue = spec.material.name not in ("air",)
        mat = tp.jitter_dielectric(
            spec.material,
            rng,
            std=jitter_std if is_tissue else 0.0,
            common_factor=hyd if is_tissue else 1.0,
        )
        t = float(rng.uniform(spec.t_min, spec.t_max))
        materials.append((spec.name, mat.eps_c(f_hz)))
        thicknesses.append(t)
        meta[f"t_{spec.name}"] = t

    layers = [Layer(n, e, t) for (n, e), t in zip(materials, thicknesses)]
    t_no_lesion = total_thickness(layers)

    # --- inclusion -------------------------------------------------------------------
    mal_d, ben_d = tp.build_lesion_dielectrics(contrast_scale)
    has_inclusion = bool(label == 1 or rng.random() < benign_lesion_prob)
    meta["has_inclusion"] = float(has_inclusion)
    meta["lesion_top_depth"] = 0.0
    meta["lesion_thickness"] = 0.0

    if has_inclusion:
        # Geometry is drawn from the SAME distribution for benign and malignant
        # inclusions, so geometry carries no label information either.
        skin_thickness = sum(
            meta[f"t_{n}"] for n in ("stratum_corneum", "epidermis", "dermis")
        )
        d_top = float(rng.uniform(*tp.LESION_TOP_DEPTH_RANGE))
        d_top = min(d_top, max(0.0, skin_thickness - 0.2e-3))
        t_les = float(rng.uniform(*tp.LESION_THICKNESS_RANGE))
        t_les = min(t_les, skin_thickness - d_top)  # keep the lesion inside the skin

        mat = mal_d if label == 1 else ben_d
        # Lesion material inherits the same hydration factor -- otherwise hydration
        # itself becomes a label proxy.
        mat = tp.jitter_dielectric(mat, rng, std=jitter_std, common_factor=hyd)
        layers = carve_inclusion(
            layers,
            top_depth=d_top,
            thickness=t_les,
            inclusion_name="lesion",
            inclusion_eps=mat.eps_c(f_hz),
            depth_origin_index=1,  # depths measured from the top of the stratum corneum
        )
        meta["lesion_top_depth"] = d_top
        meta["lesion_thickness"] = t_les

    # Leakage tripwire: carving must not have changed the total stack thickness.
    t_after = total_thickness(layers)
    if abs(t_after - t_no_lesion) > 1e-12:
        raise AssertionError(
            f"carve_inclusion changed total thickness by {t_after - t_no_lesion:.3e} m; "
            "that is a label leak"
        )
    meta["t_total"] = t_after

    eps_ref = tp.REFERENCE_MEDIUM.eps_c(f_hz)
    sig = s11(layers, f_hz, eps_ref, check=False)
    return Sample(s11=sig, label=int(label), meta=meta, layers=layers if keep_layers else None)


# --------------------------------------------------------------------------------------
# Measurement realism (spec section 3.1)
# --------------------------------------------------------------------------------------
def apply_measurement_noise(
    s11_clean: np.ndarray,
    f_hz: np.ndarray,
    rng: np.random.Generator,
    snr_db: float = DEFAULT_NOISE["snr_db"],
    mag_gain_std: float = DEFAULT_NOISE["mag_gain_std"],
    mag_drift_std: float = DEFAULT_NOISE["mag_drift_std"],
    phase_err_deg: float = DEFAULT_NOISE["phase_err_deg"],
    phase_noise_frac: float = DEFAULT_NOISE["phase_noise_frac"],
    phase_drift_frac: float = DEFAULT_NOISE["phase_drift_frac"],
    delay_err_frac: float = DEFAULT_NOISE["delay_err_frac"],
    band_slices: Optional[List[slice]] = None,
) -> np.ndarray:
    """Corrupt clean S11 the way a real front end would.

    s11_clean: (n_samples, n_freq) complex. Errors are drawn per sample (per sweep),
    which is the right granularity: cable flex and temperature are constant within one
    sweep and vary between them.

    Both feature sets see the same corruption. Magnitude errors are small; phase errors
    are larger and dominated by a smooth residual-delay term.
    """
    x = np.atleast_2d(np.asarray(s11_clean, dtype=complex))
    n, nf = x.shape
    f = np.asarray(f_hz, dtype=float)
    if band_slices is None:
        band_slices = [slice(0, nf)]

    mag = np.abs(x)
    pha = np.angle(x)

    # --- magnitude gain + smooth drift ----------------------------------------------
    gain = 1.0 + rng.normal(0.0, mag_gain_std, size=(n, 1)) if mag_gain_std > 0 else 1.0
    drift = np.zeros((n, nf))
    if mag_drift_std > 0:
        for sl in band_slices:
            nb = len(range(*sl.indices(nf)))
            u = np.linspace(0.0, 1.0, nb)[None, :]
            # low-order smooth ripple: 3 random harmonics, unit-variance normalised
            comp = np.zeros((n, nb))
            for k in (1, 2, 3):
                comp += rng.normal(0, 1, (n, 1)) * np.cos(2 * np.pi * k * u + rng.uniform(0, 2 * np.pi, (n, 1)))
            drift[:, sl] = comp / np.sqrt(1.5)
    mag_c = mag * gain * (1.0 + mag_drift_std * drift)

    # --- phase noise + drift ----------------------------------------------------------
    ph_rad = np.deg2rad(phase_err_deg)
    pha_c = pha.copy()
    if ph_rad > 0:
        # iid point-to-point component
        pha_c = pha_c + rng.normal(0.0, ph_rad * phase_noise_frac, size=(n, nf))
        # smooth systematic component, per band
        smooth = np.zeros((n, nf))
        for sl in band_slices:
            fb = f[sl]
            nb = fb.size
            u = (fb - fb[0]) / (fb[-1] - fb[0])

            # Residual electrical delay: phase error linear in frequency. This is the
            # dominant leftover after a one-port calibration and what cable flex
            # actually produces. `shape` is normalised deterministically to unit rms so
            # the delay/ripple power split stays at delay_err_frac on average; the
            # per-sweep amplitude is the random coefficient in front of it.
            shape = u - u.mean()
            shape = shape / np.sqrt(np.mean(shape ** 2))
            delay_term = rng.normal(0, 1, (n, 1)) * shape[None, :]

            # Slower drift/ripple: temperature, connector repeatability. Two harmonics
            # with unit-variance coefficients have rms sqrt(2 * 1/2) = 1 over the band.
            ripple = np.zeros((n, nb))
            for k in (1, 2):
                ripple += rng.normal(0, 1, (n, 1)) * np.cos(
                    2 * np.pi * k * u[None, :] + rng.uniform(0, 2 * np.pi, (n, 1))
                )

            smooth[:, sl] = (
                np.sqrt(delay_err_frac) * delay_term + np.sqrt(1.0 - delay_err_frac) * ripple
            )
            # Force exactly phase_err_deg rms per sweep, per band, so the swept parameter
            # means what its name says.
            rms = np.sqrt(np.mean(smooth[:, sl] ** 2, axis=1, keepdims=True))
            rms[rms == 0] = 1.0
            smooth[:, sl] /= rms
        pha_c = pha_c + ph_rad * phase_drift_frac * smooth

    y = mag_c * np.exp(1j * pha_c)

    # --- additive receiver noise ------------------------------------------------------
    if np.isfinite(snr_db):
        for sl in band_slices:
            p_sig = np.mean(np.abs(y[:, sl]) ** 2)
            sigma = np.sqrt(p_sig / (10 ** (snr_db / 10.0)) / 2.0)
            nb = len(range(*sl.indices(nf)))
            y[:, sl] += sigma * (rng.normal(0, 1, (n, nb)) + 1j * rng.normal(0, 1, (n, nb)))
    return y


# --------------------------------------------------------------------------------------
# Dataset build
# --------------------------------------------------------------------------------------
def build_dataset(
    n_samples: int = 12000,
    bands: Tuple[str, ...] = ("low", "mid"),
    seed: int = 20260801,
    benign_lesion_prob: float = 0.5,
    jitter_std: float = tp.JITTER_STD,
    hydration_std: float = tp.HYDRATION_STD,
    contrast_scale: float = 1.0,
    progress_every: int = 1000,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    f_parts = [tp.band_frequencies(b) for b in bands]
    f = np.concatenate(f_parts)
    band_slices, i = [], 0
    for part in f_parts:
        band_slices.append((i, i + part.size))
        i += part.size

    # Balanced by construction, then shuffled so ordering carries nothing.
    labels = np.concatenate([np.ones(n_samples // 2, int), np.zeros(n_samples - n_samples // 2, int)])
    rng.shuffle(labels)

    S = np.empty((n_samples, f.size), dtype=np.complex64)
    meta_keys: List[str] = []
    meta_rows: List[List[float]] = []
    t0 = time.time()
    for k in range(n_samples):
        smp = sample_stack(
            rng, f, label=int(labels[k]),
            benign_lesion_prob=benign_lesion_prob,
            jitter_std=jitter_std,
            hydration_std=hydration_std,
            contrast_scale=contrast_scale,
        )
        S[k] = smp.s11.astype(np.complex64)
        if not meta_keys:
            meta_keys = sorted(smp.meta.keys())
        meta_rows.append([smp.meta[key] for key in meta_keys])
        if progress_every and (k + 1) % progress_every == 0:
            el = time.time() - t0
            print(f"  {k+1}/{n_samples}  ({el:.1f}s, {(k+1)/el:.0f} samples/s)", flush=True)

    return dict(
        s11=S,
        y=labels.astype(np.int8),
        f=f,
        bands=np.array(bands),
        band_edges=np.array(band_slices),
        meta=np.array(meta_rows, dtype=np.float64),
        meta_keys=np.array(meta_keys),
        config=json.dumps(
            dict(
                n_samples=n_samples, bands=list(bands), seed=seed,
                benign_lesion_prob=benign_lesion_prob, jitter_std=jitter_std,
                hydration_std=hydration_std, contrast_scale=contrast_scale,
                points_per_band={b: tp.BANDS[b][2] for b in bands},
                model="1D layered, plane wave, normal incidence",
            )
        ),
    )


def load_dataset(path: str) -> Dict[str, object]:
    d = np.load(path, allow_pickle=False)
    out = {k: d[k] for k in d.files}
    return out


def band_slices_from(data: Dict[str, object]) -> List[slice]:
    return [slice(int(a), int(b)) for a, b in np.asarray(data["band_edges"])]


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate the layered-media S11 dataset.")
    ap.add_argument("--n", type=int, default=12000, help="number of samples (spec: 10k-20k)")
    ap.add_argument("--out", default="data/dataset.npz")
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--bands", nargs="+", default=["low", "mid"], choices=list(tp.BANDS))
    ap.add_argument("--benign-lesion-prob", type=float, default=0.5)
    ap.add_argument("--jitter-std", type=float, default=tp.JITTER_STD)
    ap.add_argument("--hydration-std", type=float, default=tp.HYDRATION_STD)
    ap.add_argument("--contrast", type=float, default=1.0,
                    help="scale the normal->malignant dielectric contrast (0 = null experiment)")
    ap.add_argument("--skip-validation", action="store_true",
                    help="do not do this; the spec says stop if validation fails")
    args = ap.parse_args()

    if not args.skip_validation:
        import validate
        if not validate.run_all(verbose=False):
            print("\nRefusing to generate a dataset on a forward model that fails validation.")
            return 1
        print()

    print(f"generating {args.n} samples over bands {args.bands} ...")
    data = build_dataset(
        n_samples=args.n, bands=tuple(args.bands), seed=args.seed,
        benign_lesion_prob=args.benign_lesion_prob, jitter_std=args.jitter_std,
        hydration_std=args.hydration_std, contrast_scale=args.contrast,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **data)
    mb = os.path.getsize(args.out) / 1e6
    print(f"wrote {args.out}  ({mb:.1f} MB, {data['s11'].shape[0]} x {data['s11'].shape[1]} complex)")
    print(f"class balance: {int(np.sum(data['y']))} malignant / {int(np.sum(1 - data['y']))} benign")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
