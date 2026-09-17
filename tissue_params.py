"""
tissue_params.py -- single source of truth for every dielectric number in this repo.

Rule from the handoff spec: no dielectric constant is allowed to live anywhere else.
Every entry carries a `source` string. Anything not traceable to a published table is
marked `provisional=True` and is listed by `python tissue_params.py --todo`, which
prints a TODO_SOURCE report. If the contrast values here are wrong, every accuracy
number downstream is meaningless in a way that will not be obvious, so this file is
the one to audit first.

Dispersion model
----------------
N-pole Cole-Cole with a static (ionic) conductivity term, time convention exp(+jwt):

    eps_c(w) = eps_inf + sum_n  d_eps_n / (1 + (j*w*tau_n)**(1 - alpha_n))
                       - 1j * sigma_s / (w * eps_0)

alpha_n = 0 collapses pole n to single-pole Debye, so the "Cole-Cole or single-pole
Debye" choice in the spec is the same code path.

Verification
------------
`python tissue_params.py` prints eps' and effective conductivity at anchor
frequencies next to the published values in ANCHORS. Those anchors are the check
that the parameters were transcribed correctly -- run it after touching anything.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

EPS_0 = 8.8541878128e-12  # F/m
C0 = 299792458.0  # m/s
ETA_0 = 376.730313668  # ohm

# Unit helpers, used so the tables below read like the papers they came from.
ps = 1e-12
ns = 1e-9
us = 1e-6
ms = 1e-3
um = 1e-6
mm = 1e-3


# --------------------------------------------------------------------------------------
# Dielectric model containers
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ColePole:
    """One Cole-Cole dispersion pole. alpha=0 -> single-pole Debye."""

    d_eps: float
    tau: float  # seconds
    alpha: float = 0.0


@dataclass(frozen=True)
class Dielectric:
    """Frequency-dependent complex permittivity of one material."""

    name: str
    eps_inf: float
    poles: Tuple[ColePole, ...]
    sigma_s: float  # S/m, static/ionic conductivity
    source: str
    provisional: bool = False
    notes: str = ""

    def eps_c(self, f_hz: np.ndarray) -> np.ndarray:
        """Complex relative permittivity eps' - j*eps'' at frequencies f_hz."""
        f = np.asarray(f_hz, dtype=float)
        w = 2.0 * np.pi * f
        eps = np.full(f.shape, self.eps_inf, dtype=complex)
        for p in self.poles:
            if p.d_eps == 0.0:
                continue
            eps = eps + p.d_eps / (1.0 + (1j * w * p.tau) ** (1.0 - p.alpha))
        with np.errstate(divide="ignore", invalid="ignore"):
            eps = eps - 1j * self.sigma_s / (w * EPS_0)
        return eps

    def sigma_eff(self, f_hz: np.ndarray) -> np.ndarray:
        """Effective conductivity S/m = eps'' * w * eps_0 (what papers usually tabulate)."""
        f = np.asarray(f_hz, dtype=float)
        return -np.imag(self.eps_c(f)) * (2.0 * np.pi * f) * EPS_0


def scale_dielectric(
    base: Dielectric,
    name: str,
    source: str,
    d_eps_scale: float = 1.0,
    tau_scale: float = 1.0,
    eps_inf_scale: float = 1.0,
    sigma_add: float = 0.0,
    pole_indices: Optional[Tuple[int, ...]] = None,
    provisional: bool = True,
    notes: str = "",
) -> Dielectric:
    """Derive a material from another by an explicit, auditable contrast.

    Used for the lesion materials: instead of dropping in opaque numbers, the tumour is
    defined as "dermis, but with this much more water and this much more ionic loss",
    so the contrast that drives the whole experiment is visible in one place and easy
    to sweep (see CONTRAST_SCALE / build_lesion_dielectrics).
    """
    idx = pole_indices if pole_indices is not None else tuple(range(len(base.poles)))
    poles = tuple(
        replace(p, d_eps=p.d_eps * d_eps_scale, tau=p.tau * tau_scale) if i in idx else p
        for i, p in enumerate(base.poles)
    )
    return Dielectric(
        name=name,
        eps_inf=base.eps_inf * eps_inf_scale,
        poles=poles,
        sigma_s=base.sigma_s + sigma_add,
        source=source,
        provisional=provisional,
        notes=notes,
    )


# --------------------------------------------------------------------------------------
# Reference materials
# --------------------------------------------------------------------------------------
GABRIEL = (
    "Gabriel, Gabriel & Corthout, 'The dielectric properties of biological tissues: "
    "III. Parametric models for the dielectric spectrum of tissues', Phys. Med. Biol. "
    "41(11), 2271-2293, 1996 (4-Cole-Cole table; also IFAC/Andreuccetti online tool)."
)
MIRBEIK = (
    "Mirbeik-Sabzevari, Ashinoff & Tavassolian, 'Ultra-wideband millimeter-wave "
    "dielectric characteristics of freshly excised normal and malignant human skin "
    "tissues', IEEE Trans. Biomed. Eng. 65(5), 1320-1329, 2018."
)
ROGERS = "Rogers Corp. RT/duroid 5870/5880 datasheet (eps_r 2.20 +/- 0.02, tan_d 0.0009 @ 10 GHz)."

# --- coupling medium / antenna substrate: sets the reference impedance ---------------
DUROID_5880 = Dielectric(
    name="rt_duroid_5880",
    eps_inf=2.2,
    poles=(),
    sigma_s=2.2 * 0.0009 * 2 * math.pi * 10e9 * EPS_0,  # tan_d 0.0009 referenced at 10 GHz
    source=ROGERS,
    provisional=False,
    notes="Reference medium at the probe face. Non-dispersive over 2-26 GHz to within datasheet spread.",
)

AIR = Dielectric(
    name="air",
    eps_inf=1.00059,
    poles=(),
    sigma_s=0.0,
    source="Standard atmosphere.",
    provisional=False,
    notes="Probe standoff / imperfect contact gap.",
)

# --- Gabriel 4-Cole-Cole tissues ------------------------------------------------------
SKIN_DRY = Dielectric(
    name="skin_dry",
    eps_inf=4.0,
    poles=(
        ColePole(32.0, 7.23 * ps, 0.00),
        ColePole(1100.0, 32.48 * ns, 0.20),
        ColePole(0.0, 159.15 * us, 0.20),
        ColePole(0.0, 15.915 * ms, 0.20),
    ),
    sigma_s=0.0002,
    source=GABRIEL + " Entry 'Skin (Dry)'.",
    provisional=False,
)

SKIN_WET = Dielectric(
    name="skin_wet",
    eps_inf=4.0,
    poles=(
        ColePole(39.0, 7.96 * ps, 0.10),
        ColePole(280.0, 79.58 * ns, 0.00),
        ColePole(3.0e4, 1.59 * us, 0.16),
        ColePole(3.0e4, 1.59 * ms, 0.20),
    ),
    sigma_s=0.0004,
    source=GABRIEL + " Entry 'Skin (Wet)'.",
    provisional=False,
)

FAT_NOT_INFILTRATED = Dielectric(
    name="fat_not_infiltrated",
    eps_inf=2.5,
    poles=(
        ColePole(3.0, 7.96 * ps, 0.20),
        ColePole(15.0, 15.92 * ns, 0.10),
        ColePole(3.3e4, 159.15 * us, 0.05),
        ColePole(1.0e7, 15.915 * ms, 0.01),
    ),
    sigma_s=0.01,
    source=GABRIEL + " Entry 'Fat (Not Infiltrated)'.",
    provisional=False,
)

FAT_INFILTRATED = Dielectric(
    name="fat_infiltrated",
    eps_inf=2.5,
    poles=(
        ColePole(9.0, 7.96 * ps, 0.20),
        ColePole(35.0, 15.92 * ns, 0.10),
        ColePole(3.3e4, 159.15 * us, 0.05),
        ColePole(1.0e7, 15.915 * ms, 0.01),
    ),
    sigma_s=0.035,
    source=GABRIEL + " Entry 'Fat (Infiltrated)'. TODO_SOURCE: unverifiable against the "
    "IFAC web tool, which has no 'Fat (Infiltrated)' entry -- its 'BreastFat' is a "
    "different tissue (eps' 5.1467, sigma 0.13704 at 2.45 GHz, versus 10.82 / 0.268 "
    "here). Check this row against Gabriel 1996 Table 1 directly before relying on it.",
    provisional=True,
    notes="Not used in STACK. Kept as the wetter-fat alternative for the hypodermis.",
)

# --- stratum corneum ------------------------------------------------------------------
# TODO_SOURCE: Gabriel's tables have no stratum-corneum entry. The values below are a
# low-water Debye consistent with the SC literature (eps' ~ 3-4, tan_d < 0.1 across
# 2-26 GHz, e.g. Alekseev & Ziskin 2007) but they are OUR fit, not a transcription.
# At 10-20 um the SC is electrically thin over this band, so the classifier outcome is
# insensitive to it -- confirm that with the thin-layer validation case before caring.
STRATUM_CORNEUM = Dielectric(
    name="stratum_corneum",
    eps_inf=2.0,
    poles=(ColePole(2.0, 8.0 * ps, 0.0),),
    sigma_s=0.002,
    source="TODO_SOURCE: our low-water Debye fit; cf. Alekseev & Ziskin, Bioelectromagnetics 28(5), 2007.",
    provisional=True,
    notes="Electrically thin at 2-26 GHz; low sensitivity expected.",
)

# --- viable epidermis -----------------------------------------------------------------
# TODO_SOURCE: 'Skin (Dry)' is the usual stand-in for the epidermis-dominated response
# and 'Skin (Wet)' for the dermis-dominated one. That mapping is convention, not a
# measurement of separated layers.
EPIDERMIS = replace(
    SKIN_DRY,
    name="epidermis",
    source=GABRIEL + " Entry 'Skin (Dry)' used as the viable-epidermis stand-in. TODO_SOURCE: layer mapping is convention.",
    provisional=True,
)

DERMIS = replace(
    SKIN_WET,
    name="dermis",
    source=GABRIEL + " Entry 'Skin (Wet)' used as the dermis stand-in. TODO_SOURCE: layer mapping is convention.",
    provisional=True,
    notes="Dominant layer in our band.",
)

HYPODERMIS = replace(
    FAT_NOT_INFILTRATED,
    name="hypodermis",
    source=GABRIEL + " Entry 'Fat (Not Infiltrated)'. Subcutaneous fat; semi-infinite in this model.",
    provisional=False,
)


# --------------------------------------------------------------------------------------
# Lesion materials -- the contrast the whole experiment rides on
# --------------------------------------------------------------------------------------
# Mirbeik-Sabzevari 2018 report malignant skin (BCC/SCC) as higher in both eps' and
# eps'' than normal skin across 5-50 GHz, the accepted explanation being higher water
# and ionic content. The numbers below are chosen to land near their reported contrast
# (order +15% in eps', +20-25% in eps'') rather than transcribed pointwise from their
# figures, so they are provisional and flagged as such. `--contrast` re-scales them for
# sensitivity analysis: report accuracy at 0.5x and 1.5x alongside the nominal, because
# a conclusion that only survives at 1.0x is not a conclusion.
TUMOR_D_EPS_SCALE = 1.16
TUMOR_TAU_SCALE = 0.97
TUMOR_SIGMA_ADD = 0.35  # S/m
BENIGN_LESION_FRACTION = 0.5  # benign lesion sits this far along the normal->malignant path


def build_lesion_dielectrics(contrast_scale: float = 1.0) -> Tuple[Dielectric, Dielectric]:
    """Return (malignant, benign_lesion) dielectrics at the requested contrast scale.

    contrast_scale multiplies the *departure* from dermis, so 0.0 makes the tumour
    dielectrically identical to dermis (a useful null: accuracy must collapse to chance).
    """
    cs = float(contrast_scale)
    mal = scale_dielectric(
        DERMIS,
        name="malignant_lesion",
        source=MIRBEIK + " TODO_SOURCE: contrast fitted to the reported trend, not transcribed pointwise.",
        d_eps_scale=1.0 + (TUMOR_D_EPS_SCALE - 1.0) * cs,
        tau_scale=1.0 + (TUMOR_TAU_SCALE - 1.0) * cs,
        sigma_add=TUMOR_SIGMA_ADD * cs,
        pole_indices=(0,),
        notes="Higher water + ionic content than dermis.",
    )
    ben = scale_dielectric(
        DERMIS,
        name="benign_lesion",
        source="TODO_SOURCE: intermediate material, no direct measurement. Placed at "
        f"{BENIGN_LESION_FRACTION:.2f} of the normal->malignant contrast.",
        d_eps_scale=1.0 + (TUMOR_D_EPS_SCALE - 1.0) * cs * BENIGN_LESION_FRACTION,
        tau_scale=1.0 + (TUMOR_TAU_SCALE - 1.0) * cs * BENIGN_LESION_FRACTION,
        sigma_add=TUMOR_SIGMA_ADD * cs * BENIGN_LESION_FRACTION,
        pole_indices=(0,),
        notes="Exists so the classifier cannot win by detecting 'any inclusion'.",
    )
    return mal, ben


MALIGNANT, BENIGN_LESION = build_lesion_dielectrics(1.0)


# --------------------------------------------------------------------------------------
# Layer stack geometry
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LayerSpec:
    name: str
    material: Dielectric
    t_min: float  # m
    t_max: float  # m
    semi_infinite: bool = False
    source: str = ""


# Thickness ranges are the table in section 2 of the handoff spec.
STACK: Tuple[LayerSpec, ...] = (
    LayerSpec(
        "coupling_gap", AIR, 0.0, 150 * um,
        source="Probe standoff / contact-pressure proxy. Range is our assumption; "
               "TODO_SOURCE: measure achievable contact repeatability on the real probe.",
    ),
    LayerSpec(
        "stratum_corneum", STRATUM_CORNEUM, 10 * um, 20 * um,
        source="Handoff spec table (10-20 um).",
    ),
    LayerSpec(
        "epidermis", EPIDERMIS, 50 * um, 150 * um,
        source="Handoff spec table (50-150 um).",
    ),
    LayerSpec(
        "dermis", DERMIS, 1.0 * mm, 3.0 * mm,
        source="Handoff spec table (1.0-3.0 mm). Dominant layer in our band.",
    ),
    LayerSpec(
        "hypodermis", HYPODERMIS, 4.0 * mm, 10.0 * mm, semi_infinite=True,
        source="Handoff spec table (4-10 mm); effectively semi-infinite above 2 GHz.",
    ),
)

REFERENCE_MEDIUM = DUROID_5880  # eta_ref at the probe face

# Lesion geometry. Depths are measured from the skin surface (top of stratum corneum),
# so they are independent of the coupling gap.
LESION_TOP_DEPTH_RANGE = (0.05 * mm, 1.50 * mm)
LESION_THICKNESS_RANGE = (0.30 * mm, 2.50 * mm)

# Physiological variation (section 2.2). Two components:
#   - a per-sample hydration factor shared by all skin layers (body site, age, hydration)
#   - independent per-layer multiplicative jitter
# Both are label-independent by construction; that is what keeps section 5's leakage
# guard honest. HYDRATION_STD is deliberately large enough to move eps' by more than the
# tumour contrast in parts of the band -- that is the physical situation, and a model
# without it produces a fake 99%.
JITTER_STD = 0.10
HYDRATION_STD = 0.08
JITTERED_FIELDS = ("eps_inf", "d_eps", "tau", "sigma_s")

# Frequency bands: the two existing self-complementary Archimedean spirals.
BANDS: Dict[str, Tuple[float, float, int]] = {
    "low": (2.0e9, 12.0e9, 301),
    "mid": (8.0e9, 26.0e9, 301),
}


def band_frequencies(band: str) -> np.ndarray:
    f0, f1, n = BANDS[band]
    return np.linspace(f0, f1, n)


# --------------------------------------------------------------------------------------
# Per-sample sampling of materials
# --------------------------------------------------------------------------------------
def jitter_dielectric(
    d: Dielectric,
    rng: np.random.Generator,
    std: float = JITTER_STD,
    common_factor: float = 1.0,
) -> Dielectric:
    """Multiplicative log-normal jitter around nominal, plus a shared hydration factor.

    Log-normal rather than Gaussian so a 10% std cannot produce a negative permittivity.
    `common_factor` scales the water-driven quantities (d_eps and sigma_s) only; eps_inf
    and tau are the non-hydration-driven parameters and get independent jitter alone.
    """
    if std <= 0.0 and common_factor == 1.0:
        return d

    def lg() -> float:
        if std <= 0.0:
            return 1.0
        sigma = math.sqrt(math.log(1.0 + std * std))
        return float(rng.lognormal(mean=-0.5 * sigma * sigma, sigma=sigma))

    poles = tuple(
        ColePole(
            d_eps=p.d_eps * lg() * common_factor if "d_eps" in JITTERED_FIELDS else p.d_eps,
            tau=p.tau * lg() if "tau" in JITTERED_FIELDS else p.tau,
            alpha=p.alpha,
        )
        for p in d.poles
    )
    return replace(
        d,
        eps_inf=d.eps_inf * (lg() if "eps_inf" in JITTERED_FIELDS else 1.0),
        poles=poles,
        sigma_s=d.sigma_s * (lg() if "sigma_s" in JITTERED_FIELDS else 1.0) * common_factor,
    )


# --------------------------------------------------------------------------------------
# Verification / reporting
# --------------------------------------------------------------------------------------
# Published anchors used to check that the tables above were transcribed correctly.
# (tissue key, f_Hz, eps', sigma_eff S/m, source)
IFAC_TOOL = (
    "Andreuccetti, Fossi & Petrucci, 'An Internet resource for the calculation of the "
    "dielectric properties of body tissues in the frequency range 10 Hz - 100 GHz', "
    "IFAC-CNR, Florence, 1997. https://niremf.ifac.cnr.it/tissprop/ "
    "Values read off the single-tissue/single-frequency query on 2026-08-01."
)


@dataclass(frozen=True)
class Anchor:
    """A published (tissue, frequency) -> (eps', sigma) point used to check transcription.

    Tolerances are tight on purpose. These are not measurements with error bars; they
    are the output of the same parametric model, so a correct transcription should
    reproduce them to the printed precision. A percent-level miss means a parameter is
    wrong, not that the tolerance needs loosening.

    known_mismatch documents an anchor we already know does not reproduce. Those are
    reported as WARN rather than FAIL, so the discrepancy stays visible in every run
    instead of being buried under a widened tolerance.
    """

    tissue: str
    f_hz: float
    eps_prime: float
    sigma: float
    source: str
    tol_eps: float = 0.01
    tol_sigma: float = 0.02
    known_mismatch: str = ""


ANCHORS: Tuple[Anchor, ...] = (
    Anchor("skin_dry", 2.45e9, 38.007, 1.4640, IFAC_TOOL),
    Anchor("skin_dry", 10.0e9, 31.290, 8.0138, IFAC_TOOL),
    Anchor("skin_wet", 2.45e9, 42.853, 1.5919, IFAC_TOOL),
    Anchor("skin_wet", 10.0e9, 33.528, 8.9510, IFAC_TOOL),
    Anchor(
        "fat_not_infiltrated", 2.45e9, 5.2801, 0.10452, IFAC_TOOL,
        known_mismatch=(
            "eps' reproduces to 0.01% but sigma comes out ~6% low (0.098 vs 0.1045). "
            "An exact eps' match means eps_inf and the first pole are right, so the "
            "error is in the low-frequency poles or sigma_s, which only matter through "
            "the residual loss. TODO_SOURCE: re-transcribe the Fat (Not Infiltrated) "
            "row from Gabriel 1996 Table 1 rather than from the IFAC web output. "
            "Impact is small -- this is the semi-infinite backing layer, well below the "
            "lesion -- but do not use fat loss for anything load-bearing until it is fixed."
        ),
    ),
    # No anchor for fat_infiltrated: the IFAC tool has no 'Fat (Infiltrated)' entry, and
    # its 'BreastFat' is a DIFFERENT tissue (eps' 5.1467, sigma 0.13704 at 2.45 GHz), not
    # this one. Checking fat_infiltrated against our own output would be circular, so it
    # is left unchecked and flagged provisional instead. It is not used in STACK.
)

REGISTRY: Dict[str, Dielectric] = {
    d.name: d
    for d in (
        DUROID_5880, AIR, SKIN_DRY, SKIN_WET, FAT_NOT_INFILTRATED, FAT_INFILTRATED,
        STRATUM_CORNEUM, EPIDERMIS, DERMIS, HYPODERMIS, MALIGNANT, BENIGN_LESION,
    )
}


def check_anchors(verbose: bool = True) -> List[str]:
    """Compare the model against published anchor values.

    Returns the list of *unexpected* failures. Anchors carrying a known_mismatch note
    are printed as WARN and are not returned, so a documented discrepancy does not block
    the pipeline but also never disappears.
    """
    failures: List[str] = []
    warnings: List[str] = []
    if verbose:
        print(f"{'tissue':<22}{'f GHz':>7}{'eps model':>11}{'eps pub':>9}"
              f"{'sig model':>11}{'sig pub':>9}{'d eps':>8}{'d sig':>8}   status")
    for a in ANCHORS:
        d = REGISTRY[a.tissue]
        eps_m = float(np.real(d.eps_c(np.array([a.f_hz]))[0]))
        sig_m = float(d.sigma_eff(np.array([a.f_hz]))[0])
        e_err = abs(eps_m - a.eps_prime) / a.eps_prime
        s_err = abs(sig_m - a.sigma) / a.sigma
        ok = e_err <= a.tol_eps and s_err <= a.tol_sigma
        if ok:
            status = "OK"
        elif a.known_mismatch:
            status = "WARN"
            warnings.append(f"{a.tissue} @ {a.f_hz/1e9:.2f} GHz: {a.known_mismatch}")
        else:
            status = "FAIL"
            failures.append(
                f"{a.tissue} @ {a.f_hz/1e9:.2f} GHz: eps err {e_err:.2%} "
                f"(tol {a.tol_eps:.0%}), sigma err {s_err:.2%} (tol {a.tol_sigma:.0%})"
            )
        if verbose:
            print(
                f"{a.tissue:<22}{a.f_hz/1e9:>7.2f}{eps_m:>11.3f}{a.eps_prime:>9.3f}"
                f"{sig_m:>11.4f}{a.sigma:>9.4f}{e_err:>8.2%}{s_err:>8.2%}   {status}"
            )
    if verbose and warnings:
        print("\nknown mismatches (documented, not blocking):")
        for w in warnings:
            print(f"  - {w}")
    return failures


def todo_report() -> List[str]:
    """List everything still marked provisional."""
    rows = []
    for d in REGISTRY.values():
        if d.provisional or "TODO_SOURCE" in d.source:
            rows.append(f"  [{d.name}] {d.source}")
    for ls in STACK:
        if "TODO_SOURCE" in ls.source:
            rows.append(f"  [geometry:{ls.name}] {ls.source}")
    return rows


def contrast_report(freqs_ghz=(2.0, 5.0, 10.0, 18.0, 26.0)) -> None:
    """Normal-vs-lesion contrast at a few frequencies. This is the number that matters."""
    f = np.array(freqs_ghz) * 1e9
    dn = DERMIS.eps_c(f)
    dm = MALIGNANT.eps_c(f)
    db = BENIGN_LESION.eps_c(f)
    hdr = ("f GHz".rjust(7) + "dermis eps".rjust(12) + "malig eps".rjust(11) + "d%".rjust(7)
           + 'dermis eps"'.rjust(13) + 'malig eps"'.rjust(12) + "d%".rjust(7)
           + "benign d% eps".rjust(15))
    print("\n" + hdr)
    for i, fi in enumerate(freqs_ghz):
        e1, e2, e3 = dn[i], dm[i], db[i]
        print(
            f"{fi:>7.1f}{e1.real:>12.2f}{e2.real:>11.2f}{100*(e2.real/e1.real-1):>7.1f}"
            f"{-e1.imag:>13.2f}{-e2.imag:>12.2f}{100*(e2.imag/e1.imag-1):>7.1f}"
            f"{100*(e3.real/e1.real-1):>15.1f}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect and verify the dielectric tables.")
    ap.add_argument("--todo", action="store_true", help="only print the TODO_SOURCE report")
    ap.add_argument("--plot", action="store_true", help="write results/fig_dielectrics.png")
    args = ap.parse_args()

    todos = todo_report()
    if args.todo:
        print("TODO_SOURCE / provisional entries:")
        print("\n".join(todos) if todos else "  (none)")
        return 0

    print("=== anchor check against published values ===")
    failures = check_anchors()
    print("\n=== normal vs lesion contrast ===")
    contrast_report()
    print(f"\n=== {len(todos)} provisional entries (run --todo for detail) ===")
    for t in todos:
        print(t)

    if args.plot:
        import os
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs("results", exist_ok=True)
        f = np.linspace(1e9, 30e9, 400)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for d in (STRATUM_CORNEUM, EPIDERMIS, DERMIS, HYPODERMIS, MALIGNANT, BENIGN_LESION):
            e = d.eps_c(f)
            ax[0].plot(f / 1e9, e.real, label=d.name)
            ax[1].plot(f / 1e9, -e.imag, label=d.name)
        for a in ANCHORS:
            ax[0].plot(a.f_hz / 1e9, a.eps_prime, "k*", ms=9)
        ax[0].set_xlabel("GHz"); ax[0].set_ylabel("eps'"); ax[0].set_title("real part (* = published anchor)")
        ax[1].set_xlabel("GHz"); ax[1].set_ylabel('eps"'); ax[1].set_title("loss")
        ax[1].legend(fontsize=7)
        for a in ax:
            a.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig("results/fig_dielectrics.png", dpi=150)
        print("\nwrote results/fig_dielectrics.png")

    if failures:
        print("\nANCHOR CHECK FAILED:")
        for f_ in failures:
            print("  " + f_)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
