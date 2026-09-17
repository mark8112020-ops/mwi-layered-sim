"""
validate.py -- section 7 of the handoff spec, in order.

    1. Lossless half-space against the closed-form Fresnel coefficient.
    2. Lossy half-space: confirm the field decays rather than grows.
    3. Quarter-wave matching layer produces a null at the design frequency.
    4. Thin-layer limit: as a thickness goes to zero, S11 matches the stack without it.
    5. Passivity: |S11| never exceeds 1 for a passive stack.

If any of these fail, STOP. Do not run the ML stage on a broken forward model.
generate.py calls run_all() before it writes a dataset, and refuses to write if a
check fails (override with --skip-validation, which you should not use).

    python validate.py            # run everything, exit 1 on failure
    python validate.py -v         # print the numbers, not just pass/fail
"""

from __future__ import annotations

import argparse
from typing import Callable, List, Tuple

import numpy as np

import tissue_params as tp
from layered import (
    Layer,
    fresnel_halfspace,
    intrinsic_impedance,
    propagation_constant,
    s11,
)

F_TEST = np.linspace(2e9, 26e9, 241)


def _const(eps: complex, n: int) -> np.ndarray:
    return np.full(n, eps, dtype=complex)


# --------------------------------------------------------------------------------------
# 1. Lossless half-space vs Fresnel
# --------------------------------------------------------------------------------------
def check_lossless_halfspace(verbose: bool = False) -> Tuple[bool, str]:
    f = F_TEST
    eps_ref = _const(2.2, f.size)
    worst = 0.0
    for eps_r in (1.0, 4.0, 12.0, 40.0):
        eps_t = _const(eps_r, f.size)
        got = s11([Layer("halfspace", eps_t, 0.0)], f, eps_ref)
        want = fresnel_halfspace(eps_ref, eps_t)
        worst = max(worst, float(np.max(np.abs(got - want))))
    ok = worst < 1e-12
    msg = f"max |S11_recursion - S11_fresnel| = {worst:.3e} (tol 1e-12)"
    if verbose:
        # Sanity in physical terms: 2.2 -> 40 should reflect hard and with a sign flip.
        r = fresnel_halfspace(_const(2.2, 1), _const(40.0, 1))[0]
        msg += f"\n    duroid->eps40: S11 = {r.real:+.4f}{r.imag:+.4f}j (expect ~-0.60, real)"
    return ok, msg


# --------------------------------------------------------------------------------------
# 2. Lossy half-space decays
# --------------------------------------------------------------------------------------
def check_lossy_halfspace_decays(verbose: bool = False) -> Tuple[bool, str]:
    f = F_TEST
    lines: List[str] = []
    ok = True
    for mat in (tp.DERMIS, tp.MALIGNANT, tp.HYPODERMIS, tp.SKIN_DRY):
        eps = mat.eps_c(f)
        if np.any(np.imag(eps) > 0):
            ok = False
            lines.append(f"{mat.name}: eps'' has the wrong sign (passive media must have Im(eps_c) <= 0)")
            continue
        gamma = propagation_constant(eps, f)
        eta = intrinsic_impedance(eps)
        if np.any(np.real(gamma) <= 0):
            ok = False
            lines.append(f"{mat.name}: Re(gamma) <= 0, the wave grows into the tissue")
        if np.any(np.real(eta) <= 0):
            ok = False
            lines.append(f"{mat.name}: Re(eta) <= 0, the medium delivers power instead of absorbing it")
        # Skin depth must also be physically sane: mm-scale at 2 GHz, sub-mm at 26 GHz.
        delta = 1.0 / np.real(gamma)
        if verbose:
            lines.append(
                f"{mat.name:<18} skin depth {delta[0]*1e3:6.2f} mm @2 GHz -> "
                f"{delta[-1]*1e3:5.2f} mm @26 GHz"
            )
        if np.any(np.diff(delta) > 0) and mat is not tp.HYPODERMIS:
            lines.append(f"note: {mat.name} skin depth is not monotonically decreasing with f")
    return ok, "\n    ".join(lines) if lines else "all lossy media attenuate; Re(gamma) > 0, Re(eta) > 0"


# --------------------------------------------------------------------------------------
# 3. Quarter-wave matching layer
# --------------------------------------------------------------------------------------
def check_quarter_wave_null(verbose: bool = False) -> Tuple[bool, str]:
    eps1, eps2 = 2.2, 12.0  # duroid reference into a lossless dielectric half-space
    eps_m = np.sqrt(eps1 * eps2)  # matching layer: eta_m = sqrt(eta1*eta2)
    f0 = 10e9
    lam_m = tp.C0 / (f0 * np.sqrt(eps_m))
    d = lam_m / 4.0

    # Grid deliberately runs past the instrument bands: a quarter-wave transformer also
    # matches at every odd harmonic, and 3*f0 = 30 GHz is the second half of this check.
    f = np.linspace(2e9, 34e9, 6401)
    stack = [
        Layer("match", _const(eps_m, f.size), d),
        Layer("halfspace", _const(eps2, f.size), 0.0),
    ]
    r = np.abs(s11(stack, f, _const(eps1, f.size)))

    i0 = int(np.argmin(np.abs(f - f0)))
    depth_at_f0 = r[i0]
    i_min = int(np.argmin(r))
    f_min = f[i_min]
    # A quarter-wave transformer also repeats every half wavelength -> nulls at 3*f0.
    i_3f0 = int(np.argmin(np.abs(f - 3 * f0)))

    ok = depth_at_f0 < 1e-9 and abs(f_min - f0) / f0 < 1e-3 and r[i_3f0] < 1e-9
    msg = (
        f"|S11| at f0=10 GHz = {depth_at_f0:.3e} (tol 1e-9); global min at "
        f"{f_min/1e9:.4f} GHz; |S11| at 3*f0 = {r[i_3f0]:.3e}"
    )
    if verbose:
        msg += (
            f"\n    matching layer eps={eps_m:.4f}, d={d*1e3:.4f} mm, "
            f"|S11| without it = {abs((np.sqrt(eps1)-np.sqrt(eps2))/(np.sqrt(eps1)+np.sqrt(eps2))):.4f}"
        )
    return ok, msg


# --------------------------------------------------------------------------------------
# 4. Thin-layer limit
# --------------------------------------------------------------------------------------
def check_thin_layer_limit(verbose: bool = False) -> Tuple[bool, str]:
    f = F_TEST
    eps_ref = tp.REFERENCE_MEDIUM.eps_c(f)
    base = [
        Layer("epidermis", tp.EPIDERMIS.eps_c(f), 100e-6),
        Layer("dermis", tp.DERMIS.eps_c(f), 2e-3),
        Layer("hypodermis", tp.HYPODERMIS.eps_c(f), 0.0),
    ]
    ref = s11(base, f, eps_ref)

    rows = []
    ok = True
    prev = None
    for d in (1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 0.0):
        stack = [Layer("thin", tp.MALIGNANT.eps_c(f), d)] + base
        err = float(np.max(np.abs(s11(stack, f, eps_ref) - ref)))
        rows.append(f"d = {d*1e6:9.3f} um -> max |dS11| = {err:.3e}")
        if prev is not None and err > prev * 1.5 + 1e-15:
            ok = False  # must be monotone in the limit
        prev = err
    if prev is None or prev > 1e-14:
        ok = False
    msg = "\n    ".join(rows) if verbose else rows[-1] + f"  (converging: {rows[0]} -> {rows[-1]})"
    return ok, msg


# --------------------------------------------------------------------------------------
# 5. Passivity
# --------------------------------------------------------------------------------------
def check_passivity(verbose: bool = False, n: int = 400, seed: int = 0) -> Tuple[bool, str]:
    import generate  # imported here so validate.py stays importable on its own

    rng = np.random.default_rng(seed)
    f = np.concatenate([tp.band_frequencies("low"), tp.band_frequencies("mid")])
    worst = 0.0
    for _ in range(n):
        sample = generate.sample_stack(rng, f, label=int(rng.integers(0, 2)))
        worst = max(worst, float(np.max(np.abs(sample.s11))))
    ok = worst <= 1.0 + 1e-12
    return ok, f"max |S11| over {n} random stacks = {worst:.9f} (must be <= 1)"


CHECKS: List[Tuple[str, Callable[[bool], Tuple[bool, str]]]] = [
    ("1. lossless half-space vs Fresnel", check_lossless_halfspace),
    ("2. lossy half-space decays", check_lossy_halfspace_decays),
    ("3. quarter-wave null at design f", check_quarter_wave_null),
    ("4. thin-layer limit", check_thin_layer_limit),
    ("5. passivity |S11| <= 1", check_passivity),
]


def run_all(verbose: bool = False) -> bool:
    print("=" * 78)
    print("forward-model validation (spec section 7)")
    print("=" * 78)
    all_ok = True
    for name, fn in CHECKS:
        try:
            ok, msg = fn(verbose)
        except Exception as exc:  # a crash is a failure, not a traceback to ignore
            ok, msg = False, f"raised {type(exc).__name__}: {exc}"
        all_ok &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        print(f"    {msg}")
    print("-" * 78)
    print("ALL CHECKS PASSED" if all_ok else "VALIDATION FAILED -- do not proceed to the ML stage")
    return all_ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate the layered forward model.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    return 0 if run_all(args.verbose) else 1


if __name__ == "__main__":
    raise SystemExit(main())
