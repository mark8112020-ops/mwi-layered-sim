"""
layered.py -- 1D layered-media forward model.

Plane wave at normal incidence onto a stack of homogeneous planar layers, S11 from a
transmission-line impedance recursion run from the semi-infinite bottom layer upward.

    eta_i   = eta_0 / sqrt(eps_c_i)
    gamma_i = 1j * (w/c) * sqrt(eps_c_i)
    Z_N     = eta_N                                        (semi-infinite bottom)
    Z_i     = eta_i * (Z_{i+1} + eta_i*tanh(g_i d_i)) / (eta_i + Z_{i+1}*tanh(g_i d_i))
    S11     = (Z_1 - eta_ref) / (Z_1 + eta_ref)

Branch convention (the spec calls this out, and it is the one thing here that silently
ruins everything downstream if it is wrong): with time dependence exp(+jwt) and
eps_c = eps' - j*eps'', numpy's principal sqrt returns a root with positive real part
and negative imaginary part, so gamma = j*(w/c)*sqrt(eps_c) has POSITIVE real part and
exp(-gamma*z) decays into the tissue. `assert_decaying()` checks that explicitly; the
lossy half-space case in validate.py checks it end to end.

Everything is vectorised over frequency. Layer count is small, so the recursion loop
over layers is fine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from tissue_params import C0, ETA_0, Dielectric

# tanh saturates for |Re(arg)| beyond this; clipping prevents overflow warnings on very
# lossy or very thick layers without changing the answer (tanh(30) == 1 to 1e-26).
_TANH_CLIP = 30.0


@dataclass
class Layer:
    """One physical layer: a complex permittivity spectrum and a thickness."""

    name: str
    eps_c: np.ndarray  # complex, shape (n_freq,)
    thickness: float  # m; ignored for the bottom (semi-infinite) layer


def eps_c(dielectric: Dielectric, f_hz: np.ndarray) -> np.ndarray:
    """Complex relative permittivity of a material over a frequency grid."""
    return dielectric.eps_c(f_hz)


def intrinsic_impedance(eps_c_arr: np.ndarray) -> np.ndarray:
    """eta = eta_0 / sqrt(eps_c), principal branch."""
    return ETA_0 / np.sqrt(np.asarray(eps_c_arr, dtype=complex))


def propagation_constant(eps_c_arr: np.ndarray, f_hz: np.ndarray) -> np.ndarray:
    """gamma = alpha + j*beta = 1j*(w/c)*sqrt(eps_c), principal branch."""
    w = 2.0 * np.pi * np.asarray(f_hz, dtype=float)
    return 1j * (w / C0) * np.sqrt(np.asarray(eps_c_arr, dtype=complex))


def assert_decaying(gamma: np.ndarray, name: str = "layer") -> None:
    """Guard the branch cut: a passive lossy medium must attenuate, never grow."""
    if np.any(np.real(gamma) < -1e-9):
        raise ValueError(
            f"{name}: Re(gamma) < 0, the field grows into the medium. "
            "The sqrt branch or the sign of eps'' is wrong."
        )


def _tanh(x: np.ndarray) -> np.ndarray:
    re = np.clip(np.real(x), -_TANH_CLIP, _TANH_CLIP)
    return np.tanh(re + 1j * np.imag(x))


def input_impedance(layers: Sequence[Layer], f_hz: np.ndarray, check: bool = True) -> np.ndarray:
    """Impedance looking into the top of the stack.

    layers[-1] is treated as semi-infinite; its thickness is ignored.
    """
    if len(layers) == 0:
        raise ValueError("empty stack")
    f = np.asarray(f_hz, dtype=float)

    bottom = layers[-1]
    eta_b = intrinsic_impedance(bottom.eps_c)
    if check:
        assert_decaying(propagation_constant(bottom.eps_c, f), bottom.name)
    Z = eta_b

    for layer in reversed(layers[:-1]):
        eta = intrinsic_impedance(layer.eps_c)
        gamma = propagation_constant(layer.eps_c, f)
        if check:
            assert_decaying(gamma, layer.name)
        if layer.thickness <= 0.0:
            continue  # degenerate layer: pass the impedance through untouched
        t = _tanh(gamma * layer.thickness)
        Z = eta * (Z + eta * t) / (eta + Z * t)
    return Z


def s11_from_impedance(Z1: np.ndarray, eta_ref: np.ndarray) -> np.ndarray:
    return (Z1 - eta_ref) / (Z1 + eta_ref)


def s11(
    layers: Sequence[Layer],
    f_hz: np.ndarray,
    eps_c_ref: np.ndarray,
    check: bool = True,
) -> np.ndarray:
    """Complex S11 at the probe face for the given stack.

    eps_c_ref is the complex permittivity of the reference medium (the coupling
    medium / antenna substrate), which sets eta_ref.
    """
    eta_ref = intrinsic_impedance(eps_c_ref)
    Z1 = input_impedance(layers, f_hz, check=check)
    return s11_from_impedance(Z1, eta_ref)


def fresnel_halfspace(eps_c_1: np.ndarray, eps_c_2: np.ndarray) -> np.ndarray:
    """Closed-form normal-incidence reflection between two half-spaces (validation ref)."""
    eta1 = intrinsic_impedance(eps_c_1)
    eta2 = intrinsic_impedance(eps_c_2)
    return (eta2 - eta1) / (eta2 + eta1)


# --------------------------------------------------------------------------------------
# Stack construction with inclusion carving
# --------------------------------------------------------------------------------------
def build_layers(
    materials: Sequence[Tuple[str, np.ndarray]],
    thicknesses: Sequence[float],
) -> List[Layer]:
    return [Layer(n, e, t) for (n, e), t in zip(materials, thicknesses)]


def carve_inclusion(
    layers: Sequence[Layer],
    top_depth: float,
    thickness: float,
    inclusion_name: str,
    inclusion_eps: np.ndarray,
    depth_origin_index: int = 0,
) -> List[Layer]:
    """Replace the material in a depth window with the inclusion material.

    This is how the spec's "tumour displaces part of the epidermis/dermis" is
    implemented, and it is also the anti-leakage mechanism from section 5: the window
    is cut *out of* the existing stack, so every interface below the lesion keeps its
    position and the total stack thickness is bit-for-bit identical to what it would
    have been without the lesion. A tree ensemble therefore cannot reach the label
    through the geometry -- only through the dielectric contrast.

    top_depth is measured from the top of layer `depth_origin_index` (the skin surface),
    so it is independent of the coupling-gap thickness.

    The bottom layer is semi-infinite and is never carved into.
    """
    if thickness <= 0.0:
        return list(layers)

    out: List[Layer] = list(layers[:depth_origin_index])
    z = 0.0  # depth of the top of the current layer, measured from the origin layer
    z0, z1 = top_depth, top_depth + thickness
    inserted = False

    body = list(layers[depth_origin_index:])
    for i, layer in enumerate(body):
        last = i == len(body) - 1
        if last:
            # Semi-infinite bottom layer: never carved, always terminates the stack.
            out.append(layer)
            break
        a, b = z, z + layer.thickness  # this layer spans [a, b)
        z = b
        if b <= z0 or a >= z1:
            out.append(layer)  # entirely outside the window
            continue
        # Part above the window stays as the original material.
        if a < z0:
            out.append(Layer(layer.name, layer.eps_c, z0 - a))
        # The overlapping part becomes inclusion material (merged into one layer).
        lo, hi = max(a, z0), min(b, z1)
        if hi > lo:
            if inserted and out and out[-1].name == inclusion_name:
                out[-1] = Layer(inclusion_name, inclusion_eps, out[-1].thickness + (hi - lo))
            else:
                out.append(Layer(inclusion_name, inclusion_eps, hi - lo))
                inserted = True
        # Part below the window stays as the original material.
        if b > z1:
            out.append(Layer(layer.name, layer.eps_c, b - z1))
    return out


def total_thickness(layers: Sequence[Layer]) -> float:
    """Sum of finite thicknesses (excludes the semi-infinite bottom layer)."""
    return float(sum(l.thickness for l in layers[:-1]))


def describe(layers: Sequence[Layer]) -> str:
    rows = [f"  {l.name:<20}{l.thickness*1e6:>10.1f} um" for l in layers[:-1]]
    rows.append(f"  {layers[-1].name:<20}{'semi-inf':>13}")
    return "\n".join(rows)
