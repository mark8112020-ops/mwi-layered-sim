"""
sweep_phase_error.py -- the headline experiment.

Sweep the receiver phase-error magnitude and plot classification performance of every
feature set against it. The crossing point, if there is one, is the engineering spec
for how good the phase reference has to be: to the left of it a phase-coherent receiver
buys accuracy, to the right of it a scalar detector front end is just as good and the
six-port / ADL5961 / ADF4371 complexity is not paying for itself.

Everything else is held fixed across the sweep -- same stacks, same magnitude errors,
same additive noise realisation seed per level -- so the only thing moving is phase.

    python sweep_phase_error.py --data data/dataset.npz --n 6000

Outputs results/phase_sweep.json and results/fig_phase_sweep.png.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

import features as feat
import generate
import train

# Degrees rms, total phase error. 0 is the noiseless-phase reference; 90 is a receiver
# with effectively no usable phase information.
DEFAULT_GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 45.0, 90.0]

# Open question 1 in the spec: what a six-port at 26 GHz actually achieves after
# calibration. Until that is measured, this band is a guess and is drawn as a guess.
ASSUMED_SIXPORT_RANGE_DEG = (2.0, 10.0)


def run_sweep(
    data: Dict[str, object],
    grid: Sequence[float] = DEFAULT_GRID,
    sets: Sequence[str] = ("A", "B", "C", "D"),
    model: str = "rf",
    k: int = 5,
    seed: int = train.DEFAULT_SEED,
    snr_db: Optional[float] = None,
    repeats: int = 1,
) -> Dict[str, object]:
    y = np.asarray(data["y"]).astype(int)
    f = np.asarray(data["f"])
    bs = generate.band_slices_from(data)
    clean = np.asarray(data["s11"])

    rows: List[Dict[str, object]] = []
    print(f"{'phase deg':>10}{'rep':>5}{'set':>5}{'acc':>9}{'sens':>9}{'spec':>9}"
          f"{'AUC':>9}{'spec@95':>9}{'sec':>7}")
    for ph in grid:
        for rep in range(repeats):
            # Same seed offset per (level, rep): the magnitude errors and the stacks are
            # identical across the sweep, only the phase-error scale changes.
            noisy = train.prepare_noisy(
                data, seed=seed + 1000 * rep, phase_err_deg=ph, snr_db=snr_db
            )
            for fs in sets:
                X, _cols, _fc = feat.extract(noisy, f, fs, band_slices=bs)
                t0 = time.time()
                r = train.cross_validate(X, y, kind=model, k=k, seed=seed)
                r.update(phase_err_deg=float(ph), feature_set=fs, repeat=rep,
                         uses_phase=feat.uses_phase(fs))
                rows.append(r)
                print(f"{ph:>10.2f}{rep:>5}{fs:>5}{r['accuracy']:>9.4f}{r['sensitivity']:>9.4f}"
                      f"{r['specificity']:>9.4f}{r['roc_auc']:>9.4f}"
                      f"{r['specificity_at_95_sensitivity']:>9.4f}{time.time()-t0:>7.1f}", flush=True)
    return dict(rows=rows, grid=list(grid), sets=list(sets), model=model, k=k, seed=seed,
                n_samples=int(y.size), repeats=repeats)


# --------------------------------------------------------------------------------------
# Crossing point
# --------------------------------------------------------------------------------------
def _curve(rows, fs, metric) -> Dict[float, float]:
    out: Dict[float, List[float]] = {}
    for r in rows:
        if r["feature_set"] == fs:
            out.setdefault(r["phase_err_deg"], []).append(float(r[metric]))
    return {k: float(np.mean(v)) for k, v in sorted(out.items())}


def find_crossing(
    rows, coherent: str = "A", scalar: str = "B", metric: str = "roc_auc"
) -> Dict[str, object]:
    """Where does the coherent feature set stop beating the scalar one?

    Linear interpolation in log10(phase error) on the sign change of (coherent - scalar).
    Returns crossing=None if the coherent set never leads, or never loses its lead
    inside the swept range -- both are meaningful engineering answers and are reported
    as such rather than being forced into a number.
    """
    ca, cb = _curve(rows, coherent, metric), _curve(rows, scalar, metric)
    xs = sorted(set(ca) & set(cb))
    d = [ca[x] - cb[x] for x in xs]

    res: Dict[str, object] = dict(
        metric=metric, coherent=coherent, scalar=scalar,
        phase_deg=xs, delta=d,
        advantage_at_zero=float(d[0]) if d else float("nan"),
    )
    if not d or d[0] <= 0:
        res.update(crossing_deg=None,
                   verdict=f"set {coherent} never leads set {scalar}, even at zero phase error: "
                           "magnitude-only is sufficient")
        return res
    for i in range(1, len(xs)):
        if d[i] <= 0 < d[i - 1]:
            x0 = xs[i - 1]
            if x0 <= 0:
                # The lead is already gone at the smallest nonzero point swept. Say that
                # instead of interpolating from log10(0) and inventing a number.
                res.update(
                    crossing_deg=None,
                    crossing_upper_bound_deg=float(xs[i]),
                    verdict=(f"set {coherent}'s lead is gone by {xs[i]:.2f} deg rms, the smallest "
                             "nonzero phase error swept -- refine the grid below this to place the "
                             "crossing, but the practical answer is that any realistic phase "
                             "reference is already too coarse to help"),
                )
                return res
            # interpolate in log10 phase
            lx = np.log10(x0) + (np.log10(xs[i]) - np.log10(x0)) * d[i - 1] / (d[i - 1] - d[i])
            cross = float(10 ** lx)
            res.update(
                crossing_deg=cross,
                verdict=(f"a phase-coherent receiver pays for itself only below "
                         f"~{cross:.2f} deg rms phase error; above that, magnitude-only "
                         f"matches or beats it"),
            )
            return res
    res.update(crossing_deg=None,
               verdict=f"set {coherent} still leads at {xs[-1]:.0f} deg rms; no crossing inside "
                       "the swept range -- a coherent receiver is worth it at any realistic "
                       "phase accuracy")
    return res


# --------------------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------------------
def plot_sweep(res: Dict[str, object], path: str, crossings: Dict[str, object]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = res["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    colors = {"A": "tab:red", "B": "tab:blue", "C": "tab:orange", "D": "tab:green"}
    styles = {"A": "-", "B": "-", "C": "--", "D": "-."}

    for ax, metric, ylab in (
        (axes[0], "roc_auc", "ROC AUC"),
        (axes[1], "sensitivity", "sensitivity at 0.5 threshold"),
    ):
        for fs in res["sets"]:
            c = _curve(rows, fs, metric)
            x = np.array(list(c.keys()))
            v = np.array(list(c.values()))
            ax.plot(x, v, styles.get(fs, "-"), color=colors.get(fs), marker="o", ms=3.5,
                    label=feat.FEATURE_SET_SHORT.get(fs, fs))
        ax.set_xscale("symlog", linthresh=0.25)
        ax.set_xlabel("receiver phase error, deg rms")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3, which="both")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.12 * (hi - lo))  # headroom for the annotations
        ax.axvspan(*ASSUMED_SIXPORT_RANGE_DEG, color="grey", alpha=0.12)
        ax.text(np.sqrt(ASSUMED_SIXPORT_RANGE_DEG[0] * ASSUMED_SIXPORT_RANGE_DEG[1]),
                ax.get_ylim()[1], "assumed six-port\npost-calibration\n(TODO: measure)",
                fontsize=7, va="top", ha="center", color="dimgrey")

    cr = crossings.get("roc_auc", {})
    if cr.get("crossing_deg"):
        for ax in axes:
            ax.axvline(cr["crossing_deg"], color="k", ls=":", lw=1.2)
        axes[0].annotate(f"crossing\n{cr['crossing_deg']:.2f} deg",
                         xy=(cr["crossing_deg"], axes[0].get_ylim()[0]),
                         xytext=(-4, 14), textcoords="offset points", fontsize=8, ha="right")

    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(
        "Does magnitude-only S11 lose to complex S11 once the phase reference is realistic?\n"
        + str(cr.get("verdict", "")),
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep receiver phase error (headline experiment).")
    ap.add_argument("--data", default="data/dataset.npz")
    ap.add_argument("--out", default="results/phase_sweep.json")
    ap.add_argument("--fig", default="results/fig_phase_sweep.png")
    ap.add_argument("--grid", nargs="+", type=float, default=DEFAULT_GRID)
    ap.add_argument("--sets", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--model", default="rf", choices=["rf", "xgb"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=train.DEFAULT_SEED)
    ap.add_argument("--snr-db", type=float, default=None)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--n", type=int, default=None, help="subsample for speed")
    ap.add_argument("--replot", action="store_true",
                    help="redraw the figure from an existing results json, no refitting")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)

    if args.replot:
        with open(args.out) as fh:
            res = json.load(fh)
        plot_sweep(res, args.fig, res.get("crossings", {}))
        print(f"replotted {args.fig} from {args.out}")
        return 0

    data = generate.load_dataset(args.data)
    if args.n and args.n < np.asarray(data["y"]).size:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(np.asarray(data["y"]).size, size=args.n, replace=False)
        data = dict(data)
        for key in ("s11", "y", "meta"):
            data[key] = np.asarray(data[key])[idx]

    print(f"phase-error sweep: n={np.asarray(data['y']).size}, model={args.model}, "
          f"k={args.k}, grid={args.grid}")
    res = run_sweep(data, grid=args.grid, sets=args.sets, model=args.model, k=args.k,
                    seed=args.seed, snr_db=args.snr_db, repeats=args.repeats)
    res["dataset_config"] = json.loads(str(data["config"]))
    res["noise_config"] = dict(generate.DEFAULT_NOISE)
    if args.snr_db is not None:
        res["noise_config"]["snr_db"] = args.snr_db

    crossings = {}
    for metric in ("roc_auc", "accuracy"):
        crossings[metric] = find_crossing(res["rows"], "A", "B", metric)
    if "C" in args.sets and "B" in args.sets:
        crossings["roc_auc_C_vs_B"] = find_crossing(res["rows"], "C", "B", "roc_auc")
    res["crossings"] = crossings

    print("\n" + "=" * 78)
    print("HEADLINE RESULT")
    print("=" * 78)
    for key, cr in crossings.items():
        print(f"[{key}] {cr['verdict']}")
        print(f"    advantage of {cr['coherent']} over {cr['scalar']} at zero phase error: "
              f"{cr['advantage_at_zero']:+.4f}")
    print("=" * 78)

    plot_sweep(res, args.fig, crossings)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"wrote {args.fig}\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
