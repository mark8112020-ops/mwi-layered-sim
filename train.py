"""
train.py -- ML protocol (spec section 5).

  - Random forest first (speed, interpretability), XGBoost second (headline number).
  - Stratified k-fold, k=5, fixed seed, recorded in the results file.
  - Sensitivity and specificity reported separately, not just accuracy. A missed
    melanoma and a false alarm are not the same error, so specificity at a fixed 95%
    sensitivity operating point is reported too -- that is the number a triage device
    is actually judged on.
  - ROC AUC and the full confusion matrix.
  - Permutation importance over frequency, plotted as importance vs GHz. If the low
    sub-bands contribute nothing, the hardware band can shrink and the BOM gets cheaper.
  - Leakage audit: a model trained on the nuisance parameters alone must score ~0.5.
    Any nuisance parameter correlated with the label would be found by the trees and
    would inflate every number above it.

    python train.py --data data/dataset.npz --out results/metrics_main.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

import features as feat
import generate
import tissue_params as tp

DEFAULT_SEED = 20260801


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
def make_model(kind: str, seed: int, n_jobs: int = -1):
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=n_jobs,
            random_state=seed,
        )
    if kind == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.5,
            reg_lambda=1.0,
            tree_method="hist",
            n_jobs=n_jobs,
            random_state=seed,
            eval_metric="logloss",
        )
    raise ValueError(f"unknown model {kind!r}")


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def metrics_from_oof(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> Dict[str, object]:
    """Sensitivity and specificity separately, plus AUC and the confusion matrix."""
    yhat = (p >= threshold).astype(int)
    tn, fp, fn, tp_ = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    sens = tp_ / (tp_ + fn) if (tp_ + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    auc = float(roc_auc_score(y, p))

    # Operating point a triage device would actually be set to: fix sensitivity at 95%
    # and report what specificity survives.
    fpr, tpr, thr = roc_curve(y, p)
    i = int(np.argmax(tpr >= 0.95))
    spec_at_95 = float(1.0 - fpr[i]) if np.any(tpr >= 0.95) else float("nan")

    return dict(
        accuracy=float((yhat == y).mean()),
        sensitivity=float(sens),
        specificity=float(spec),
        roc_auc=auc,
        specificity_at_95_sensitivity=spec_at_95,
        threshold_at_95_sensitivity=float(thr[i]) if np.any(tpr >= 0.95) else float("nan"),
        confusion_matrix=dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp_)),
        n=int(y.size),
    )


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    kind: str = "rf",
    k: int = 5,
    seed: int = DEFAULT_SEED,
    return_oof: bool = False,
) -> Dict[str, object]:
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    oof = np.zeros(y.size, dtype=float)
    fold_acc: List[float] = []
    t0 = time.time()
    for tr, te in skf.split(X, y):
        m = make_model(kind, seed)
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        oof[te] = p
        fold_acc.append(float(((p >= 0.5).astype(int) == y[te]).mean()))
    out = metrics_from_oof(y, oof)
    out.update(
        model=kind, k=k, seed=seed,
        fold_accuracy_mean=float(np.mean(fold_acc)),
        fold_accuracy_std=float(np.std(fold_acc)),
        fit_seconds=round(time.time() - t0, 1),
    )
    if return_oof:
        out["_oof"] = oof
    return out


# --------------------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------------------
def prepare_noisy(
    data: Dict[str, object],
    seed: int = DEFAULT_SEED,
    **noise_kwargs,
) -> np.ndarray:
    """Apply the measurement-realism model to the stored clean S11."""
    rng = np.random.default_rng(seed)
    kw = dict(generate.DEFAULT_NOISE)
    kw.update({k: v for k, v in noise_kwargs.items() if v is not None})
    return generate.apply_measurement_noise(
        np.asarray(data["s11"]),
        np.asarray(data["f"]),
        rng,
        band_slices=generate.band_slices_from(data),
        **kw,
    )


def build_features(
    s11_noisy: np.ndarray, data: Dict[str, object], which: Sequence[str]
) -> Dict[str, Tuple[np.ndarray, List[str], np.ndarray]]:
    f = np.asarray(data["f"])
    bs = generate.band_slices_from(data)
    return {fs: feat.extract(s11_noisy, f, fs, band_slices=bs) for fs in which}


# --------------------------------------------------------------------------------------
# Leakage audit (spec section 5)
# --------------------------------------------------------------------------------------
def leakage_audit(data: Dict[str, object], seed: int = DEFAULT_SEED) -> Dict[str, object]:
    """Train on nuisance parameters only. Anything above chance in the first two
    audits is a leak, and every score above it in the results file is inflated.

    Three separate questions, because they have three different right answers:

    stack_geometry
        Layer thicknesses, hydration factor, coupling gap, total stack thickness.
        These are the variables the trees could use as a label proxy if the lesion
        had been *appended* to the stack instead of carved out of it. Must be ~0.5.

    lesion_geometry_given_inclusion
        Lesion depth and thickness, among samples that have an inclusion at all.
        Benign and malignant inclusions are drawn from the same geometry
        distribution, so this must also be ~0.5.

    has_inclusion (design invariant, NOT a leak)
        Every malignant sample has an inclusion; only `benign_lesion_prob` of the
        benign ones do, so this variable is genuinely predictive with an expected AUC
        of 1 - benign_lesion_prob/2. It is not measurable by the instrument, so it
        cannot inflate anything -- it is reported to confirm the generator did what
        it was told. Setting benign_lesion_prob=1.0 drives it to 0.5.
    """
    meta = np.asarray(data["meta"], dtype=float)
    keys = [str(k) for k in np.asarray(data["meta_keys"])]
    y = np.asarray(data["y"]).astype(int)
    col = {k: i for i, k in enumerate(keys)}
    lesion_cols = {"lesion_top_depth", "lesion_thickness", "has_inclusion"}

    out: Dict[str, object] = {"meta_keys": keys}

    idx_stack = [i for k, i in col.items() if k not in lesion_cols]
    r = cross_validate(meta[:, idx_stack], y, kind="rf", k=5, seed=seed)
    r.update(columns=[keys[i] for i in idx_stack], expected_auc=0.5, is_leak=bool(r["roc_auc"] > 0.56))
    out["stack_geometry"] = r

    has_inc = meta[:, col["has_inclusion"]] > 0.5
    if has_inc.sum() > 100 and 0 < y[has_inc].mean() < 1:
        idx_les = [col["lesion_top_depth"], col["lesion_thickness"]]
        r2 = cross_validate(meta[has_inc][:, idx_les], y[has_inc], kind="rf", k=5, seed=seed)
        r2.update(columns=["lesion_top_depth", "lesion_thickness"], expected_auc=0.5,
                  is_leak=bool(r2["roc_auc"] > 0.56))
        out["lesion_geometry_given_inclusion"] = r2

    blp = float(json.loads(str(data["config"])).get("benign_lesion_prob", 0.5))
    auc_inc = float(roc_auc_score(y, meta[:, col["has_inclusion"]]))
    out["has_inclusion_design_invariant"] = dict(
        roc_auc=auc_inc,
        expected_auc=1.0 - blp / 2.0,
        ok=bool(abs(auc_inc - (1.0 - blp / 2.0)) < 0.03),
        note="unobservable variable; predictive by design, cannot inflate the S11 scores",
    )
    return out


# --------------------------------------------------------------------------------------
# Permutation importance vs frequency
# --------------------------------------------------------------------------------------
def importance_vs_frequency(
    X: np.ndarray,
    y: np.ndarray,
    fcol: np.ndarray,
    kind: str = "rf",
    seed: int = DEFAULT_SEED,
    n_repeats: int = 5,
    n_eval: int = 2500,
    n_blocks: int = 24,
) -> Dict[str, object]:
    """Grouped permutation importance over contiguous frequency blocks.

    Single-feature permutation is the wrong tool here and gives a garbage answer.
    Adjacent frequency points are almost perfectly correlated, so permuting one of 602
    of them changes the prediction by nothing: every importance sits in the noise
    around zero, and the ranking is meaningless.

    Permuting a whole sub-band at once asks the question we actually care about --
    "if this slice of spectrum were not measured, how much accuracy would we lose?" --
    which is exactly the band-narrowing decision in open question 2. Columns inside a
    block are permuted with the SAME row order, so the within-block correlation
    structure survives and only the block-to-label relationship is destroyed.

    Importance is reported in AUC points lost, so the numbers are directly readable:
    0.01 means dropping that slice costs one AUC point.

    Note the two antenna bands overlap over 8-12 GHz, so blocks in that range contain
    columns from both.
    """
    rng = np.random.default_rng(seed)
    n = y.size
    perm = rng.permutation(n)
    n_te = min(n_eval, n // 3)
    te, tr = perm[:n_te], perm[n_te:]

    m = make_model(kind, seed)
    m.fit(X[tr], y[tr])
    Xte, yte = X[te].copy(), y[te]
    base = float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))

    finite = np.isfinite(fcol)
    edges = np.linspace(np.min(fcol[finite]), np.max(fcol[finite]), n_blocks + 1)
    centres, imps, stds = [], [], []
    for b in range(n_blocks):
        lo, hi = edges[b], edges[b + 1]
        cols = np.where(finite & (fcol >= lo) & (fcol < hi if b < n_blocks - 1 else fcol <= hi))[0]
        if cols.size == 0:
            continue
        drops = []
        for _ in range(n_repeats):
            saved = Xte[:, cols].copy()
            order = rng.permutation(Xte.shape[0])
            Xte[:, cols] = saved[order]  # one shared row order across the block
            drops.append(base - roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
            Xte[:, cols] = saved
        centres.append(0.5 * (lo + hi))
        imps.append(float(np.mean(drops)))
        stds.append(float(np.std(drops)))

    # Engineered scalars (set D) are not tied to one frequency; treat them as one group.
    scalar_imp = 0.0
    if np.any(~finite):
        cols = np.where(~finite)[0]
        drops = []
        for _ in range(n_repeats):
            saved = Xte[:, cols].copy()
            order = rng.permutation(Xte.shape[0])
            Xte[:, cols] = saved[order]
            drops.append(base - roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
            Xte[:, cols] = saved
        scalar_imp = float(np.mean(drops))

    return dict(
        freqs_hz=centres,
        importance=imps,
        importance_std=stds,
        block_width_hz=float(edges[1] - edges[0]),
        scalar_importance=scalar_imp,
        baseline_auc=base,
        n_repeats=n_repeats,
        n_blocks=n_blocks,
        n_eval=int(n_te),
        model=kind,
        units="AUC points lost when this frequency block is permuted",
    )


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_importance(imp: Dict[str, Dict[str, object]], path: str, bands) -> None:
    """Importance in AUC points lost, not normalised -- the absolute scale is the point.

    A block whose bar is inside the shaded noise floor is a slice of spectrum the
    classifier is not using, and therefore a candidate for deleting from the hardware.
    """
    plt = _plt()
    import features as _feat

    fig, ax = plt.subplots(figsize=(10, 4.8))
    colors = {"A": "tab:red", "B": "tab:blue", "C": "tab:orange", "D": "tab:green"}
    for fs, r in imp.items():
        fq = np.array(r["freqs_hz"]) / 1e9
        v = np.array(r["importance"])
        e = np.array(r.get("importance_std", np.zeros_like(v)))
        w = float(r.get("block_width_hz", 1e9)) / 1e9
        ax.errorbar(fq, v, yerr=e, lw=1.4, marker="o", ms=4, capsize=2,
                    color=colors.get(fs),
                    label=f"{_feat.FEATURE_SET_SHORT.get(fs, fs)}  base AUC {r['baseline_auc']:.3f}")
        if r.get("scalar_importance"):
            ax.axhline(r["scalar_importance"], color=colors.get(fs), ls=":", lw=1.0)
    ax.axhline(0.0, color="k", lw=0.8)

    for b, (f0, f1, _n) in tp.BANDS.items():
        if b in [str(x) for x in np.asarray(bands)]:
            ax.axvspan(f0 / 1e9, f1 / 1e9, color="k", alpha=0.045)
    ax.set_xlabel("frequency, GHz  (blocks; the two antenna bands overlap over 8-12 GHz)")
    ax.set_ylabel("AUC points lost when the block is permuted")
    ax.set_title("Where the discriminative information lives\n"
                 "(blocks at zero are spectrum the classifier is not using -- "
                 "candidates for cutting the hardware band)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(oof: Dict[str, np.ndarray], y: np.ndarray, path: str, title: str) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    for fs, p in oof.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, lw=1.4, label=f"set {fs} (AUC {roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.axhline(0.95, color="r", ls=":", lw=0.9)
    ax.text(0.02, 0.955, "95% sensitivity", color="r", fontsize=7, va="bottom")
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("sensitivity")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_example_spectra(data: Dict[str, object], s11_noisy: np.ndarray, path: str) -> None:
    plt = _plt()
    f = np.asarray(data["f"]) / 1e9
    y = np.asarray(data["y"]).astype(int)
    clean = np.asarray(data["s11"])
    bs = generate.band_slices_from(data)

    fig, ax = plt.subplots(2, 2, figsize=(11, 6.4), sharex="col")
    for bi, sl in enumerate(bs):
        fb = f[sl]
        for lbl, colr, nm in ((0, "tab:blue", "benign"), (1, "tab:red", "malignant")):
            m = 20 * np.log10(np.abs(clean[y == lbl][:, sl]) + 1e-12)
            mu, sd = m.mean(0), m.std(0)
            ax[0, bi].plot(fb, mu, color=colr, label=nm)
            ax[0, bi].fill_between(fb, mu - sd, mu + sd, color=colr, alpha=0.15)
            ph = np.unwrap(np.angle(clean[y == lbl][:, sl]), axis=1)
            ax[1, bi].plot(fb, np.rad2deg(ph.mean(0)), color=colr, label=nm)
        ax[0, bi].set_ylabel("|S11|, dB")
        ax[1, bi].set_ylabel("unwrapped phase, deg")
        ax[1, bi].set_xlabel("GHz")
        ax[0, bi].set_title(f"band {bi}: class means +/- 1 sd (clean)")
        for a in (ax[0, bi], ax[1, bi]):
            a.grid(alpha=0.3)
        ax[0, bi].legend(fontsize=8)
    fig.suptitle("Class separation is far smaller than physiological spread -- as it should be", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Train and compare feature sets.")
    ap.add_argument("--data", default="data/dataset.npz")
    ap.add_argument("--out", default="results/metrics_main.json")
    ap.add_argument("--sets", nargs="+", default=list(feat.FEATURE_SETS))
    ap.add_argument("--models", nargs="+", default=["rf", "xgb"], choices=["rf", "xgb"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--phase-err-deg", type=float, default=None,
                    help="override the nominal total rms phase error")
    ap.add_argument("--snr-db", type=float, default=None)
    ap.add_argument("--n", type=int, default=None, help="subsample the dataset")
    ap.add_argument("--band", default="both",
                    help="restrict to one antenna band ('low', 'mid') to price it "
                         "individually; 'both' uses the full grid")
    ap.add_argument("--fmin", type=float, default=None,
                    help="drop frequencies below this (Hz), applied after --band; use "
                         "with --fmax to price a narrowed instrument band")
    ap.add_argument("--fmax", type=float, default=None, help="drop frequencies above this (Hz)")
    ap.add_argument("--skip-importance", action="store_true")
    ap.add_argument("--skip-leakage-audit", action="store_true")
    ap.add_argument("--importance-only", action="store_true",
                    help="redo only the importance stage and merge it into an existing "
                         "results json; skips cross-validation entirely")
    ap.add_argument("--importance-sets", nargs="+", default=["A", "B"])
    ap.add_argument("--ac-tolerance", type=float, default=0.02,
                    help="max allowed |AUC(A) - AUC(C)| before the run is declared untrustworthy")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    # Figures are named after the metrics file, so an ablation run cannot silently
    # overwrite the main run's figures with a restricted-band version of itself.
    stem = os.path.splitext(os.path.basename(args.out))[0]
    sfx = "" if stem in ("metrics_main", "metrics") else "_" + stem.replace("metrics_", "")
    fig_roc_path = f"results/fig_roc{sfx}.png"
    fig_spec_path = f"results/fig_example_s11{sfx}.png"
    fig_imp_path = f"results/fig_importance_vs_ghz{sfx}.png"

    data = generate.load_dataset(args.data)
    if args.n:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(np.asarray(data["y"]).size, size=args.n, replace=False)
        data = dict(data)
        data["s11"] = np.asarray(data["s11"])[idx]
        data["y"] = np.asarray(data["y"])[idx]
        data["meta"] = np.asarray(data["meta"])[idx]

    if args.band != "both":
        names = [str(b) for b in np.asarray(data["bands"])]
        if args.band not in names:
            raise SystemExit(f"band {args.band!r} not in dataset (has {names})")
        j = names.index(args.band)
        a, b = np.asarray(data["band_edges"])[j]
        data = dict(data)
        data["s11"] = np.asarray(data["s11"])[:, a:b]
        data["f"] = np.asarray(data["f"])[a:b]
        data["bands"] = np.array([args.band])
        data["band_edges"] = np.array([[0, b - a]])

    if args.fmin is not None or args.fmax is not None:
        # Applied per band so the band structure (and per-band phase unwrapping) survives.
        f_all = np.asarray(data["f"])
        keep = np.ones(f_all.size, bool)
        if args.fmin is not None:
            keep &= f_all >= args.fmin
        if args.fmax is not None:
            keep &= f_all <= args.fmax
        if keep.sum() < 8:
            raise SystemExit(f"frequency window keeps only {keep.sum()} points")
        edges, i = [], 0
        for a, b in np.asarray(data["band_edges"]):
            nb = int(keep[a:b].sum())
            if nb:
                edges.append([i, i + nb])
                i += nb
        data = dict(data)
        data["s11"] = np.asarray(data["s11"])[:, keep]
        data["f"] = f_all[keep]
        data["band_edges"] = np.array(edges)
        data["bands"] = np.array([
            str(b) for (a2, b2), b in zip(np.asarray(data["band_edges"]), np.asarray(data["bands"]))
        ][: len(edges)]) if len(edges) == len(np.asarray(data["bands"])) else data["bands"][: len(edges)]

    y = np.asarray(data["y"]).astype(int)
    cfg = json.loads(str(data["config"]))
    f_now = np.asarray(data["f"])
    window = f"  [{f_now.min()/1e9:.1f}-{f_now.max()/1e9:.1f} GHz, {f_now.size} pts]"
    print(f"dataset: {args.data}  n={y.size}  bands={cfg['bands']}  seed={cfg['seed']}"
          + (f"  RESTRICTED TO {args.band.upper()} BAND" if args.band != "both" else "")
          + window)

    noisy = prepare_noisy(data, seed=args.seed, phase_err_deg=args.phase_err_deg, snr_db=args.snr_db)
    noise_cfg = dict(generate.DEFAULT_NOISE)
    if args.phase_err_deg is not None:
        noise_cfg["phase_err_deg"] = args.phase_err_deg
    if args.snr_db is not None:
        noise_cfg["snr_db"] = args.snr_db
    print(f"measurement realism: SNR {noise_cfg['snr_db']} dB, "
          f"phase error {noise_cfg['phase_err_deg']} deg rms, "
          f"gain {noise_cfg['mag_gain_std']*100:.1f}%")

    print("\nextracting features ...")
    F = build_features(noisy, data, args.sets)
    for fs, (X, cols, _fc) in F.items():
        print(f"  set {fs}: {X.shape[1]:5d} features  -- {feat.FEATURE_SET_DESCRIPTIONS[fs]}")

    if args.importance_only:
        with open(args.out) as fh:
            results = json.load(fh)
        imp = {}
        for fs in args.importance_sets:
            X, _cols, fc = F[fs]
            print(f"  set {fs} ... ", end="", flush=True)
            t0 = time.time()
            imp[fs] = importance_vs_frequency(X, y, fc, kind="rf", seed=args.seed)
            print(f"{time.time()-t0:.0f}s  (base AUC {imp[fs]['baseline_auc']:.4f}, "
                  f"max block {max(imp[fs]['importance']):.4f} AUC)")
        results["importance"] = imp
        plot_importance(imp, fig_imp_path, data["bands"])
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"wrote {fig_imp_path}")
        print(f"updated {args.out}")
        return 0

    results: Dict[str, object] = {
        "dataset": args.data,
        "dataset_config": cfg,
        "noise_config": noise_cfg,
        "band_restriction": {
            "band": args.band,
            "fmin_hz": args.fmin,
            "fmax_hz": args.fmax,
            "f_min_used_hz": float(f_now.min()),
            "f_max_used_hz": float(f_now.max()),
            "n_points": int(f_now.size),
        },
        "seed": args.seed,
        "k": args.k,
        "feature_sets": {fs: feat.FEATURE_SET_DESCRIPTIONS[fs] for fs in args.sets},
        "scores": {},
    }

    oof_by_set: Dict[str, np.ndarray] = {}
    for kind in args.models:
        print(f"\n=== {kind.upper()} , stratified {args.k}-fold, seed {args.seed} ===")
        hdr = (f"{'set':<5}{'acc':>8}{'sens':>8}{'spec':>8}{'AUC':>8}"
               f"{'spec@95sens':>13}{'sec':>7}")
        print(hdr)
        for fs in args.sets:
            X, _cols, _fc = F[fs]
            r = cross_validate(X, y, kind=kind, k=args.k, seed=args.seed, return_oof=True)
            if kind == args.models[-1]:
                oof_by_set[fs] = r.pop("_oof")
            else:
                r.pop("_oof", None)
            results["scores"].setdefault(kind, {})[fs] = r
            print(f"{fs:<5}{r['accuracy']:>8.4f}{r['sensitivity']:>8.4f}{r['specificity']:>8.4f}"
                  f"{r['roc_auc']:>8.4f}{r['specificity_at_95_sensitivity']:>13.4f}{r['fit_seconds']:>7.1f}")

        # A vs C sanity check
        if "A" in args.sets and "C" in args.sets:
            a = results["scores"][kind]["A"]["roc_auc"]
            c = results["scores"][kind]["C"]["roc_auc"]
            ok = abs(a - c) <= args.ac_tolerance
            results.setdefault("ac_equivalence", {})[kind] = dict(
                auc_A=a, auc_C=c, delta=abs(a - c), tolerance=args.ac_tolerance, ok=bool(ok)
            )
            print(f"A/C equivalence: |{a:.4f} - {c:.4f}| = {abs(a-c):.4f} "
                  f"({'OK' if ok else 'FAIL -- check phase unwrapping/scaling; run is not trustworthy'})")

    # --- leakage audit ---------------------------------------------------------------
    if not args.skip_leakage_audit:
        print("\n=== leakage audit: nuisance parameters only (must be ~0.5 AUC) ===")
        la = leakage_audit(data, seed=args.seed)
        results["leakage_audit"] = la
        for name in ("stack_geometry", "lesion_geometry_given_inclusion"):
            if name not in la:
                continue
            r = la[name]
            flag = "LEAK -- investigate before believing anything above" if r["is_leak"] else "OK"
            print(f"  {name:<32} AUC {r['roc_auc']:.4f}  acc {r['accuracy']:.4f}   {flag}")
        d = la["has_inclusion_design_invariant"]
        print(f"  {'has_inclusion (design, not a leak)':<32} AUC {d['roc_auc']:.4f}  "
              f"expected {d['expected_auc']:.4f}   {'OK' if d['ok'] else 'GENERATOR MISMATCH'}")

    # --- permutation importance -------------------------------------------------------
    if not args.skip_importance:
        print("\n=== permutation importance vs frequency ===")
        imp: Dict[str, Dict[str, object]] = {}
        for fs in args.importance_sets:
            if fs not in F:
                continue
            X, _cols, fc = F[fs]
            print(f"  set {fs} ... ", end="", flush=True)
            t0 = time.time()
            imp[fs] = importance_vs_frequency(X, y, fc, kind="rf", seed=args.seed)
            print(f"{time.time()-t0:.0f}s")
        results["importance"] = imp
        if imp:
            plot_importance(imp, fig_imp_path, data["bands"])
            print(f"  wrote {fig_imp_path}")

    # --- figures ----------------------------------------------------------------------
    if oof_by_set:
        plot_roc(oof_by_set, y, fig_roc_path,
                 f"ROC, {args.models[-1].upper()}, phase err {noise_cfg['phase_err_deg']} deg rms")
        print(f"wrote {fig_roc_path}")
    plot_example_spectra(data, noisy, fig_spec_path)
    print(f"wrote {fig_spec_path}")

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
