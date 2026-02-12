from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from entropy_horizon_recon.sirens import load_mu_forward_posterior
from entropy_horizon_recon.void_prism import eg_gr_baseline_concat_from_background, predict_EG_void_concat_from_mu

# Keep threaded BLAS/OpenMP from oversubscribing cores.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _logmeanexp(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    m = float(np.max(x))
    return float(m + np.log(np.mean(np.exp(x - m))))


def _p_upper(null_values: np.ndarray, observed: float) -> float:
    """Finite-sample one-sided upper-tail p-value with +1 correction."""
    z = np.asarray(null_values, dtype=float)
    if z.size == 0:
        return float("nan")
    return float((1 + np.sum(z >= observed)) / (z.size + 1))


@dataclass(frozen=True)
class GaussianScore:
    """Fast repeated scoring with fixed observation/covariance.

    We whiten once via Cholesky and then evaluate many draw-wise model predictions.
    """

    y: np.ndarray
    cov: np.ndarray
    fit_amplitude: bool

    def __post_init__(self) -> None:
        y = np.asarray(self.y, dtype=float)
        C = np.asarray(self.cov, dtype=float)
        if y.ndim != 1:
            raise ValueError("y must be 1D.")
        if C.shape != (y.size, y.size):
            raise ValueError("cov shape mismatch.")
        C = 0.5 * (C + C.T)
        jitter = 1e-12 * float(np.trace(C)) / max(1, y.size)
        C = C + np.eye(y.size) * jitter
        L = np.linalg.cholesky(C)
        logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
        const = -0.5 * (y.size * np.log(2.0 * np.pi) + logdet)

        # Precompute whitened data vector y_w = L^{-1} y.
        y_w = np.linalg.solve(L, y)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "cov", C)
        object.__setattr__(self, "_L", L)
        object.__setattr__(self, "_const", const)
        object.__setattr__(self, "_y_w", y_w)
        object.__setattr__(self, "_yy", float(np.dot(y_w, y_w)))

    def score(self, pred_draws: np.ndarray) -> tuple[float, np.ndarray | None]:
        pred = np.asarray(pred_draws, dtype=float)
        if pred.ndim != 2 or pred.shape[1] != self.y.size:
            raise ValueError("pred_draws must have shape (n_draws, n_dim).")

        # Whiten each draw: p_w = L^{-1} p.
        P_w = np.linalg.solve(self._L, pred.T).T
        if not self.fit_amplitude:
            # ||y_w - p_w||^2 = ||y_w||^2 + ||p_w||^2 - 2 y_w.p_w
            pp = np.sum(P_w * P_w, axis=1)
            py = P_w @ self._y_w
            sq = self._yy + pp - 2.0 * py
            logp = self._const - 0.5 * sq
            return _logmeanexp(logp), None

        py = P_w @ self._y_w
        pp = np.sum(P_w * P_w, axis=1)
        bad = (~np.isfinite(pp)) | (np.abs(pp) < 1e-30)
        A = py / pp
        A[bad] = 0.0
        # ||y_w - A p_w||^2 = ||y_w||^2 + A^2 ||p_w||^2 - 2A y_w.p_w
        sq = self._yy + (A * A) * pp - 2.0 * A * py
        logp = self._const - 0.5 * sq
        return _logmeanexp(logp), A


def _load_suite(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray]:
    d = json.loads(path.read_text())
    meta = d.get("meta") or {}
    blocks = d.get("blocks")
    y_obs = d.get("y_obs")
    cov = d.get("cov")
    if not isinstance(blocks, list) or y_obs is None or cov is None:
        raise ValueError("suite_joint.json missing required keys blocks/y_obs/cov.")
    y = np.asarray(y_obs, dtype=float)
    C = np.asarray(cov, dtype=float)
    if C.shape != (y.size, y.size):
        raise ValueError("Covariance shape mismatch.")
    return meta, blocks, y, C


def _block_indices(blocks: list[dict[str, Any]], dim: int) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for i, b in enumerate(blocks):
        sl = b.get("slice")
        if not (isinstance(sl, list) and len(sl) == 2):
            raise ValueError(f"Missing/invalid slice for block {i}.")
        lo, hi = int(sl[0]), int(sl[1])
        if not (0 <= lo < hi <= dim):
            raise ValueError(f"Invalid slice {sl} for block {i}.")
        out.append(np.arange(lo, hi, dtype=int))
    return out


def _select_cols(mat: np.ndarray, cols: np.ndarray) -> np.ndarray:
    return np.asarray(mat[:, cols], dtype=float)


def _score_delta(
    *,
    y: np.ndarray,
    cov: np.ndarray,
    pred: np.ndarray,
    gr: np.ndarray,
    fit_amplitude: bool,
) -> tuple[float, float, float]:
    scorer = GaussianScore(y=y, cov=cov, fit_amplitude=fit_amplitude)
    lpd, _ = scorer.score(pred)
    lpd_gr, _ = scorer.score(gr)
    return float(lpd), float(lpd_gr), float(lpd - lpd_gr)


def _subset_from_blocks(
    y: np.ndarray,
    cov: np.ndarray,
    pred: np.ndarray,
    gr: np.ndarray,
    block_cols: list[np.ndarray],
    keep_blocks: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cols = np.concatenate([block_cols[i] for i in keep_blocks], axis=0)
    y_s = np.asarray(y[cols], dtype=float)
    C_s = np.asarray(cov[np.ix_(cols, cols)], dtype=float)
    p_s = _select_cols(pred, cols)
    g_s = _select_cols(gr, cols)
    return y_s, C_s, p_s, g_s


def _block_kind(name: str) -> str:
    n = str(name)
    if "_small_" in n:
        return "small"
    if "_large_" in n:
        return "large"
    return "other"


def _summ(vals: np.ndarray) -> dict[str, float]:
    x = np.asarray(vals, dtype=float)
    if x.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Fast evidence battery for void-prism signal stability. "
            "Runs block-permutation nulls, block-sign nulls, leave-one-block-out, and split-consistency checks."
        )
    )
    ap.add_argument("--run-dir", action="append", required=True, help="Finished run directory (contains samples/mu_forward_posterior.npz).")
    ap.add_argument("--suite-json", required=True, help="suite_joint.json from measurement stage.")
    ap.add_argument("--out", default=None, help="Output directory (default outputs/void_prism_signal_battery_<UTC>).")
    ap.add_argument("--convention", choices=["A", "B"], default="A")
    ap.add_argument(
        "--embedding",
        action="append",
        choices=["minimal", "slip_allowed", "screening_allowed"],
        default=None,
        help="Embedding(s) to test. Default: minimal,slip_allowed,screening_allowed.",
    )
    ap.add_argument("--eta0", type=float, default=1.12)
    ap.add_argument("--eta1", type=float, default=-0.18)
    ap.add_argument("--env-proxy", type=float, default=1.0)
    ap.add_argument("--env-alpha", type=float, default=0.25)
    ap.add_argument("--muP-highz", type=float, default=1.05)
    ap.add_argument("--fit-amplitude", action="store_true", help="Use amplitude-fitted channel for primary scoring.")
    ap.add_argument("--max-draws", type=int, default=256, help="Posterior draw cap (default 256 for speed).")
    ap.add_argument("--n-perm", type=int, default=200, help="Block-permutation null draws.")
    ap.add_argument("--n-sign", type=int, default=200, help="Block-sign null draws.")
    ap.add_argument("--seed", type=int, default=20260212)
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path("outputs") / f"void_prism_signal_battery_{_utc_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tab_dir = out_dir / "tables"
    tab_dir.mkdir(parents=True, exist_ok=True)

    embeddings = list(args.embedding) if args.embedding else ["minimal", "slip_allowed", "screening_allowed"]
    meta, blocks, y_obs, cov = _load_suite(Path(args.suite_json))
    block_cols = _block_indices(blocks, dim=y_obs.size)
    block_tuples = [(float(b["z_eff"]), np.asarray(b["ell"], dtype=int)) for b in blocks]

    z_eff = np.array([float(b["z_eff"]) for b in blocks], dtype=float)
    z_med = float(np.median(z_eff))
    low_blocks = [i for i, z in enumerate(z_eff) if z <= z_med]
    high_blocks = [i for i, z in enumerate(z_eff) if z > z_med]
    small_blocks = [i for i, b in enumerate(blocks) if _block_kind(str(b.get("name", ""))) == "small"]
    large_blocks = [i for i, b in enumerate(blocks) if _block_kind(str(b.get("name", ""))) == "large"]

    rng = np.random.default_rng(int(args.seed))
    fit_amp = bool(args.fit_amplitude)

    rows: list[dict[str, Any]] = []
    for run_dir in args.run_dir:
        run_path = Path(run_dir)
        run_name = run_path.name
        print(f"[battery] run={run_name} loading posterior...", flush=True)
        post = load_mu_forward_posterior(run_path)

        print(f"[battery] run={run_name} predicting GR...", flush=True)
        pred_gr = eg_gr_baseline_concat_from_background(post, blocks=block_tuples, max_draws=int(args.max_draws))
        for emb in embeddings:
            print(f"[battery] run={run_name} embedding={emb} predicting...", flush=True)
            pred = predict_EG_void_concat_from_mu(
                post,
                blocks=block_tuples,
                convention=args.convention,  # type: ignore[arg-type]
                embedding=emb,  # type: ignore[arg-type]
                eta0=float(args.eta0),
                eta1=float(args.eta1),
                env_proxy=float(args.env_proxy),
                env_alpha=float(args.env_alpha),
                muP_highz=float(args.muP_highz),
                max_draws=int(args.max_draws),
            )

            _, _, delta_obs = _score_delta(y=y_obs, cov=cov, pred=pred, gr=pred_gr, fit_amplitude=fit_amp)

            scorer_full = GaussianScore(y=y_obs, cov=cov, fit_amplitude=fit_amp)

            # 1) Block-permutation null (misalignment null).
            perm_vals = np.empty(int(args.n_perm), dtype=float)
            for k in range(int(args.n_perm)):
                perm = rng.permutation(len(block_cols))
                cols = np.concatenate([block_cols[i] for i in perm], axis=0)
                lpd_perm, _ = scorer_full.score(_select_cols(pred, cols))
                lpd_gr_perm, _ = scorer_full.score(_select_cols(pred_gr, cols))
                perm_vals[k] = float(lpd_perm - lpd_gr_perm)

            # 2) Block-sign null (coherent-sign null).
            sign_vals = np.empty(int(args.n_sign), dtype=float)
            for k in range(int(args.n_sign)):
                s_block = rng.choice(np.array([-1.0, 1.0]), size=len(block_cols), replace=True)
                s_vec = np.concatenate([np.full(c.size, s_block[i], dtype=float) for i, c in enumerate(block_cols)], axis=0)
                lpd_s, _ = scorer_full.score(pred * s_vec.reshape((1, -1)))
                lpd_gr_s, _ = scorer_full.score(pred_gr * s_vec.reshape((1, -1)))
                sign_vals[k] = float(lpd_s - lpd_gr_s)

            # 3) Leave-one-block-out robustness.
            loo = []
            for b in range(len(block_cols)):
                keep = [i for i in range(len(block_cols)) if i != b]
                y_s, C_s, p_s, g_s = _subset_from_blocks(y_obs, cov, pred, pred_gr, block_cols, keep)
                _, _, d_s = _score_delta(y=y_s, cov=C_s, pred=p_s, gr=g_s, fit_amplitude=fit_amp)
                loo.append(d_s)
            loo_vals = np.asarray(loo, dtype=float)

            # 4) Split consistency.
            split = {}
            if low_blocks and high_blocks:
                y_l, C_l, p_l, g_l = _subset_from_blocks(y_obs, cov, pred, pred_gr, block_cols, low_blocks)
                y_h, C_h, p_h, g_h = _subset_from_blocks(y_obs, cov, pred, pred_gr, block_cols, high_blocks)
                split["low_z"] = _score_delta(y=y_l, cov=C_l, pred=p_l, gr=g_l, fit_amplitude=fit_amp)[2]
                split["high_z"] = _score_delta(y=y_h, cov=C_h, pred=p_h, gr=g_h, fit_amplitude=fit_amp)[2]
            if small_blocks and large_blocks:
                y_s, C_s, p_s, g_s = _subset_from_blocks(y_obs, cov, pred, pred_gr, block_cols, small_blocks)
                y_lg, C_lg, p_lg, g_lg = _subset_from_blocks(y_obs, cov, pred, pred_gr, block_cols, large_blocks)
                split["small_rv"] = _score_delta(y=y_s, cov=C_s, pred=p_s, gr=g_s, fit_amplitude=fit_amp)[2]
                split["large_rv"] = _score_delta(y=y_lg, cov=C_lg, pred=p_lg, gr=g_lg, fit_amplitude=fit_amp)[2]

            row = {
                "run": run_name,
                "embedding": emb,
                "fit_amplitude": fit_amp,
                "max_draws": int(args.max_draws),
                "delta_obs": float(delta_obs),
                "perm_null": {
                    "summary": _summ(perm_vals),
                    "p_upper": _p_upper(perm_vals, float(delta_obs)),
                },
                "sign_null": {
                    "summary": _summ(sign_vals),
                    "p_upper": _p_upper(sign_vals, float(delta_obs)),
                },
                "leave_one_block_out": {
                    "summary": _summ(loo_vals),
                    "all": loo_vals.tolist(),
                    "all_positive": bool(np.all(loo_vals > 0.0)),
                },
                "split_consistency": {
                    **split,
                    "all_available_positive": bool(all(float(v) > 0.0 for v in split.values())) if split else False,
                },
            }
            rows.append(row)
            (tab_dir / "battery_partial.json").write_text(json.dumps(rows, indent=2))
            print(
                f"[battery] run={run_name} emb={emb} "
                f"delta_obs={delta_obs:+.4f} "
                f"p_perm={row['perm_null']['p_upper']:.4f} "
                f"p_sign={row['sign_null']['p_upper']:.4f} "
                f"loo_all_pos={row['leave_one_block_out']['all_positive']}",
                flush=True,
            )

    # Aggregate summary by embedding.
    by_embedding: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_embedding.setdefault(str(r["embedding"]), []).append(r)

    embedding_summary: list[dict[str, Any]] = []
    for emb, rr in by_embedding.items():
        deltas = np.array([float(x["delta_obs"]) for x in rr], dtype=float)
        p_perm = np.array([float(x["perm_null"]["p_upper"]) for x in rr], dtype=float)
        p_sign = np.array([float(x["sign_null"]["p_upper"]) for x in rr], dtype=float)
        loo_ok = np.array([bool(x["leave_one_block_out"]["all_positive"]) for x in rr], dtype=bool)
        split_ok = np.array([bool(x["split_consistency"]["all_available_positive"]) for x in rr], dtype=bool)
        embedding_summary.append(
            {
                "embedding": emb,
                "n_runs": int(len(rr)),
                "delta_obs": _summ(deltas),
                "perm_p_upper": _summ(p_perm),
                "sign_p_upper": _summ(p_sign),
                "positive_delta_fraction": float(np.mean(deltas > 0.0)),
                "loo_all_positive_fraction": float(np.mean(loo_ok)),
                "split_all_positive_fraction": float(np.mean(split_ok)),
            }
        )

    out = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "suite_json": str(args.suite_json),
            "run_dirs": [str(p) for p in args.run_dir],
            "convention": str(args.convention),
            "embeddings": embeddings,
            "fit_amplitude": fit_amp,
            "max_draws": int(args.max_draws),
            "n_perm": int(args.n_perm),
            "n_sign": int(args.n_sign),
            "seed": int(args.seed),
            "eta0": float(args.eta0),
            "eta1": float(args.eta1),
            "env_proxy": float(args.env_proxy),
            "env_alpha": float(args.env_alpha),
            "muP_highz": float(args.muP_highz),
            "suite_meta": meta,
        },
        "rows": rows,
        "embedding_summary": embedding_summary,
    }
    (tab_dir / "battery_results.json").write_text(json.dumps(out, indent=2))

    md_lines = [
        "# Void-Prism Fast Signal Battery",
        "",
        f"- Created: `{out['meta']['created_utc']}`",
        f"- Fit amplitude: `{fit_amp}`",
        f"- Draw cap: `{int(args.max_draws)}`",
        f"- Null reps: perm=`{int(args.n_perm)}`, sign=`{int(args.n_sign)}`",
        "",
        "## Embedding Summary",
        "",
    ]
    for s in embedding_summary:
        md_lines.append(f"### `{s['embedding']}`")
        md_lines.append(f"- runs: `{s['n_runs']}`")
        md_lines.append(f"- delta_obs mean: `{s['delta_obs']['mean']:.6f}`")
        md_lines.append(f"- perm p_upper mean: `{s['perm_p_upper']['mean']:.4f}`")
        md_lines.append(f"- sign p_upper mean: `{s['sign_p_upper']['mean']:.4f}`")
        md_lines.append(f"- positive delta fraction: `{s['positive_delta_fraction']:.3f}`")
        md_lines.append(f"- LOO all-positive fraction: `{s['loo_all_positive_fraction']:.3f}`")
        md_lines.append(f"- split all-positive fraction: `{s['split_all_positive_fraction']:.3f}`")
        md_lines.append("")
    (tab_dir / "battery_report.md").write_text("\n".join(md_lines))

    print(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
