from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Keep threaded BLAS/OpenMP from oversubscribing cores.
# Must be set before importing numpy/scipy stack.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from entropy_horizon_recon.cache import DataPaths
from entropy_horizon_recon.optical_bias.ingest_planck_lensing import load_planck_kappa
from entropy_horizon_recon.optical_bias.maps import _require_healpy, radec_to_healpix
from entropy_horizon_recon.sirens import load_mu_forward_posterior
from entropy_horizon_recon.void_prism import eg_gr_baseline_concat_from_background, predict_EG_void_concat_from_mu
from entropy_horizon_recon.void_prism_maps import eg_void_from_spectra, load_void_catalog_csv, measure_void_prism_spectra, void_overdensity_map_from_pix


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _logmeanexp(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    m = float(np.max(x))
    return float(m + np.log(np.mean(np.exp(x - m))))


def _p_upper(null_values: np.ndarray, observed: float) -> float:
    z = np.asarray(null_values, dtype=float)
    if z.size == 0:
        return float("nan")
    return float((1 + np.sum(z >= observed)) / (z.size + 1))


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


class GaussianScore:
    """Fast repeated scoring with fixed observation/covariance."""

    def __init__(self, y: np.ndarray, cov: np.ndarray, fit_amplitude: bool) -> None:
        y = np.asarray(y, dtype=float)
        C = np.asarray(cov, dtype=float)
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
        y_w = np.linalg.solve(L, y)

        self.y = y
        self.cov = C
        self.fit_amplitude = bool(fit_amplitude)
        self._L = L
        self._const = const
        self._y_w = y_w
        self._yy = float(np.dot(y_w, y_w))

    def score(self, pred_draws: np.ndarray) -> tuple[float, np.ndarray | None]:
        pred = np.asarray(pred_draws, dtype=float)
        if pred.ndim != 2 or pred.shape[1] != self.y.size:
            raise ValueError("pred_draws must have shape (n_draws, n_dim).")
        P_w = np.linalg.solve(self._L, pred.T).T
        if not self.fit_amplitude:
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


def _resolve_path(path_like: str, *, project_root: Path) -> Path:
    p = Path(str(path_like))
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p.resolve()
    p2 = project_root / p
    if p2.exists():
        return p2.resolve()
    raise FileNotFoundError(f"Could not resolve path: {path_like}")


def _parse_zbin_idx(name: str) -> int:
    m = re.search(r"zbin(\d+)", str(name))
    if m is None:
        raise ValueError(f"Block name does not encode zbin index: {name}")
    return int(m.group(1))


def _as_float_or_none(x: Any) -> float | None:
    if x is None:
        return None
    try:
        if isinstance(x, str) and x.strip().lower() in ("none", "null", ""):
            return None
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        return None
    return None


def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Draw a uniform random rotation matrix in SO(3)."""
    u1, u2, u3 = rng.random(3)
    qx = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
    qy = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
    qz = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
    qw = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)
    x, y, z, w = qx, qy, qz, qw
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


@dataclass(frozen=True)
class BlockPlaceboSpec:
    name: str
    zbin_idx: int
    ell: np.ndarray
    pix: np.ndarray
    weights: np.ndarray
    m_block: np.ndarray
    good_pix: np.ndarray
    vec: np.ndarray


@dataclass(frozen=True)
class PlaceboContext:
    nside: int
    lmax: int | None
    bin_edges: np.ndarray
    prefactor: float
    eg_sign: float
    kappa: np.ndarray
    theta_list: list[np.ndarray]
    specs: list[BlockPlaceboSpec]


def _prepare_context(
    *,
    suite_meta: dict[str, Any],
    blocks: list[dict[str, Any]],
    project_root: Path,
    void_csv_override: str | None,
) -> PlaceboContext:
    hp = _require_healpy()
    nside = int(suite_meta["nside"])
    lmax = int(suite_meta["lmax"]) if suite_meta.get("lmax") is not None else None
    bin_edges = np.asarray(suite_meta["bin_edges"], dtype=int)
    prefactor = float(suite_meta["prefactor"])
    eg_sign = float(suite_meta.get("eg_sign", 1.0))
    frame = str(suite_meta.get("frame", "galactic"))

    theta_paths_raw = suite_meta.get("theta_fits")
    if not isinstance(theta_paths_raw, list) or len(theta_paths_raw) == 0:
        raise ValueError("suite_meta.theta_fits missing/empty.")
    theta_paths = [_resolve_path(str(p), project_root=project_root) for p in theta_paths_raw]

    z_edges = np.asarray(suite_meta.get("z_edges", []), dtype=float)
    if z_edges.size < 2:
        raise ValueError("suite_meta.z_edges missing/invalid.")
    n_zbins = int(z_edges.size - 1)

    theta_list = [np.asarray(hp.read_map(str(p), verbose=False), dtype=float) for p in theta_paths]
    if len(theta_list) == 1:
        theta_list = [theta_list[0] for _ in range(n_zbins)]
    if len(theta_list) != n_zbins:
        raise ValueError(f"theta map count mismatch: got {len(theta_list)} need 1 or {n_zbins}.")
    theta_list = [hp.ud_grade(t, nside) if hp.get_nside(t) != nside else t for t in theta_list]

    theta_mask_paths_raw = suite_meta.get("theta_mask_fits")
    if theta_mask_paths_raw is None:
        theta_masks = [np.ones_like(theta_list[0], dtype=float) for _ in range(n_zbins)]
    else:
        if not isinstance(theta_mask_paths_raw, list) or len(theta_mask_paths_raw) == 0:
            raise ValueError("suite_meta.theta_mask_fits invalid.")
        theta_mask_paths = [_resolve_path(str(p), project_root=project_root) for p in theta_mask_paths_raw]
        theta_masks = [np.asarray(hp.read_map(str(p), verbose=False), dtype=float) for p in theta_mask_paths]
        if len(theta_masks) == 1:
            theta_masks = [theta_masks[0] for _ in range(n_zbins)]
        if len(theta_masks) != n_zbins:
            raise ValueError(f"theta mask count mismatch: got {len(theta_masks)} need 1 or {n_zbins}.")
        theta_masks = [hp.ud_grade(t, nside) if hp.get_nside(t) != nside else t for t in theta_masks]

    kappa_source = str(suite_meta.get("kappa_source", "planck"))
    if kappa_source == "planck":
        planck = load_planck_kappa(paths=DataPaths(project_root), nside_out=nside)
        kappa = np.asarray(planck.kappa_map, dtype=float)
        mask = np.asarray(planck.mask, dtype=float) if planck.mask is not None else np.ones_like(kappa, dtype=float)
    else:
        kappa_path = _resolve_path(kappa_source, project_root=project_root)
        kappa = np.asarray(hp.read_map(str(kappa_path), verbose=False), dtype=float)
        if hp.get_nside(kappa) != nside:
            kappa = hp.ud_grade(kappa, nside)
        mask = np.ones_like(kappa, dtype=float)

    extra_mask = suite_meta.get("extra_mask_fits")
    if extra_mask is not None:
        extra_path = _resolve_path(str(extra_mask), project_root=project_root)
        em = np.asarray(hp.read_map(str(extra_path), verbose=False), dtype=float)
        if hp.get_nside(em) != nside:
            em = hp.ud_grade(em, nside)
        mask = mask * em

    void_csv_raw = void_csv_override if void_csv_override else str(suite_meta.get("void_csv"))
    if not void_csv_raw:
        raise ValueError("void CSV path missing. Pass --void-csv or ensure suite_meta.void_csv is set.")
    void_csv = _resolve_path(str(void_csv_raw), project_root=project_root)

    Rv_col = suite_meta.get("Rv_col")
    weight_col = suite_meta.get("weight_col")
    Rv_col_s = None if Rv_col is None else str(Rv_col)
    weight_col_s = None if weight_col is None else str(weight_col)
    cat = load_void_catalog_csv(
        str(void_csv),
        ra_col=str(suite_meta.get("ra_col", "ra")),
        dec_col=str(suite_meta.get("dec_col", "dec")),
        z_col=str(suite_meta.get("z_col", "z")),
        Rv_col=Rv_col_s,
        weight_col=weight_col_s,
    )

    z = np.asarray(cat.z, dtype=float)
    m_global = (z >= float(z_edges[0])) & (z <= float(z_edges[-1])) & np.isfinite(z)
    if not np.any(m_global):
        raise ValueError("No catalog objects in suite z range.")
    ra = np.asarray(cat.ra_deg, dtype=float)[m_global]
    dec = np.asarray(cat.dec_deg, dtype=float)[m_global]
    z = z[m_global]
    Rv = np.asarray(cat.Rv, dtype=float)[m_global] if cat.Rv is not None else None
    w_all = np.asarray(cat.weight, dtype=float)[m_global] if cat.weight is not None else np.ones(np.sum(m_global), dtype=float)

    pix_all = radec_to_healpix(ra, dec, nside=nside, frame=frame, nest=False)

    specs: list[BlockPlaceboSpec] = []
    for b in blocks:
        name = str(b.get("name"))
        env = b.get("env") or {}
        zmin = float(env["z_min"])
        zmax = float(env["z_max"])
        m = (z >= zmin) & (z < zmax)

        rv_min = _as_float_or_none(env.get("rv_min"))
        rv_max = _as_float_or_none(env.get("rv_max"))
        if rv_min is not None or rv_max is not None:
            if Rv is None:
                raise ValueError(f"Block {name} has Rv selection but catalog has no Rv column.")
            if rv_min is not None:
                # Match suite construction: 'large' bins use strict lower bound at split.
                if rv_max is None:
                    m &= Rv > rv_min
                else:
                    m &= Rv >= rv_min
            if rv_max is not None:
                m &= Rv <= rv_max

        pix = np.asarray(pix_all[m], dtype=int)
        ww = np.asarray(w_all[m], dtype=float)
        if pix.size == 0:
            raise ValueError(f"Block {name} selects zero voids; cannot build placebo maps.")

        zbin_idx = _parse_zbin_idx(name)
        tmask = np.asarray(theta_masks[zbin_idx], dtype=float)
        m_block = np.asarray(mask, dtype=float) * tmask
        good_pix = np.flatnonzero(m_block > 0.0).astype(int)
        if good_pix.size == 0:
            raise ValueError(f"Block {name} has empty analysis mask.")

        vx, vy, vz = hp.pix2vec(nside, pix, nest=False)
        vec = np.vstack([vx, vy, vz]).astype(float)
        ell = np.asarray(b["ell"], dtype=int)

        specs.append(
            BlockPlaceboSpec(
                name=name,
                zbin_idx=zbin_idx,
                ell=ell,
                pix=pix,
                weights=ww,
                m_block=m_block,
                good_pix=good_pix,
                vec=vec,
            )
        )

    return PlaceboContext(
        nside=nside,
        lmax=lmax,
        bin_edges=bin_edges,
        prefactor=prefactor,
        eg_sign=eg_sign,
        kappa=np.asarray(kappa, dtype=float),
        theta_list=[np.asarray(t, dtype=float) for t in theta_list],
        specs=specs,
    )


def _make_placebo_vector(mode: str, *, ctx: PlaceboContext, rng: np.random.Generator) -> np.ndarray:
    hp = _require_healpy()
    if mode not in ("rotate", "random_mask"):
        raise ValueError(f"Unknown placebo mode: {mode}")

    R = _random_rotation_matrix(rng) if mode == "rotate" else None
    pieces: list[np.ndarray] = []
    for sp in ctx.specs:
        if mode == "rotate":
            assert R is not None
            vec_r = R @ sp.vec
            pix_p = hp.vec2pix(ctx.nside, vec_r[0], vec_r[1], vec_r[2], nest=False)
        else:
            pix_p = rng.choice(sp.good_pix, size=sp.pix.size, replace=True)

        vmap = void_overdensity_map_from_pix(
            np.asarray(pix_p, dtype=int),
            nside=ctx.nside,
            mask=sp.m_block,
            weights=sp.weights,
            allow_empty=True,
        )
        res = measure_void_prism_spectra(
            kappa_map=ctx.kappa,
            theta_map=ctx.theta_list[sp.zbin_idx],
            void_delta_map=vmap,
            mask=sp.m_block,
            lmax=ctx.lmax,
            bin_edges=ctx.bin_edges,
        )
        eg = ctx.eg_sign * eg_void_from_spectra(
            cl_kappa_void=np.asarray(res["cl_kappa_void"], dtype=float),
            cl_theta_void=np.asarray(res["cl_theta_void"], dtype=float),
            prefactor=ctx.prefactor,
        )
        if eg.size != sp.ell.size:
            raise RuntimeError(f"Block {sp.name} produced unexpected ell dimension: {eg.size} vs {sp.ell.size}")
        pieces.append(np.asarray(eg, dtype=float))
    return np.concatenate(pieces, axis=0)


def _score_delta(
    *,
    scorer: GaussianScore,
    pred: np.ndarray,
    gr: np.ndarray,
) -> tuple[float, float, float]:
    lpd, _ = scorer.score(pred)
    lpd_gr, _ = scorer.score(gr)
    return float(lpd), float(lpd_gr), float(lpd - lpd_gr)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Map-level placebo battery for void-prism signal validation. "
            "Generates rotated/randomized-center placebo EG vectors and compares observed delta vs placebo nulls."
        )
    )
    ap.add_argument("--run-dir", action="append", required=True, help="Finished run directory (contains samples/mu_forward_posterior.npz).")
    ap.add_argument("--suite-json", required=True, help="suite_joint.json from measurement stage.")
    ap.add_argument("--project-root", default=None, help="Project/data root for resolving relative suite paths (default: current working directory).")
    ap.add_argument("--void-csv", default=None, help="Override void catalog CSV path (otherwise suite_meta.void_csv).")
    ap.add_argument("--out", default=None, help="Output directory (default outputs/void_prism_map_placebo_<UTC>).")
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
    ap.add_argument("--max-draws", type=int, default=256, help="Posterior draw cap.")
    ap.add_argument(
        "--placebo-mode",
        action="append",
        choices=["rotate", "random_mask"],
        default=None,
        help="Placebo mode(s) to include. Repeat flag. Default: rotate,random_mask.",
    )
    ap.add_argument("--n-placebo", type=int, default=64, help="Placebo draws per mode.")
    ap.add_argument("--seed", type=int, default=20260212)
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path("outputs") / f"void_prism_map_placebo_{_utc_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tab_dir = out_dir / "tables"
    tab_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(args.project_root).resolve() if args.project_root else Path.cwd().resolve()
    embeddings = list(args.embedding) if args.embedding else ["minimal", "slip_allowed", "screening_allowed"]
    modes = list(args.placebo_mode) if args.placebo_mode else ["rotate", "random_mask"]
    n_placebo = int(args.n_placebo)
    if n_placebo <= 0:
        raise ValueError("--n-placebo must be > 0.")

    suite_path = Path(args.suite_json)
    meta, blocks, y_obs, cov = _load_suite(suite_path)
    rng = np.random.default_rng(int(args.seed))
    fit_amp = bool(args.fit_amplitude)

    print(
        f"[map_placebo] preparing context blocks={len(blocks)} modes={modes} n_placebo={n_placebo} "
        f"project_root={project_root}",
        flush=True,
    )
    ctx = _prepare_context(
        suite_meta=meta,
        blocks=blocks,
        project_root=project_root,
        void_csv_override=args.void_csv,
    )

    print("[map_placebo] generating placebo vectors...", flush=True)
    placebo_vectors: dict[str, list[np.ndarray]] = {m: [] for m in modes}
    for m in modes:
        for i in range(n_placebo):
            y_p = _make_placebo_vector(m, ctx=ctx, rng=rng)
            placebo_vectors[m].append(np.asarray(y_p, dtype=float))
            if (i + 1) % max(1, n_placebo // 8) == 0 or (i + 1) == n_placebo:
                print(f"[map_placebo] mode={m} generated {i+1}/{n_placebo}", flush=True)

    placebo_scorers: dict[str, list[GaussianScore]] = {
        m: [GaussianScore(y=v, cov=cov, fit_amplitude=fit_amp) for v in vv] for m, vv in placebo_vectors.items()
    }
    scorer_obs = GaussianScore(y=y_obs, cov=cov, fit_amplitude=fit_amp)

    block_tuples = [(float(b["z_eff"]), np.asarray(b["ell"], dtype=int)) for b in blocks]
    rows: list[dict[str, Any]] = []
    for run_dir in args.run_dir:
        run_path = Path(run_dir)
        run_name = run_path.name
        print(f"[map_placebo] run={run_name} loading posterior...", flush=True)
        post = load_mu_forward_posterior(run_path)

        print(f"[map_placebo] run={run_name} predicting GR...", flush=True)
        pred_gr = eg_gr_baseline_concat_from_background(post, blocks=block_tuples, max_draws=int(args.max_draws))
        for emb in embeddings:
            print(f"[map_placebo] run={run_name} embedding={emb} predicting...", flush=True)
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
            lpd, lpd_gr, delta_obs = _score_delta(scorer=scorer_obs, pred=pred, gr=pred_gr)

            placebo_out: dict[str, Any] = {}
            for mode in modes:
                vals = np.empty(len(placebo_scorers[mode]), dtype=float)
                for i, sc in enumerate(placebo_scorers[mode]):
                    lpd_p, lpd_gr_p, d = _score_delta(scorer=sc, pred=pred, gr=pred_gr)
                    vals[i] = d
                    # avoid unused warnings while keeping transparent structure
                    _ = (lpd_p, lpd_gr_p)
                placebo_out[mode] = {
                    "summary": _summ(vals),
                    "p_upper": _p_upper(vals, float(delta_obs)),
                }

            row = {
                "run": run_name,
                "embedding": emb,
                "fit_amplitude": fit_amp,
                "max_draws": int(args.max_draws),
                "lpd_obs": float(lpd),
                "lpd_gr_obs": float(lpd_gr),
                "delta_obs": float(delta_obs),
                "map_placebo_nulls": placebo_out,
            }
            rows.append(row)
            (tab_dir / "map_placebo_partial.json").write_text(json.dumps(rows, indent=2))
            mode_bits = " ".join([f"p_{m}={row['map_placebo_nulls'][m]['p_upper']:.4f}" for m in modes])
            print(f"[map_placebo] run={run_name} emb={emb} delta_obs={delta_obs:+.4f} {mode_bits}", flush=True)

    by_embedding: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_embedding.setdefault(str(r["embedding"]), []).append(r)

    embedding_summary: list[dict[str, Any]] = []
    for emb, rr in by_embedding.items():
        deltas = np.array([float(x["delta_obs"]) for x in rr], dtype=float)
        mode_summary: dict[str, Any] = {}
        for mode in modes:
            pp = np.array([float(x["map_placebo_nulls"][mode]["p_upper"]) for x in rr], dtype=float)
            mode_summary[mode] = {"p_upper": _summ(pp)}
        embedding_summary.append(
            {
                "embedding": emb,
                "n_runs": int(len(rr)),
                "delta_obs": _summ(deltas),
                "positive_delta_fraction": float(np.mean(deltas > 0.0)),
                "placebo_modes": mode_summary,
            }
        )

    out = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "suite_json": str(args.suite_json),
            "project_root": str(project_root),
            "void_csv_override": str(args.void_csv) if args.void_csv else None,
            "run_dirs": [str(p) for p in args.run_dir],
            "convention": str(args.convention),
            "embeddings": embeddings,
            "fit_amplitude": fit_amp,
            "max_draws": int(args.max_draws),
            "placebo_modes": modes,
            "n_placebo": int(args.n_placebo),
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
    (tab_dir / "map_placebo_results.json").write_text(json.dumps(out, indent=2))

    md_lines = [
        "# Void-Prism Map-Level Placebo Battery",
        "",
        f"- Created: `{out['meta']['created_utc']}`",
        f"- Fit amplitude: `{fit_amp}`",
        f"- Draw cap: `{int(args.max_draws)}`",
        f"- Placebo modes: `{','.join(modes)}`",
        f"- Placebo draws per mode: `{int(args.n_placebo)}`",
        "",
        "## Embedding Summary",
        "",
    ]
    for s in embedding_summary:
        md_lines.append(f"### `{s['embedding']}`")
        md_lines.append(f"- runs: `{s['n_runs']}`")
        md_lines.append(f"- delta_obs mean: `{s['delta_obs']['mean']:.6f}`")
        md_lines.append(f"- positive delta fraction: `{s['positive_delta_fraction']:.3f}`")
        for mode in modes:
            pm = s["placebo_modes"][mode]["p_upper"]["mean"]
            md_lines.append(f"- `{mode}` p_upper mean: `{pm:.4f}`")
        md_lines.append("")
    (tab_dir / "map_placebo_report.md").write_text("\n".join(md_lines))

    print(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
