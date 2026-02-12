from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from entropy_horizon_recon.optical_bias.maps import _require_astropy


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


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


def _parse_rows_for_split(
    *,
    void_csv: Path,
    ra_col: str,
    dec_col: str,
) -> tuple[list[dict[str, str]], np.ndarray]:
    rows: list[dict[str, str]] = []
    ra: list[float] = []
    dec: list[float] = []
    with void_csv.open("r", newline="") as f:
        rdr = csv.DictReader(f)
        if rdr.fieldnames is None:
            raise ValueError(f"No header in CSV: {void_csv}")
        for r in rdr:
            if ra_col not in r or dec_col not in r:
                raise ValueError(f"Missing required columns {ra_col}/{dec_col} in {void_csv}")
            try:
                ra_v = float(r[ra_col])
                dec_v = float(r[dec_col])
            except Exception:
                continue
            if not (np.isfinite(ra_v) and np.isfinite(dec_v)):
                continue
            rows.append(r)
            ra.append(ra_v)
            dec.append(dec_v)
    if not rows:
        raise ValueError(f"No valid rows parsed from {void_csv}")
    return rows, np.vstack([np.asarray(ra, dtype=float), np.asarray(dec, dtype=float)]).T


def _split_masks(*, ra_dec: np.ndarray, split_axis: str) -> dict[str, np.ndarray]:
    ra = ra_dec[:, 0]
    dec = ra_dec[:, 1]
    if split_axis == "galactic_b":
        SkyCoord, u = _require_astropy()
        c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        b = np.asarray(c.galactic.b.deg, dtype=float)
        return {
            "NGC_b>=0": b >= 0.0,
            "SGC_b<0": b < 0.0,
        }
    if split_axis == "dec":
        return {
            "DEC>=0": dec >= 0.0,
            "DEC<0": dec < 0.0,
        }
    raise ValueError(f"Unknown split axis: {split_axis}")


def _write_split_csv(
    *,
    out_csv: Path,
    rows: list[dict[str, str]],
    keep_mask: np.ndarray,
    columns: list[str],
) -> int:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_csv.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=columns)
        wr.writeheader()
        for i, r in enumerate(rows):
            if not bool(keep_mask[i]):
                continue
            wr.writerow({k: r.get(k, "") for k in columns})
            n += 1
    return n


def _run(cmd: list[str], *, cwd: Path) -> None:
    print("[split_repl] running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _embedding_summary_from_joint(results_json: Path) -> dict[str, Any]:
    rows = json.loads(results_json.read_text())
    by: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(str(r["embedding"]), []).append(r)
    out: dict[str, Any] = {}
    for emb, rr in by.items():
        d = np.array([float(x["delta_lpd_vs_gr"]) for x in rr], dtype=float)
        out[emb] = {
            "n": int(d.size),
            "mean_delta_lpd_vs_gr": float(np.mean(d)),
            "std_delta_lpd_vs_gr": float(np.std(d, ddof=1)) if d.size > 1 else 0.0,
            "min_delta_lpd_vs_gr": float(np.min(d)),
            "max_delta_lpd_vs_gr": float(np.max(d)),
            "positive_fraction": float(np.mean(d > 0.0)),
        }
    return out


def _compare_two_splits(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    keys = sorted(set(a.keys()).intersection(b.keys()))
    comp: dict[str, Any] = {}
    for k in keys:
        ma = float(a[k]["mean_delta_lpd_vs_gr"])
        mb = float(b[k]["mean_delta_lpd_vs_gr"])
        same_sign = (ma > 0 and mb > 0) or (ma < 0 and mb < 0)
        abs_a = abs(ma)
        abs_b = abs(mb)
        if min(abs_a, abs_b) > 0.0:
            amp_ratio = max(abs_a, abs_b) / min(abs_a, abs_b)
        else:
            amp_ratio = float("inf")
        comp[k] = {
            "mean_a": ma,
            "mean_b": mb,
            "same_sign": bool(same_sign),
            "abs_diff": float(abs(ma - mb)),
            "amp_ratio": float(amp_ratio),
            "within_factor_2": bool(amp_ratio <= 2.0),
        }
    return comp


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Automated split replication runner. "
            "Builds disjoint sky-split catalogs, reruns measurement + joint scoring per split, and writes a comparison report."
        )
    )
    ap.add_argument("--suite-json", required=True, help="Reference suite_joint.json used to clone settings.")
    ap.add_argument("--run-dir", action="append", required=True, help="Finished run dir (repeat flag).")
    ap.add_argument("--project-root", default=None, help="Root for resolving relative data paths in suite meta (default cwd).")
    ap.add_argument("--python", default=sys.executable, help="Python interpreter for subprocess calls.")
    ap.add_argument("--split-axis", choices=["galactic_b", "dec"], default="galactic_b")
    ap.add_argument("--min-catalog-rows", type=int, default=100, help="Minimum rows required in each split catalog.")
    ap.add_argument(
        "--embedding",
        action="append",
        choices=["minimal", "slip_allowed", "screening_allowed"],
        default=None,
        help="Embedding(s) for joint scoring. Default: minimal,slip_allowed,screening_allowed.",
    )
    ap.add_argument("--convention", choices=["A", "B"], default="A")
    ap.add_argument("--eta0", type=float, default=1.12)
    ap.add_argument("--eta1", type=float, default=-0.18)
    ap.add_argument("--env-proxy", type=float, default=1.0)
    ap.add_argument("--env-alpha", type=float, default=0.25)
    ap.add_argument("--muP-highz", type=float, default=1.05)
    ap.add_argument("--fit-amplitude", action="store_true")
    ap.add_argument("--max-draws", type=int, default=2048)
    ap.add_argument("--jackknife-nside", type=int, default=None, help="Override jackknife nside from suite meta.")
    ap.add_argument("--n-proc", type=int, default=None, help="Override jackknife n_proc from suite meta.")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="If set, reuse existing split outputs (skip measurement/joint when results already exist).",
    )
    ap.add_argument("--out", default=None, help="Output directory (default outputs/void_prism_split_replication_<UTC>).")
    args = ap.parse_args()

    repo_root = Path.cwd().resolve()
    project_root = Path(args.project_root).resolve() if args.project_root else repo_root
    out_dir = Path(args.out) if args.out else Path("outputs") / f"void_prism_split_replication_{_utc_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tab_dir = out_dir / "tables"
    tab_dir.mkdir(parents=True, exist_ok=True)

    embeddings = list(args.embedding) if args.embedding else ["minimal", "slip_allowed", "screening_allowed"]
    suite_path = Path(args.suite_json)
    suite_meta, _blocks, _y, _C = _load_suite(suite_path)

    void_csv = _resolve_path(str(suite_meta["void_csv"]), project_root=project_root)
    ra_col = str(suite_meta.get("ra_col", "ra"))
    dec_col = str(suite_meta.get("dec_col", "dec"))
    z_col = str(suite_meta.get("z_col", "z"))
    rv_col_raw = suite_meta.get("Rv_col")
    wt_col_raw = suite_meta.get("weight_col")
    rv_col = None if rv_col_raw in (None, "", "none", "None") else str(rv_col_raw)
    wt_col = None if wt_col_raw in (None, "", "none", "None") else str(wt_col_raw)

    keep_cols = [ra_col, dec_col, z_col]
    if rv_col is not None:
        keep_cols.append(rv_col)
    if wt_col is not None:
        keep_cols.append(wt_col)

    rows, ra_dec = _parse_rows_for_split(void_csv=void_csv, ra_col=ra_col, dec_col=dec_col)
    split_masks = _split_masks(ra_dec=ra_dec, split_axis=str(args.split_axis))
    split_csvs: dict[str, Path] = {}
    split_counts: dict[str, int] = {}
    for split_name, m in split_masks.items():
        out_csv = tab_dir / f"void_catalog_{split_name.replace('/', '_')}.csv"
        n = _write_split_csv(out_csv=out_csv, rows=rows, keep_mask=m, columns=keep_cols)
        if n < int(args.min_catalog_rows):
            raise RuntimeError(f"Split {split_name} has only {n} rows (<{args.min_catalog_rows}).")
        split_csvs[split_name] = out_csv
        split_counts[split_name] = int(n)
        print(f"[split_repl] split={split_name} rows={n}", flush=True)

    theta_fits = [str(_resolve_path(str(p), project_root=project_root)) for p in list(suite_meta["theta_fits"])]
    theta_mask_fits_raw = suite_meta.get("theta_mask_fits")
    theta_mask_fits = (
        [str(_resolve_path(str(p), project_root=project_root)) for p in list(theta_mask_fits_raw)]
        if theta_mask_fits_raw is not None
        else None
    )
    extra_mask = suite_meta.get("extra_mask_fits")
    extra_mask_res = str(_resolve_path(str(extra_mask), project_root=project_root)) if extra_mask is not None else None
    kappa_source = str(suite_meta.get("kappa_source", "planck"))
    nside = int(suite_meta["nside"])
    kappa_fits: str | None = None
    mask_fits: str | None = None
    if kappa_source == "planck":
        # Prefer precomputed local Planck products if available in project_root.
        cand_kappa = project_root / "data" / "processed" / "planck_lensing" / f"planck2018_kappa_nside{nside}.fits"
        cand_mask = project_root / "data" / "processed" / "planck_lensing" / f"planck2018_mask_nside{nside}.fits"
        if cand_kappa.exists():
            kappa_fits = str(cand_kappa.resolve())
            if cand_mask.exists():
                mask_fits = str(cand_mask.resolve())
    else:
        kappa_fits = str(_resolve_path(kappa_source, project_root=project_root))

    z_edges = ",".join(str(x) for x in list(suite_meta["z_edges"]))
    bin_edges = ",".join(str(int(x)) for x in list(suite_meta["bin_edges"]))
    rv_split = str(suite_meta.get("rv_split_mode", "median"))
    frame = str(suite_meta.get("frame", "galactic"))
    lmax = suite_meta.get("lmax")
    prefactor = float(suite_meta.get("prefactor", 1.0))
    eg_sign = float(suite_meta.get("eg_sign", 1.0))

    jk_nside = int(args.jackknife_nside) if args.jackknife_nside is not None else int(suite_meta.get("jackknife_nside", 8))
    n_proc = int(args.n_proc) if args.n_proc is not None else int(suite_meta.get("n_proc", 1))

    created_utc = datetime.now(timezone.utc).isoformat()
    meta_common = {
        "created_utc": created_utc,
        "suite_json": str(args.suite_json),
        "project_root": str(project_root),
        "split_axis": str(args.split_axis),
        "run_dirs": [str(r) for r in args.run_dir],
        "embeddings": embeddings,
        "fit_amplitude": bool(args.fit_amplitude),
        "max_draws": int(args.max_draws),
        "convention": str(args.convention),
        "eta0": float(args.eta0),
        "eta1": float(args.eta1),
        "env_proxy": float(args.env_proxy),
        "env_alpha": float(args.env_alpha),
        "muP_highz": float(args.muP_highz),
        "jackknife_nside": int(jk_nside),
        "n_proc": int(n_proc),
        "split_catalog_rows": split_counts,
    }

    per_split_summary: dict[str, Any] = {}
    for split_name, split_csv in split_csvs.items():
        split_dir = out_dir / split_name.replace("/", "_")
        measure_out = split_dir / "measurement"
        joint_out = split_dir / "joint"
        measure_out.mkdir(parents=True, exist_ok=True)
        joint_out.mkdir(parents=True, exist_ok=True)

        split_suite = measure_out / "tables" / "suite_joint.json"
        res_json = joint_out / "tables" / "results.json"
        if bool(args.resume) and res_json.exists():
            print(f"[split_repl] split={split_name} using existing joint results: {res_json}", flush=True)
            per_split_summary[split_name] = _embedding_summary_from_joint(res_json)
        else:
            measure_cmd = [
                str(args.python),
                str(repo_root / "scripts" / "measure_void_prism_eg_suite_jackknife.py"),
                "--void-csv",
                str(split_csv),
                "--theta-fits",
                ",".join(theta_fits),
                "--ra-col",
                ra_col,
                "--dec-col",
                dec_col,
                "--z-col",
                z_col,
                "--Rv-col",
                rv_col if rv_col is not None else "none",
                "--weight-col",
                wt_col if wt_col is not None else "none",
                "--frame",
                frame,
                "--nside",
                str(nside),
                "--bin-edges",
                bin_edges,
                "--prefactor",
                str(prefactor),
                "--eg-sign",
                str(eg_sign),
                "--z-edges",
                z_edges,
                "--rv-split",
                rv_split,
                "--jackknife-nside",
                str(jk_nside),
                "--n-proc",
                str(n_proc),
                "--out-base",
                str(measure_out),
            ]
            if lmax is not None:
                measure_cmd.extend(["--lmax", str(int(lmax))])
            if theta_mask_fits is not None:
                measure_cmd.extend(["--theta-mask-fits", ",".join(theta_mask_fits)])
            if kappa_fits is None and kappa_source == "planck":
                measure_cmd.append("--planck")
            else:
                if kappa_fits is None:
                    raise RuntimeError("Non-planck kappa source but no kappa path resolved.")
                measure_cmd.extend(["--kappa-fits", kappa_fits])
                if mask_fits is not None:
                    measure_cmd.extend(["--mask-fits", mask_fits])
            if extra_mask_res is not None:
                measure_cmd.extend(["--extra-mask-fits", extra_mask_res])

            if not (bool(args.resume) and split_suite.exists()):
                _run(measure_cmd, cwd=repo_root)
                if not split_suite.exists():
                    raise FileNotFoundError(f"Expected split suite output missing: {split_suite}")
            else:
                print(f"[split_repl] split={split_name} using existing measurement: {split_suite}", flush=True)

            joint_cmd = [
                str(args.python),
                str(repo_root / "scripts" / "run_void_prism_eg_joint_test.py"),
            ]
            for rd in args.run_dir:
                joint_cmd.extend(["--run-dir", str(rd)])
            joint_cmd.extend(
                [
                    "--suite-json",
                    str(split_suite),
                    "--convention",
                    str(args.convention),
                    "--eta0",
                    str(float(args.eta0)),
                    "--eta1",
                    str(float(args.eta1)),
                    "--env-proxy",
                    str(float(args.env_proxy)),
                    "--env-alpha",
                    str(float(args.env_alpha)),
                    "--muP-highz",
                    str(float(args.muP_highz)),
                    "--max-draws",
                    str(int(args.max_draws)),
                    "--out",
                    str(joint_out),
                ]
            )
            if bool(args.resume):
                joint_cmd.append("--resume")
            if bool(args.fit_amplitude):
                joint_cmd.append("--fit-amplitude")
            for emb in embeddings:
                joint_cmd.extend(["--embedding", emb])

            _run(joint_cmd, cwd=repo_root)

            if not res_json.exists():
                raise FileNotFoundError(f"Expected joint results missing: {res_json}")
            per_split_summary[split_name] = _embedding_summary_from_joint(res_json)

        partial = {
            "meta": meta_common,
            "per_split_summary": per_split_summary,
            "status": {"completed_splits": list(per_split_summary.keys()), "expected_splits": list(split_csvs.keys())},
        }
        _atomic_write_text(tab_dir / "split_replication_partial.json", json.dumps(partial, indent=2))

    split_names = list(per_split_summary.keys())
    if len(split_names) != 2:
        raise RuntimeError(f"Expected exactly 2 splits, got {split_names}")
    comparison = _compare_two_splits(per_split_summary[split_names[0]], per_split_summary[split_names[1]])

    out = {
        "meta": {
            **meta_common,
        },
        "per_split_summary": per_split_summary,
        "comparison": comparison,
    }
    _atomic_write_text(tab_dir / "split_replication_results.json", json.dumps(out, indent=2))

    md = [
        "# Void-Prism Split Replication",
        "",
        f"- Created: `{out['meta']['created_utc']}`",
        f"- Split axis: `{out['meta']['split_axis']}`",
        f"- Split row counts: `{out['meta']['split_catalog_rows']}`",
        f"- Embeddings: `{','.join(embeddings)}`",
        f"- Fit amplitude: `{out['meta']['fit_amplitude']}`",
        "",
        "## Per-Split Means",
        "",
    ]
    for sn in split_names:
        md.append(f"### `{sn}`")
        for emb in embeddings:
            s = per_split_summary[sn][emb]
            md.append(
                f"- `{emb}`: mean=`{s['mean_delta_lpd_vs_gr']:+.6f}` std=`{s['std_delta_lpd_vs_gr']:.6f}` "
                f"positive_fraction=`{s['positive_fraction']:.3f}`"
            )
        md.append("")
    md.append("## Split Comparison")
    md.append("")
    md.append(f"Comparing `{split_names[0]}` vs `{split_names[1]}`")
    md.append("")
    for emb in embeddings:
        c = comparison[emb]
        md.append(
            f"- `{emb}`: same_sign=`{c['same_sign']}` within_factor_2=`{c['within_factor_2']}` "
            f"amp_ratio=`{c['amp_ratio']:.3f}` abs_diff=`{c['abs_diff']:.6f}`"
        )
    _atomic_write_text(tab_dir / "split_replication_report.md", "\n".join(md))

    print(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
