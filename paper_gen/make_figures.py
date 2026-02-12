from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

EXPLICIT = ROOT / "artifacts" / "ancillary" / "void" / "void_prism_joint_explicit_multiembed_20260212_001503UTC" / "tables" / "results.json"
BATTERY = ROOT / "artifacts" / "ancillary" / "void" / "void_prism_signal_battery_20260212_003206UTC" / "tables" / "battery_results.json"


def _run_key(run: str) -> tuple[int, str]:
    s = str(run)
    if "start" in s:
        try:
            return (int(s.split("start")[-1]), s)
        except Exception:
            return (10**9, s)
    return (10**9, s)


def _load_json(path: Path):
    return json.loads(path.read_text())


def _embedding_order(values: list[str]) -> list[str]:
    pref = ["minimal", "slip_allowed", "screening_allowed"]
    seen = set(values)
    out = [x for x in pref if x in seen]
    out.extend([x for x in sorted(seen) if x not in out])
    return out


def make_figure_delta_by_seed(explicit_rows: list[dict]) -> None:
    rows = sorted(explicit_rows, key=lambda r: (_run_key(r["run"]), r["embedding"]))
    runs = sorted({str(r["run"]) for r in rows}, key=_run_key)
    embeds = _embedding_order([str(r["embedding"]) for r in rows])
    cmap = {
        "minimal": "#0f766e",
        "slip_allowed": "#b45309",
        "screening_allowed": "#7c3aed",
    }

    x = np.arange(len(runs), dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, emb in enumerate(embeds):
        y = []
        for run in runs:
            rr = [r for r in rows if str(r["run"]) == run and str(r["embedding"]) == emb]
            y.append(float(rr[0]["delta_lpd_vs_gr"]))
        y = np.array(y, dtype=float)
        ax.plot(x, y, marker="o", lw=1.8, color=cmap.get(emb, None), label=emb)
        ax.axhline(np.mean(y), ls="--", lw=0.8, color=cmap.get(emb, None), alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(runs, rotation=0)
    ax.set_ylabel(r"$\Delta \mathrm{LPD}_{\rm vs\,GR}$")
    ax.set_xlabel("Run seed")
    ax.set_title("Joint scoring across seeds and embeddings")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / "fig_delta_by_seed.pdf")
    fig.savefig(HERE / "fig_delta_by_seed.png", dpi=220)
    plt.close(fig)


def make_figure_battery_pvals(battery: dict) -> None:
    rows = battery["rows"]
    runs = sorted({str(r["run"]) for r in rows}, key=_run_key)
    embeds = _embedding_order([str(r["embedding"]) for r in rows])
    cmap = {
        "minimal": "#0f766e",
        "slip_allowed": "#b45309",
        "screening_allowed": "#7c3aed",
    }

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1), sharey=True)
    for j, key in enumerate(["perm_null", "sign_null"]):
        ax = axes[j]
        for i, emb in enumerate(embeds):
            y = []
            for run in runs:
                rr = [r for r in rows if str(r["run"]) == run and str(r["embedding"]) == emb]
                y.append(float(rr[0][key]["p_upper"]))
            y = np.array(y, dtype=float)
            # small horizontal jitter to avoid overplotting
            xx = np.full(y.shape, i, dtype=float) + np.linspace(-0.08, 0.08, y.size)
            ax.scatter(xx, y, s=22, color=cmap.get(emb, None), alpha=0.9)
            ax.errorbar(
                [i],
                [np.mean(y)],
                yerr=[np.std(y, ddof=1)],
                fmt="s",
                ms=4.5,
                lw=1.0,
                color=cmap.get(emb, None),
                capsize=3,
            )

        ax.axhline(0.05, color="crimson", ls="--", lw=0.9, alpha=0.7)
        ax.axhline(0.01, color="crimson", ls=":", lw=0.9, alpha=0.7)
        ax.set_xticks(np.arange(len(embeds), dtype=float))
        ax.set_xticklabels(embeds, rotation=18, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25)
        ax.set_title("Permutation null" if key == "perm_null" else "Block-sign null")

    axes[0].set_ylabel("Upper-tail p-value")
    fig.suptitle("Fast battery null-test diagnostics (5 seeds)", y=1.02)
    fig.tight_layout()
    fig.savefig(HERE / "fig_battery_pvalues.pdf")
    fig.savefig(HERE / "fig_battery_pvalues.png", dpi=220)
    plt.close(fig)


def make_figure_robustness(battery: dict) -> None:
    rows = battery["rows"]
    embeds = _embedding_order([str(r["embedding"]) for r in rows])
    cmap = {
        "minimal": "#0f766e",
        "slip_allowed": "#b45309",
        "screening_allowed": "#7c3aed",
    }

    loo_min = {}
    split_means = {}
    keys = ["low_z", "high_z", "small_rv", "large_rv"]
    for emb in embeds:
        rr = [r for r in rows if str(r["embedding"]) == emb]
        loo_min[emb] = np.array([float(r["leave_one_block_out"]["summary"]["min"]) for r in rr], dtype=float)
        sm = {}
        for k in keys:
            vals = [float(r["split_consistency"][k]) for r in rr if k in r["split_consistency"]]
            sm[k] = float(np.mean(vals)) if vals else float("nan")
        split_means[emb] = sm

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))

    # Panel A: LOO minimum delta per embedding.
    x = np.arange(len(embeds), dtype=float)
    means = np.array([np.mean(loo_min[e]) for e in embeds], dtype=float)
    sds = np.array([np.std(loo_min[e], ddof=1) if loo_min[e].size > 1 else 0.0 for e in embeds], dtype=float)
    colors = [cmap.get(e, "#333333") for e in embeds]
    axes[0].bar(x, means, yerr=sds, color=colors, alpha=0.85, capsize=3)
    axes[0].axhline(0.0, color="black", lw=0.9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(embeds, rotation=18, ha="right")
    axes[0].set_ylabel(r"LOO minimum $\Delta\mathrm{LPD}$")
    axes[0].set_title("Leave-one-block-out fragility")
    axes[0].grid(axis="y", alpha=0.25)

    # Panel B: split means.
    sx = np.arange(len(keys), dtype=float)
    width = 0.22
    for i, emb in enumerate(embeds):
        vals = np.array([split_means[emb][k] for k in keys], dtype=float)
        axes[1].bar(sx + (i - 1) * width, vals, width=width, color=cmap.get(emb, None), label=emb, alpha=0.9)
    axes[1].axhline(0.0, color="black", lw=0.9)
    axes[1].set_xticks(sx)
    axes[1].set_xticklabels(["low z", "high z", "small Rv", "large Rv"], rotation=0)
    axes[1].set_ylabel(r"Mean split $\Delta\mathrm{LPD}$")
    axes[1].set_title("Split consistency")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(HERE / "fig_robustness_checks.pdf")
    fig.savefig(HERE / "fig_robustness_checks.png", dpi=220)
    plt.close(fig)


def main() -> int:
    explicit_rows = _load_json(EXPLICIT)
    battery = _load_json(BATTERY)

    make_figure_delta_by_seed(explicit_rows)
    make_figure_battery_pvals(battery)
    make_figure_robustness(battery)
    print(str(HERE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
