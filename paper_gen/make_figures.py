from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

EXPLICIT = ROOT / "artifacts" / "ancillary" / "void" / "void_prism_joint_explicit_multiembed_20260212_001503UTC" / "tables" / "results.json"
BATTERY = ROOT / "outputs" / "void_prism_signal_battery_l2o_placebo_20260212_rerun" / "tables" / "battery_results.json"
MAP_PLACEBO = ROOT / "outputs" / "void_prism_map_placebo_20260212_64x2" / "tables" / "map_placebo_results.json"
SPLIT_REPL = ROOT / "outputs" / "void_prism_split_replication_20260212_5000" / "tables" / "split_replication_results.json"


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


def _cmap() -> dict[str, str]:
    return {
        "minimal": "#0f766e",
        "slip_allowed": "#b45309",
        "screening_allowed": "#1d4ed8",
    }


def make_figure_delta_by_seed(explicit_rows: list[dict]) -> None:
    rows = sorted(explicit_rows, key=lambda r: (_run_key(r["run"]), r["embedding"]))
    runs = sorted({str(r["run"]) for r in rows}, key=_run_key)
    embeds = _embedding_order([str(r["embedding"]) for r in rows])
    cmap = _cmap()

    x = np.arange(len(runs), dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for emb in embeds:
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
    cmap = _cmap()

    keys = [
        ("perm_null", "Permutation null"),
        ("sign_null", "Block-sign null"),
        ("placebo_data_perm_null", "Data-placebo perm."),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.1), sharey=True)
    for j, (key, title) in enumerate(keys):
        ax = axes[j]
        for i, emb in enumerate(embeds):
            y = []
            for run in runs:
                rr = [r for r in rows if str(r["run"]) == run and str(r["embedding"]) == emb]
                y.append(float(rr[0][key]["p_upper"]))
            y = np.array(y, dtype=float)
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
        ax.set_title(title)

    axes[0].set_ylabel("Upper-tail p-value")
    fig.suptitle("Fast battery null-test diagnostics (5 seeds)", y=1.02)
    fig.tight_layout()
    fig.savefig(HERE / "fig_battery_pvalues.pdf")
    fig.savefig(HERE / "fig_battery_pvalues.png", dpi=220)
    plt.close(fig)


def make_figure_robustness(battery: dict) -> None:
    rows = battery["rows"]
    embeds = _embedding_order([str(r["embedding"]) for r in rows])
    cmap = _cmap()

    loo_nonpos = {}
    l2o_nonpos = {}
    split_means = {}
    split_keys = ["low_z", "high_z", "small_rv", "large_rv"]
    for emb in embeds:
        rr = [r for r in rows if str(r["embedding"]) == emb]
        loo_vals = []
        l2o_vals = []
        for r in rr:
            loo = np.array(r["leave_one_block_out"]["all"], dtype=float)
            l2o = np.array([float(x["delta"]) for x in r["leave_two_blocks_out"]["all"]], dtype=float)
            loo_vals.append(float(np.mean(loo <= 0.0)))
            l2o_vals.append(float(np.mean(l2o <= 0.0)))
        loo_nonpos[emb] = np.asarray(loo_vals, dtype=float)
        l2o_nonpos[emb] = np.asarray(l2o_vals, dtype=float)

        sm = {}
        for k in split_keys:
            vals = [float(r["split_consistency"][k]) for r in rr if k in r["split_consistency"]]
            sm[k] = float(np.mean(vals)) if vals else float("nan")
        split_means[emb] = sm

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))

    x = np.arange(len(embeds), dtype=float)
    width = 0.36
    loo_m = np.array([np.mean(loo_nonpos[e]) for e in embeds], dtype=float)
    loo_s = np.array([np.std(loo_nonpos[e], ddof=1) if loo_nonpos[e].size > 1 else 0.0 for e in embeds], dtype=float)
    l2o_m = np.array([np.mean(l2o_nonpos[e]) for e in embeds], dtype=float)
    l2o_s = np.array([np.std(l2o_nonpos[e], ddof=1) if l2o_nonpos[e].size > 1 else 0.0 for e in embeds], dtype=float)
    colors = [cmap.get(e, "#333333") for e in embeds]
    axes[0].bar(x - width / 2.0, loo_m, width=width, yerr=loo_s, color=colors, alpha=0.9, capsize=3, label="LOO")
    axes[0].bar(x + width / 2.0, l2o_m, width=width, yerr=l2o_s, color=colors, alpha=0.45, capsize=3, label="L2O")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(embeds, rotation=18, ha="right")
    axes[0].set_ylabel("Fraction nonpositive")
    axes[0].set_ylim(0.0, 0.5)
    axes[0].set_title("Block-drop fragility")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    sx = np.arange(len(split_keys), dtype=float)
    width2 = 0.22
    for i, emb in enumerate(embeds):
        vals = np.array([split_means[emb][k] for k in split_keys], dtype=float)
        axes[1].bar(sx + (i - 1) * width2, vals, width=width2, color=cmap.get(emb, None), label=emb, alpha=0.9)
    axes[1].axhline(0.0, color="black", lw=0.9)
    axes[1].set_xticks(sx)
    axes[1].set_xticklabels(["low z", "high z", "small Rv", "large Rv"], rotation=0)
    axes[1].set_ylabel(r"Mean split $\Delta\mathrm{LPD}$")
    axes[1].set_title("Intra-suite split consistency")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(HERE / "fig_robustness_checks.pdf")
    fig.savefig(HERE / "fig_robustness_checks.png", dpi=220)
    plt.close(fig)


def make_figure_map_placebo(map_placebo: dict) -> None:
    rows = map_placebo["rows"]
    runs = sorted({str(r["run"]) for r in rows}, key=_run_key)
    embeds = _embedding_order([str(r["embedding"]) for r in rows])
    cmap = _cmap()
    modes = [("rotate", "Global rotation placebo"), ("random_mask", "Mask-randomized centers")]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1), sharey=True)
    for j, (mode, title) in enumerate(modes):
        ax = axes[j]
        for i, emb in enumerate(embeds):
            y = []
            for run in runs:
                rr = [r for r in rows if str(r["run"]) == run and str(r["embedding"]) == emb]
                y.append(float(rr[0]["map_placebo_nulls"][mode]["p_upper"]))
            y = np.array(y, dtype=float)
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
        ax.set_title(title)

    axes[0].set_ylabel("Upper-tail p-value")
    fig.suptitle("Map-level placebo diagnostics (5 seeds)", y=1.02)
    fig.tight_layout()
    fig.savefig(HERE / "fig_map_placebo_pvalues.pdf")
    fig.savefig(HERE / "fig_map_placebo_pvalues.png", dpi=220)
    plt.close(fig)


def make_figure_split_replication(split_repl: dict) -> None:
    per = split_repl["per_split_summary"]
    comp = split_repl["comparison"]
    split_names = list(per.keys())
    embeds = _embedding_order(list(per[split_names[0]].keys()))
    cmap = _cmap()

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))

    x = np.arange(len(embeds), dtype=float)
    width = 0.36
    vals_a = np.array([float(per[split_names[0]][e]["mean_delta_lpd_vs_gr"]) for e in embeds], dtype=float)
    vals_b = np.array([float(per[split_names[1]][e]["mean_delta_lpd_vs_gr"]) for e in embeds], dtype=float)
    axes[0].bar(x - width / 2.0, vals_a, width=width, color="#0f766e", alpha=0.9, label=split_names[0])
    axes[0].bar(x + width / 2.0, vals_b, width=width, color="#b45309", alpha=0.9, label=split_names[1])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(embeds, rotation=18, ha="right")
    axes[0].set_ylabel(r"Mean $\Delta\mathrm{LPD}$")
    axes[0].set_title("Per-split means")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    ratio = np.array([float(comp[e]["amp_ratio"]) for e in embeds], dtype=float)
    axes[1].bar(x, ratio, color=[cmap.get(e, "#333333") for e in embeds], alpha=0.9)
    axes[1].axhline(2.0, color="crimson", ls="--", lw=1.0, alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(embeds, rotation=18, ha="right")
    axes[1].set_ylabel("SGC/NGC amplitude ratio")
    axes[1].set_title("Split tension")
    axes[1].set_ylim(0.0, max(18.0, float(np.max(ratio)) + 1.0))
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(HERE / "fig_split_replication.pdf")
    fig.savefig(HERE / "fig_split_replication.png", dpi=220)
    plt.close(fig)


def main() -> int:
    explicit_rows = _load_json(EXPLICIT)
    battery = _load_json(BATTERY)
    map_placebo = _load_json(MAP_PLACEBO)
    split_repl = _load_json(SPLIT_REPL)

    make_figure_delta_by_seed(explicit_rows)
    make_figure_battery_pvals(battery)
    make_figure_robustness(battery)
    make_figure_map_placebo(map_placebo)
    make_figure_split_replication(split_repl)
    print(str(HERE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
