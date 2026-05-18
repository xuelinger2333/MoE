"""G6 round-3 — Multi-domain stability of cross-layer MI.

For each (model, domain) pair, compute MI(L, L+1) − within-seq null and
the per-source entropy reduction distribution. Render F14 (per-model
MI grouped by domain) and F15 (per-source distributions overlaid by
domain). Write multidomain_mi.json with the comparison table.

Looks for traces under:
    outputs/traces/m8_<tag>_<domain>/
plus the existing NL traces (m4_qwen_wikitext, m1_deepseek_wikitext,
m7_olmoe_wikitext) treated as domain="nl".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.cross_layer_mi import (
    all_layer_pairs_mi,
    joint_counts,
    per_source_entropy_reduction,
    top1_pivot,
)
from src.probes.trace_writer import load_trace
from src.utils import get_logger

logger = get_logger("multidomain_mi")
REPO_ROOT = Path(__file__).resolve().parent.parent


# (model_label, model_tag, domain → trace_dir)
def _domain_dirs(repo_root: Path) -> dict:
    return {
        ("Qwen1.5-MoE-A2.7B", "qwen"): {
            "nl": repo_root / "outputs/traces/m4_qwen_wikitext",
            "code": repo_root / "outputs/traces/m8_qwen_code",
            "math": repo_root / "outputs/traces/m8_qwen_math",
        },
        ("DeepSeek-V2-Lite", "deepseek"): {
            "nl": repo_root / "outputs/traces/m1_deepseek_wikitext",
            "code": repo_root / "outputs/traces/m8_deepseek_code",
            "math": repo_root / "outputs/traces/m8_deepseek_math",
        },
        ("OLMoE-1B-7B", "olmoe"): {
            "nl": repo_root / "outputs/traces/m7_olmoe_wikitext",
            "code": repo_root / "outputs/traces/m8_olmoe_code",
            "math": repo_root / "outputs/traces/m8_olmoe_math",
        },
    }


def _make_sequence_ids(wide, seq_len: int) -> np.ndarray:
    return (wide.index.to_numpy() // seq_len).astype(np.int32)


def analyze_one(trace_dir: Path) -> dict | None:
    if not (trace_dir / "run_meta.json").exists():
        return None
    meta = json.loads((trace_dir / "run_meta.json").read_text())
    df = load_trace(trace_dir)
    wide = top1_pivot(df)
    seq_ids = _make_sequence_ids(wide, meta.get("seq_len", 1024))
    rows = all_layer_pairs_mi(
        wide,
        num_experts=meta["num_routed_experts"],
        sequence_ids=seq_ids,
        distances=(1,),
        null_repeats=2,
    )
    rows_d1 = [r for r in rows if r["d"] == 1]

    n_e = meta["num_routed_experts"]
    n_layers = meta["num_moe_layers"]
    all_red = []
    for L in range(n_layers - 1):
        if L not in wide.columns or (L + 1) not in wide.columns:
            continue
        x = wide[L].to_numpy()
        y = wide[L + 1].to_numpy()
        C = joint_counts(x, y, n_e, n_e)
        rep = per_source_entropy_reduction(C)
        red = rep["reduction"]
        all_red.append(red[~np.isnan(red)])
    flat_red = np.concatenate(all_red) if all_red else np.array([])

    return {
        "n_tokens": int(rows[0]["n_tokens"]) if rows else 0,
        "n_layers": n_layers,
        "num_routed_experts": n_e,
        "top_k": meta["top_k"],
        "mi_mean_nats": float(np.mean([r["MI_nats"] for r in rows_d1])) if rows_d1 else 0.0,
        "null_iid_mean_nats": float(np.mean([r["null_iid_MI_nats"] for r in rows_d1])) if rows_d1 else 0.0,
        "null_within_seq_mean_nats": float(np.mean([r["null_within_seq_MI_nats"] for r in rows_d1])) if rows_d1 else 0.0,
        "mi_minus_strict_null_mean_nats": float(np.mean([r["MI_minus_null_within_seq"] for r in rows_d1])) if rows_d1 else 0.0,
        "mi_norm_mean": float(np.mean([r["MI_norm_by_H_Lpd"] for r in rows_d1])) if rows_d1 else 0.0,
        "median_source_reduction_nats": float(np.median(flat_red)) if flat_red.size else 0.0,
        "frac_sources_reduce_ge_0p5_nat": float((flat_red >= 0.5).mean()) if flat_red.size else 0.0,
        "_per_pair_rows": rows_d1,
        "_flat_reductions": flat_red.tolist(),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--fig_dir", type=Path, default=REPO_ROOT / "outputs/figures")
    args = p.parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    grid = _domain_dirs(REPO_ROOT)
    results = {}  # {(label, tag): {domain: summary_dict}}
    for (label, tag), domain_dirs in grid.items():
        per_domain = {}
        for domain, trace_dir in domain_dirs.items():
            s = analyze_one(trace_dir)
            if s is None:
                logger.warning(f"[{label}/{domain}] no trace at {trace_dir}, skipping")
                continue
            per_domain[domain] = s
        if per_domain:
            results[(label, tag)] = per_domain

    if not results:
        logger.error("No traces found at all. Aborting.")
        return 1

    # ---------- F14: bar plot of MI−strict_null per (model, domain) ----------
    domains_seen = sorted({d for v in results.values() for d in v})
    models = list(results.keys())
    fig, ax = plt.subplots(figsize=(max(6, len(models) * 2.5), 5))
    width = 0.25
    x = np.arange(len(models))
    colors = {"nl": "#5A9BD3", "code": "#7AB07A", "math": "#D35F5F"}
    for i, dom in enumerate(domains_seen):
        ys = [results[m].get(dom, {}).get("mi_minus_strict_null_mean_nats", np.nan) for m in models]
        ax.bar(x + (i - (len(domains_seen) - 1) / 2) * width, ys, width=width,
               label=dom, color=colors.get(dom, None))
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in models], rotation=15)
    ax.set_ylabel("MI(L, L+1) − within-seq null  [nats, mean over layers]")
    ax.set_title("F14 — Cross-layer MI by domain × model  (top-1 expert per layer)")
    ax.axhline(0.3, color="orange", linestyle="--", linewidth=0.8, label="strong threshold (0.3 nat)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(args.fig_dir / "F14_multidomain_mi.png", dpi=150)
    fig.savefig(args.fig_dir / "F14_multidomain_mi.pdf")
    plt.close(fig)

    # ---------- F15: per-source histogram per model, overlaid by domain ----------
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, m in zip(axes, models):
        for dom in domains_seen:
            if dom not in results[m]:
                continue
            red = np.asarray(results[m][dom]["_flat_reductions"])
            if red.size == 0:
                continue
            ax.hist(
                red, bins=40, alpha=0.45, label=f"{dom} (median {np.median(red):.2f})",
                color=colors.get(dom, None), edgecolor="white",
            )
        ax.axvline(0.5, color="crimson", linestyle="--", linewidth=1)
        ax.set_xlabel("H(P(e_{L+1})) − H(P(e_{L+1}|e_L=i))  [nats]")
        ax.set_ylabel("count of (L, source) pairs")
        ax.set_title(m[0])
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("F15 — Per-source entropy reduction histogram, by domain")
    fig.tight_layout()
    fig.savefig(args.fig_dir / "F15_per_source_by_domain.png", dpi=150)
    fig.savefig(args.fig_dir / "F15_per_source_by_domain.pdf")
    plt.close(fig)

    # ---------- F16: per-layer MI line plot, color by domain ----------
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, m in zip(axes, models):
        for dom in domains_seen:
            if dom not in results[m]:
                continue
            rows = results[m][dom]["_per_pair_rows"]
            xs = [r["L"] for r in rows]
            ys = [r["MI_minus_null_within_seq"] for r in rows]
            ax.plot(xs, ys, marker="o", linewidth=1.5, label=dom, color=colors.get(dom, None))
        ax.axhline(0.3, color="orange", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Source layer L")
        ax.set_ylabel("MI(L, L+1) − strict null  [nats]")
        ax.set_title(m[0])
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("F16 — Per-layer cross-layer MI by domain")
    fig.tight_layout()
    fig.savefig(args.fig_dir / "F16_per_layer_by_domain.png", dpi=150)
    fig.savefig(args.fig_dir / "F16_per_layer_by_domain.pdf")
    plt.close(fig)

    # ---------- save JSON (strip large arrays) ----------
    json_out = {}
    for (label, tag), per_domain in results.items():
        json_out[label] = {}
        for dom, s in per_domain.items():
            s2 = {k: v for k, v in s.items() if not k.startswith("_")}
            json_out[label][dom] = s2
    (args.fig_dir / "multidomain_mi.json").write_text(json.dumps(json_out, indent=2))
    logger.info(f"Wrote {args.fig_dir / 'multidomain_mi.json'}")

    # ---------- console summary ----------
    print("\n=== Multi-domain cross-layer MI (MI − within-seq null, mean over layer pairs) ===\n")
    header = f"{'Model':28s}  " + "  ".join(f"{d:>10s}" for d in domains_seen) + f"   {'range':>7s}"
    print(header)
    print("-" * len(header))
    for (label, tag), per_domain in results.items():
        vals = [per_domain.get(d, {}).get("mi_minus_strict_null_mean_nats", float("nan")) for d in domains_seen]
        rng = (np.nanmax(vals) - np.nanmin(vals)) if any(not np.isnan(v) for v in vals) else float("nan")
        cells = "  ".join(f"{v:>10.3f}" if not np.isnan(v) else f"{'—':>10s}" for v in vals)
        print(f"{label:28s}  {cells}   {rng:>7.3f}")
    print()
    print("=== Per-source frac ≥ 0.5 nat by (model, domain) ===\n")
    print(header)
    print("-" * len(header))
    for (label, tag), per_domain in results.items():
        vals = [per_domain.get(d, {}).get("frac_sources_reduce_ge_0p5_nat", float("nan")) for d in domains_seen]
        rng = (np.nanmax(vals) - np.nanmin(vals)) if any(not np.isnan(v) for v in vals) else float("nan")
        cells = "  ".join(f"{v*100:>9.1f}%" if not np.isnan(v) else f"{'—':>10s}" for v in vals)
        print(f"{label:28s}  {cells}   {rng*100:>6.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
