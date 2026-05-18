"""G8 disambiguation — Router entropy analysis + filtered-MI test.

Tests whether code's lower cross-layer MI is an OOD artifact (G8a) or a
structural difference (G8b):

1. Compare per-token router entropy distributions across (model, domain).
   - G8a predicts: code has higher entropy than NL (less confident routing).
   - G8b predicts: entropy distributions are similar across domains.

2. Filter to "confident" tokens (entropy < threshold = model's NL median),
   then recompute cross-layer MI. Compare:
   - If filtered MI gap (code vs NL) shrinks substantially → OOD explains.
   - If gap persists at the same magnitude → structural.

Renders F17 (entropy histogram), F18 (effective experts per layer), F19
(filtered MI bar). Writes router_entropy_summary.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.router_entropy import (
    EntropyData,
    filtered_mi_summary,
    load_entropy,
    per_layer_summary,
)
from src.utils import get_logger

logger = get_logger("router_entropy")
REPO_ROOT = Path(__file__).resolve().parent.parent

MODELS = [
    ("Qwen1.5-MoE-A2.7B", "qwen"),
    ("DeepSeek-V2-Lite", "deepseek"),
    ("OLMoE-1B-7B", "olmoe"),
]
DOMAINS = ["nl", "code", "math"]


def _trace_dir_for(tag: str, domain: str, repo_root: Path) -> Path:
    if domain == "nl":
        nl_map = {
            "qwen": "m4_qwen_wikitext",
            "deepseek": "m1_deepseek_wikitext",
            "olmoe": "m7_olmoe_wikitext",
        }
        return repo_root / "outputs/traces" / nl_map[tag]
    return repo_root / "outputs/traces" / f"m8_{tag}_{domain}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ent_root", type=Path,
                   default=REPO_ROOT / "outputs" / "router_entropy")
    p.add_argument("--fig_dir", type=Path,
                   default=REPO_ROOT / "outputs" / "figures")
    args = p.parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------- load all entropies
    data: Dict[tuple, EntropyData] = {}
    for label, tag in MODELS:
        for dom in DOMAINS:
            path = args.ent_root / f"{tag}_{dom}.npz"
            if not path.exists():
                logger.warning(f"missing {path.name}, skipping ({label}, {dom})")
                continue
            data[(label, tag, dom)] = load_entropy(path)
    if not data:
        logger.error("No entropy npz files found.")
        return 1

    # ---------------------------------------------------------- summary table
    summary_rows = []
    for (label, tag, dom), ed in data.items():
        n_e = ed.meta["num_routed_experts"]
        H_uniform = float(np.log(n_e))
        e = ed.entropies.flatten()
        summary_rows.append({
            "model": label,
            "domain": dom,
            "num_experts": n_e,
            "n_tokens": int(ed.entropies.shape[0]),
            "n_layers": int(ed.entropies.shape[1]),
            "mean_H_nats": float(e.mean()),
            "median_H_nats": float(np.median(e)),
            "p25_H_nats": float(np.percentile(e, 25)),
            "p75_H_nats": float(np.percentile(e, 75)),
            "fraction_of_uniform": float(e.mean() / H_uniform),
            "median_eff_experts": float(np.exp(np.median(e))),
            "mean_eff_experts": float(np.exp(e.mean())),
        })

    # ---------------------------------------------------------- F17: histograms
    n_models = len(MODELS)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 4.6), sharey=True)
    if n_models == 1:
        axes = [axes]
    colors = {"nl": "#5A9BD3", "code": "#7AB07A", "math": "#D35F5F"}
    for ax, (label, tag) in zip(axes, MODELS):
        present = [d for d in DOMAINS if (label, tag, d) in data]
        if not present:
            continue
        H_uniform = float(np.log(data[(label, tag, present[0])].meta["num_routed_experts"]))
        for dom in present:
            ed = data[(label, tag, dom)]
            e = ed.entropies.flatten()
            ax.hist(e, bins=80, alpha=0.45, color=colors.get(dom),
                    label=f"{dom}  median {np.median(e):.2f} nat",
                    edgecolor="white", density=True)
        ax.axvline(H_uniform, color="black", linestyle=":",
                   linewidth=1, label=f"H(uniform)={H_uniform:.2f}")
        ax.set_xlabel("router entropy H(P(e | token))   [nats]")
        ax.set_ylabel("density")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("F17 — Per-token router entropy distribution by domain  (G8 disambiguation)")
    fig.tight_layout()
    fig.savefig(args.fig_dir / "F17_router_entropy_hist.png", dpi=150)
    fig.savefig(args.fig_dir / "F17_router_entropy_hist.pdf")
    plt.close(fig)

    # ---------------------------------------------------------- F18: per-layer median entropy
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 4.5), sharey=True)
    if n_models == 1:
        axes = [axes]
    for ax, (label, tag) in zip(axes, MODELS):
        present = [d for d in DOMAINS if (label, tag, d) in data]
        if not present:
            continue
        for dom in present:
            ed = data[(label, tag, dom)]
            xs = np.arange(ed.entropies.shape[1])
            ys = np.median(ed.entropies, axis=0)
            ax.plot(xs, ys, marker="o", linewidth=1.5, color=colors.get(dom), label=dom)
        H_uniform = float(np.log(data[(label, tag, present[0])].meta["num_routed_experts"]))
        ax.axhline(H_uniform, color="black", linestyle=":", linewidth=1, label=f"H(uniform)={H_uniform:.2f}")
        ax.set_xlabel("MoE layer")
        ax.set_ylabel("median router entropy  [nats]")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("F18 — Per-layer median router entropy by domain")
    fig.tight_layout()
    fig.savefig(args.fig_dir / "F18_per_layer_entropy.png", dpi=150)
    fig.savefig(args.fig_dir / "F18_per_layer_entropy.pdf")
    plt.close(fig)

    # ---------------------------------------------------------- Filtered MI test
    # For each model: pick threshold = NL's MEDIAN entropy.
    # Recompute cross-layer MI for each (model, domain) on the confident-only subset.
    filtered_results: Dict[tuple, dict] = {}
    for label, tag in MODELS:
        nl_key = (label, tag, "nl")
        if nl_key not in data:
            continue
        nl_ent = data[nl_key].entropies.flatten()
        threshold = float(np.median(nl_ent))
        for dom in DOMAINS:
            key = (label, tag, dom)
            if key not in data:
                continue
            npz_path = args.ent_root / f"{tag}_{dom}.npz"
            trace_dir = _trace_dir_for(tag, dom, REPO_ROOT)
            if not (trace_dir / "run_meta.json").exists():
                logger.warning(f"missing trace dir {trace_dir}, skipping filtered MI")
                continue
            try:
                fmi = filtered_mi_summary(npz_path, trace_dir, threshold=threshold)
                fmi["threshold_basis"] = f"NL median = {threshold:.3f} nat"
                filtered_results[key] = fmi
                logger.info(
                    f"[{label}/{dom}] filtered MI−null = {fmi['mean_MI_filtered_minus_null']:.3f} nat "
                    f"(kept {fmi['mean_n_kept']:.0f} tokens / pair, threshold={threshold:.2f})"
                )
            except Exception as e:
                logger.warning(f"filtered MI failed for {label}/{dom}: {e}")

    # ---------------------------------------------------------- F19: filtered vs unfiltered MI
    if filtered_results:
        # We need the *unfiltered* MI from multidomain_mi.json for comparison
        un_path = args.fig_dir / "multidomain_mi.json"
        unfiltered: Dict[tuple, float] = {}
        if un_path.exists():
            mu = json.loads(un_path.read_text())
            for label, per_dom in mu.items():
                for dom, s in per_dom.items():
                    unfiltered[(label, dom)] = s.get("mi_minus_strict_null_mean_nats", float("nan"))

        fig, ax = plt.subplots(figsize=(max(6, len(MODELS) * 2.5), 5))
        width = 0.18
        x = np.arange(len(MODELS))
        offset_within_model = {"nl": -1.5, "code": -0.5, "math": 0.5}  # placeholder, actually 6 bars
        # Plot grouped bars: per model, 6 bars = (3 domains) × (filtered, unfiltered)
        for j, dom in enumerate(DOMAINS):
            uf = []
            fl = []
            for (label, tag) in MODELS:
                uf.append(unfiltered.get((label, dom), np.nan))
                fl.append(filtered_results.get((label, tag, dom), {}).get(
                    "mean_MI_filtered_minus_null", np.nan))
            uf = np.array(uf, dtype=float)
            fl = np.array(fl, dtype=float)
            base_offset = (j - 1) * (width * 2.2)  # space the 3 domain groups
            ax.bar(x + base_offset - width * 0.5, uf, width, color=colors[dom],
                   alpha=0.5, edgecolor="black",
                   label=f"{dom} unfiltered" if j == 0 else None)
            ax.bar(x + base_offset + width * 0.5, fl, width, color=colors[dom],
                   alpha=1.0, edgecolor="black", hatch="//",
                   label=f"{dom} filtered (entropy<NL_median)" if j == 0 else None)
        ax.set_xticks(x)
        ax.set_xticklabels([m[0] for m in MODELS], rotation=12)
        ax.set_ylabel("MI(L, L+1) − null  [nats]")
        ax.set_title("F19 — Cross-layer MI: unfiltered (light) vs entropy-filtered (hatched)")
        ax.axhline(0.3, color="orange", linestyle="--", linewidth=0.8)
        ax.grid(alpha=0.3, axis="y")
        # Build a custom legend
        from matplotlib.patches import Patch
        legend_handles = []
        for d in DOMAINS:
            legend_handles.append(Patch(facecolor=colors[d], alpha=0.5, edgecolor="black", label=f"{d} unfiltered"))
            legend_handles.append(Patch(facecolor=colors[d], alpha=1.0, edgecolor="black", hatch="//",
                                        label=f"{d} filtered"))
        ax.legend(handles=legend_handles, fontsize=8, ncol=3, loc="upper right")
        fig.tight_layout()
        fig.savefig(args.fig_dir / "F19_filtered_mi.png", dpi=150)
        fig.savefig(args.fig_dir / "F19_filtered_mi.pdf")
        plt.close(fig)

    # ---------------------------------------------------------- save JSON + print
    out_json = {
        "summary_rows": summary_rows,
        "filtered_mi": {
            f"{label}/{dom}": {k: v for k, v in fmi.items() if k != "per_pair"}
            for (label, tag, dom), fmi in filtered_results.items()
        },
    }
    (args.fig_dir / "router_entropy_summary.json").write_text(json.dumps(out_json, indent=2))
    logger.info(f"Wrote {args.fig_dir / 'router_entropy_summary.json'}")

    # ---------------------------------------------------------- console
    print("\n=== Router entropy: median per (model, domain), and effective experts ===\n")
    print(f"{'Model':28s}  {'Dom':>5s}  {'med H':>7s}  {'med eff':>9s}  {'frac unif':>10s}")
    print("-" * 70)
    for r in summary_rows:
        print(f"{r['model']:28s}  {r['domain']:>5s}  {r['median_H_nats']:>7.3f}  "
              f"{r['median_eff_experts']:>9.2f}  {r['fraction_of_uniform']*100:>9.1f}%")

    if filtered_results:
        print("\n=== Filtered MI: (entropy < NL_median) at both L and L+1 ===\n")
        print(f"{'Model':28s}  {'Dom':>5s}  {'unfiltered':>11s}  {'filtered':>10s}  {'kept/pair':>11s}")
        print("-" * 78)
        for label, tag in MODELS:
            for dom in DOMAINS:
                key = (label, tag, dom)
                fmi = filtered_results.get(key)
                if fmi is None:
                    continue
                unf = unfiltered.get((label, dom), float("nan"))
                fil = fmi["mean_MI_filtered_minus_null"]
                print(f"{label:28s}  {dom:>5s}  {unf:>11.3f}  {fil:>10.3f}  {fmi['mean_n_kept']:>11.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
