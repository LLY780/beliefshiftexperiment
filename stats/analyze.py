"""
Belief Shift Experiment - Statistical Analysis
==============================================
Auto-detects experimental variables from CSV column names.
Works with any version of shift.py output.

Usage:
  python analyze.py [results.csv] [stats.csv]
  python analyze.py  # auto-detects *_results*.csv and *_stats*.csv

Dependencies: pip install pandas scipy matplotlib seaborn
"""

import pandas as pd
import numpy as np
from scipy import stats as sp
from itertools import combinations
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, glob, warnings
warnings.filterwarnings('ignore')

FIG_DIR = "figures"
ALPHA = 0.05

# Known non-variable columns
RESULTS_FIXED = {"claim", "init", "shift", "generated_text", "belief_shift", "abs_shift", "claim_type"}
STATS_FIXED = {"claim", "mean_init", "abs_mean_init", "mean_shift", "abs_mean_shift", "claim_type"}

# Long labels that need abbreviation
LABEL_SHORT = {
    "commitment and consistency": "Commitment",
    "social proof": "Social Proof",
}
SHIFT_LABELS = {}  # No longer used; scale is now 0-100 continuous


def shorten(label):
    """Shorten long labels for axis display."""
    s = str(label)
    return LABEL_SHORT.get(s, s.replace("_", " ").title() if "_" in s else s.title() if s.islower() else s)


# ============================================================
# DATA LOADING
# ============================================================

def load_data(results_path, stats_path):
    r = pd.read_csv(results_path)
    s = pd.read_csv(stats_path)

    # Rename 'goal' to 'position' for clearer academic terminology
    if "goal" in r.columns:
        r = r.rename(columns={"goal": "position"})
    if "goal" in s.columns:
        s = s.rename(columns={"goal": "position"})

    # "init" = initial belief (0-100), "shift" = post-exposure belief (0-100)
    # Compute actual belief shift and absolute shift
    r["belief_shift"] = r["shift"] - r["init"]
    r["abs_shift"] = r["belief_shift"].abs()

    # Compute mean belief shift for stats (mean_shift is post-exposure, mean_init is initial)
    if "mean_shift" in s.columns and "mean_init" in s.columns:
        s["mean"] = s["mean_shift"] - s["mean_init"]
    elif "mean" not in s.columns:
        s["mean"] = 0

    # Auto-detect experimental variables
    variables = [c for c in r.columns if c not in RESULTS_FIXED]
    print(f"Detected variables: {variables}")

    # Map claim types if claims.csv available
    for path in ["claims.csv", "beliefshiftexperiment/claims.csv"]:
        if os.path.exists(path):
            cdf = pd.read_csv(path)
            tmap = dict(zip(cdf["claim"], cdf["type"]))
            r["claim_type"] = r["claim"].map(tmap).fillna("unknown")
            s["claim_type"] = s["claim"].map(tmap).fillna("unknown")
            return r, s, variables
    r["claim_type"] = "unknown"
    s["claim_type"] = "unknown"
    return r, s, variables


# ============================================================
# STATISTICAL TESTS
# ============================================================

def kruskal_wallis_test(df, groupby, value_col):
    groups = [g[value_col].dropna().values for _, g in df.groupby(groupby)]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) < 2:
        return None
    H, p = sp.kruskal(*groups)
    n = sum(len(g) for g in groups)
    eps_sq = H / (n - 1) if n > 1 else 0
    return {"H": H, "p": p, "eps_sq": eps_sq,
            "effect": "large" if eps_sq > 0.14 else "medium" if eps_sq > 0.06 else "small" if eps_sq > 0.01 else "negligible",
            "sig": p < ALPHA}

def anova_test(df, groupby, value_col):
    groups = [g[value_col].dropna().values for _, g in df.groupby(groupby)]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) < 2:
        return None
    F, p = sp.f_oneway(*groups)
    grand_mean = df[value_col].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
    ss_total = ((df[value_col] - grand_mean)**2).sum()
    eta_sq = ss_between / ss_total if ss_total > 0 else 0
    return {"F": F, "p": p, "eta_sq": eta_sq,
            "effect": "large" if eta_sq > 0.14 else "medium" if eta_sq > 0.06 else "small" if eta_sq > 0.01 else "negligible",
            "sig": p < ALPHA}

def mann_whitney_pairwise(df, groupby, value_col):
    grouped = {name: g[value_col].dropna().values for name, g in df.groupby(groupby)}
    pairs = list(combinations(sorted(grouped.keys()), 2))
    n_comp = len(pairs)
    results = []
    for a, b in pairs:
        if len(grouped[a]) < 2 or len(grouped[b]) < 2:
            continue
        U, p = sp.mannwhitneyu(grouped[a], grouped[b], alternative='two-sided')
        n1, n2 = len(grouped[a]), len(grouped[b])
        r_rb = 1 - (2 * U) / (n1 * n2)
        p_adj = min(p * n_comp, 1.0)
        results.append({
            "a": a, "b": b,
            "mean_a": np.mean(grouped[a]), "mean_b": np.mean(grouped[b]),
            "diff": np.mean(grouped[a]) - np.mean(grouped[b]),
            "U": U, "p": p, "p_adj": p_adj,
            "rank_biserial_r": r_rb,
            "sig": p_adj < ALPHA
        })
    return pd.DataFrame(results)


# ============================================================
# ANALYSIS
# ============================================================

def descriptive_stats(results_df, variables):
    print("\n" + "="*60)
    print("DESCRIPTIVE STATISTICS")
    print("="*60)
    for var in variables:
        print(f"\n--- {var} ---")
        desc = results_df.groupby(var).agg(
            n=("belief_shift", "count"),
            mean_init=("init", "mean"),
            mean_post=("shift", "mean"),
            mean_shift=("belief_shift", "mean"),
            median_shift=("belief_shift", "median"),
            std_shift=("belief_shift", "std"),
            mean_abs_shift=("abs_shift", "mean"),
        ).round(3)
        print(desc.to_string())


def main_effects(results_df, variables):
    print("\n" + "="*60)
    print("MAIN EFFECTS")
    print("="*60)

    report = []
    sig_vars = {}

    for measure, label in [("belief_shift", "Raw Shift"), ("abs_shift", "Absolute Shift")]:
        print(f"\n--- {label} ---")
        for var in variables:
            kw = kruskal_wallis_test(results_df, var, measure)
            av = anova_test(results_df, var, measure)
            if not kw:
                continue

            sig_str = "***" if kw["p"] < 0.001 else "**" if kw["p"] < 0.01 else "*" if kw["sig"] else "ns"
            print(f"\n  {var}:")
            print(f"    Kruskal-Wallis: H={kw['H']:.2f}, p={kw['p']:.2e}, ε²={kw['eps_sq']:.4f} ({kw['effect']}) {sig_str}")
            if av:
                print(f"    ANOVA (suppl.): F={av['F']:.2f}, p={av['p']:.2e}, η²={av['eta_sq']:.4f} ({av['effect']})")

            if measure == "belief_shift":
                sig_vars[var] = {"sig": kw["sig"], "p": kw["p"], "eps_sq": kw["eps_sq"], "effect": kw["effect"]}

            report.append({
                "measure": measure, "variable": var,
                "test": "Kruskal-Wallis", "statistic": round(kw["H"], 2),
                "p": kw["p"], "effect_size": round(kw["eps_sq"], 4),
                "effect_label": kw["effect"], "significant": kw["sig"]
            })
            if av:
                report.append({
                    "measure": measure, "variable": var,
                    "test": "ANOVA", "statistic": round(av["F"], 2),
                    "p": av["p"], "effect_size": round(av["eta_sq"], 4),
                    "effect_label": av["effect"], "significant": av["sig"]
                })

    return pd.DataFrame(report), sig_vars


def pairwise_comparisons(results_df, variables):
    print("\n" + "="*60)
    print("PAIRWISE COMPARISONS (Mann-Whitney U, Bonferroni corrected)")
    print("="*60)

    all_pairs = []
    for measure, label in [("belief_shift", "Raw"), ("abs_shift", "Absolute")]:
        for var in variables:
            pairs = mann_whitney_pairwise(results_df, var, measure)
            if len(pairs) == 0:
                continue
            pairs["variable"] = var
            pairs["measure"] = measure
            all_pairs.append(pairs)

            sig_only = pairs[pairs["sig"]]
            if len(sig_only) > 0:
                print(f"\n  {var} ({label}): {len(sig_only)}/{len(pairs)} significant")
                for _, row in pairs.iterrows():
                    s = "***" if row["p_adj"] < 0.001 else "**" if row["p_adj"] < 0.01 else "*" if row["sig"] else "ns"
                    print(f"    {row['a']} vs {row['b']:25s} diff={row['diff']:+.3f} r={row['rank_biserial_r']:+.3f} p_adj={row['p_adj']:.2e} {s}")

    return pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame()


def interaction_effects(stats_df, variables):
    print("\n" + "="*60)
    print("INTERACTIONS")
    print("="*60)
    for v1, v2 in combinations(variables, 2):
        print(f"\n  {v1} × {v2}:")
        pivot = stats_df.pivot_table(values="mean", index=v1, columns=v2, aggfunc="mean")
        print("  " + pivot.round(3).to_string().replace("\n", "\n  "))


def claim_type_comparison(results_df):
    types = [t for t in results_df["claim_type"].unique() if t != "unknown"]
    if len(types) < 2:
        print("\n\n  [Only one claim type available - fact/opinion comparison skipped]")
        return
    print("\n" + "="*60)
    print("FACT vs OPINION")
    print("="*60)
    for ct in types:
        sub = results_df[results_df["claim_type"] == ct]
        print(f"  {ct}: mean={sub['belief_shift'].mean():+.3f} abs_mean={sub['abs_shift'].mean():.3f} n={len(sub)}")
    facts = results_df[results_df["claim_type"] == "fact"]["belief_shift"].dropna()
    opinions = results_df[results_df["claim_type"] == "opinion"]["belief_shift"].dropna()
    if len(facts) > 0 and len(opinions) > 0:
        U, p = sp.mannwhitneyu(facts, opinions, alternative='two-sided')
        print(f"\n  Mann-Whitney U={U:.0f}, p={p:.2e}")


def ranking_summary(results_df, variables):
    print("\n" + "="*60)
    print("RANKING: WHICH METRICS WORK BEST")
    print("="*60)
    print("\n  Variables ranked by effect size (Kruskal-Wallis ε² on raw shift):")
    var_effects = []
    for var in variables:
        kw = kruskal_wallis_test(results_df, var, "belief_shift")
        if kw:
            var_effects.append((var, kw["eps_sq"], kw["p"], kw["effect"], kw["sig"]))
    var_effects.sort(key=lambda x: x[1], reverse=True)
    for rank, (var, eps, p, eff, sig) in enumerate(var_effects, 1):
        print(f"    {rank}. {var:35s} ε²={eps:.4f} ({eff:6s}) p={p:.2e} {'✓' if sig else '✗'}")

    print("\n  Most effective level per significant variable (by absolute shift):")
    for var, eps, p, eff, sig in var_effects:
        if not sig:
            continue
        means = results_df.groupby(var)["abs_shift"].mean().sort_values(ascending=False)
        print(f"    {var}: {means.index[0]} (abs_mean={means.iloc[0]:.3f})")


# ============================================================
# VISUALIZATIONS
# ============================================================

def plot_main_effects(results_df, variables, sig_vars):
    """Violin plots, auto-adapts grid to number of variables."""
    n = len(variables)
    cols = min(n, 2)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5.5 * rows))
    fig.suptitle("Belief Shift Distribution by Experimental Variable", fontsize=16, fontweight="bold")

    if n == 1:
        axes = [axes]
    else:
        axes = axes.flat

    for idx, (ax, var) in enumerate(zip(axes, variables)):
        order = results_df.groupby(var)["belief_shift"].mean().sort_values(ascending=False).index.tolist()
        n_levels = len(order)

        # Use shortened labels
        short_order = [shorten(x) for x in order]
        plot_df = results_df.copy()
        plot_df["_plot_var"] = plot_df[var].map(lambda x: shorten(x))

        sns.violinplot(data=plot_df, x="_plot_var", y="belief_shift", order=short_order,
                       palette="Set2", inner="quartile", ax=ax, cut=0)

        # Mean markers + value labels (skip if too many levels)
        means = results_df.groupby(var)["belief_shift"].mean()
        for i, level in enumerate(order):
            ax.scatter(i, means[level], color="black", s=30, zorder=5)
            if n_levels <= 4:
                ax.annotate(f"{means[level]:.1f}", (i, means[level]),
                           textcoords="offset points", xytext=(15, 0),
                           fontsize=9, fontweight="bold")

        ax.set_ylabel("Belief shift (post − initial)")
        ax.set_xlabel("")
        ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        if n_levels > 4:
            ax.tick_params(axis="x", rotation=25, labelsize=8)
        else:
            ax.tick_params(axis="x", rotation=0, labelsize=9)

        info = sig_vars.get(var, {})
        if info.get("sig") and info.get("effect") != "negligible":
            stars = "***" if info["p"] < 0.001 else "**" if info["p"] < 0.01 else "*"
            ax.set_title(f"{shorten(var)}  {stars} ({info['effect']} effect)", fontweight="bold", fontsize=12)
        elif info.get("sig") and info.get("effect") == "negligible":
            ax.set_title(f"{shorten(var)}  (negligible effect)", fontweight="bold", fontsize=12, color="gray")
        else:
            ax.set_title(f"{shorten(var)}  (not significant)", fontweight="bold", fontsize=12, color="gray")

    # Hide unused axes
    for idx in range(n, rows * cols):
        axes[idx].set_visible(False)

    fig.text(0.5, 0.01,
             "● = mean  |  *** p < 0.001, * p < 0.05  |  Shape width = data density  |  Dashed line = no change (0)",
             ha="center", fontsize=9, style="italic", color="gray")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(f"{FIG_DIR}/main_effects.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_interaction(stats_df, variables):
    """All pairwise interaction heatmaps with unified color scale."""
    pairs = list(combinations(variables, 2))
    n_pairs = len(pairs)
    if n_pairs == 0:
        return

    # Compute global min/max for unified color scale (belief shift can be negative)
    all_means = []
    for v1, v2 in pairs:
        pivot = stats_df.pivot_table(values="mean", index=v1, columns=v2, aggfunc="mean")
        all_means.extend(pivot.values.flatten())
    abs_max = max(abs(min(all_means)), abs(max(all_means)))
    vmin = -abs_max
    vmax = abs_max

    cols = min(n_pairs, 3)
    rows = math.ceil(n_pairs / cols)
    fig, axes_arr = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows))
    fig.suptitle("Interaction Effects: Mean Belief Shift Across Variable Pairs",
                 fontsize=16, fontweight="bold")

    if n_pairs == 1:
        axes_list = [axes_arr]
    else:
        axes_list = axes_arr.flat

    for idx, (ax, (v1, v2)) in enumerate(zip(axes_list, pairs)):
        pivot = stats_df.pivot_table(values="mean", index=v1, columns=v2, aggfunc="mean")
        pivot.index = [shorten(x) for x in pivot.index]
        pivot.columns = [shorten(x) for x in pivot.columns]

        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn",
                    vmin=vmin, vmax=vmax,
                    ax=ax, linewidths=0.5, annot_kws={"size": 10},
                    cbar_kws={"shrink": 0.8, "label": ""})
        ax.set_title(f"{shorten(v1)} × {shorten(v2)}", fontweight="bold", fontsize=11)
        ax.set_xlabel(shorten(v2), fontsize=10)
        ax.set_ylabel(shorten(v1), fontsize=10)
        ax.tick_params(labelsize=8)

    # Hide unused axes
    for idx in range(n_pairs, rows * cols):
        axes_list[idx].set_visible(False)

    fig.text(0.5, 0.01,
             f"Color scale: {vmin:.1f} (low shift) → {vmax:.1f} (high shift)  |  Same scale across all panels",
             ha="center", fontsize=9, style="italic", color="gray")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(f"{FIG_DIR}/interactions.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_mean_shift(results_df, variables, sig_vars):
    """Bar charts for variables with meaningful effects (not negligible, ≤5 levels)."""
    # Filter: significant, not negligible, and few enough levels for clean bars
    show_list = []
    for v in variables:
        info = sig_vars.get(v, {})
        n_levels = results_df[v].nunique()
        if info.get("sig") and info.get("effect") != "negligible" and n_levels <= 5:
            show_list.append(v)

    if not show_list:
        print("  No variables suitable for bar chart")
        return

    # Compute unified Y-axis max
    global_max = 0
    for var in show_list:
        m = results_df.groupby(var)["belief_shift"].mean().abs().max()
        if m > global_max:
            global_max = m
    y_max = global_max * 1.2

    n = len(show_list)
    fig, axes = plt.subplots(1, n, figsize=(max(4, 4 * n), 5))
    fig.suptitle("Mean Belief Shift: Variables with Meaningful Effects", fontsize=14, fontweight="bold")

    if n == 1:
        axes = [axes]

    for ax, var in zip(axes, show_list):
        means = results_df.groupby(var)["belief_shift"].mean()
        order = means.sort_values(ascending=False).index.tolist()
        vals = [means[k] for k in order]
        n_bars = len(order)
        colors = ["#2ecc71", "#f39c12", "#e74c3c"][:n_bars] if n_bars <= 3 else sns.color_palette("RdYlGn", n_bars)[::-1]
        short_labels = [shorten(x) for x in order]
        bars = ax.bar(short_labels, vals, color=colors, edgecolor="black", linewidth=0.5, width=0.6)
        for bar, v in zip(bars, vals):
            # Position label above positive bars, below negative bars
            if v >= 0:
                ax.text(bar.get_x() + bar.get_width()/2, v + global_max * 0.03,
                        f"{v:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=13)
            else:
                ax.text(bar.get_x() + bar.get_width()/2, v - global_max * 0.03,
                        f"{v:.2f}", ha="center", va="top", fontweight="bold", fontsize=13)
        ax.set_ylabel("")
        info = sig_vars.get(var, {})
        ax.set_title(f"{shorten(var)} ({info.get('effect', '')} effect)", fontweight="bold", fontsize=13)
        ax.set_ylim(-y_max, y_max)
        ax.axhline(y=0, color="black", linewidth=0.5)

    fig.text(0.5, 0.01,
             "Scale: belief shift = post-exposure score − initial score  |  0 = no change  |  Positive = shifted toward agree",
             ha="center", fontsize=9, style="italic", color="gray")
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.savefig(f"{FIG_DIR}/mean_shift.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    if len(sys.argv) >= 3:
        rp, sp_path = sys.argv[1], sys.argv[2]
    else:
        rf = sorted(glob.glob("*_results*.csv"))
        sf = sorted(glob.glob("*_stats*.csv"))
        if not rf or not sf:
            print("No CSV files found. Usage: python analyze.py results.csv stats.csv")
            return
        rp, sp_path = rf[0], sf[0]

    print(f"Results: {rp}")
    print(f"Stats:   {sp_path}")

    results_df, stats_df, variables = load_data(rp, sp_path)

    reps = results_df.groupby(["claim"] + variables).size()
    print(f"\n--- Data Summary ---")
    print(f"Results: {len(results_df)} rows, {results_df['claim'].nunique()} claims")
    print(f"Variables: {variables}")
    for v in variables:
        print(f"  {v}: {sorted(results_df[v].unique())}")
    print(f"Conditions: {len(stats_df)} combinations, {reps.min()}-{reps.max()} reps each")
    print(f"Claim types: {results_df['claim_type'].value_counts().to_dict()}")
    print(f"Init range: {results_df['init'].min()} to {results_df['init'].max()}")
    print(f"Post range: {results_df['shift'].min()} to {results_df['shift'].max()}")
    print(f"Belief shift range: {results_df['belief_shift'].min()} to {results_df['belief_shift'].max()}")

    os.makedirs(FIG_DIR, exist_ok=True)

    # Analysis
    descriptive_stats(results_df, variables)
    effects_report, sig_vars = main_effects(results_df, variables)
    pairs_report = pairwise_comparisons(results_df, variables)
    interaction_effects(stats_df, variables)
    claim_type_comparison(results_df)
    ranking_summary(results_df, variables)

    # Figures
    print("\nGenerating figures...")
    plot_main_effects(results_df, variables, sig_vars)
    print(f"  {FIG_DIR}/main_effects.png")
    plot_mean_shift(results_df, variables, sig_vars)
    print(f"  {FIG_DIR}/mean_shift.png")
    plot_interaction(stats_df, variables)
    print(f"  {FIG_DIR}/interactions.png")

    print(f"\nDone. All figures in {FIG_DIR}/")


if __name__ == "__main__":
    main()
