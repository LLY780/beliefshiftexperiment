"""
Belief Shift Experiment - Statistical Analysis
==============================================
Analyzes which experimental variables most effectively shift simulated beliefs.

Input:  *_results.csv and *_stats.csv from shift.py
Output:
  Console report
  main_effects.csv         - significance tests per variable
  pairwise_comparisons.csv - pairwise tests between levels
  figures/main_effects.png - violin plots with significance markers
  figures/distributions.png - shift distributions for significant variables
  figures/interaction.png  - lean x sentiment heatmap

Usage:
  python analyze.py [results.csv] [stats.csv]
  python analyze.py  # auto-detects files in current dir

Dependencies: pip install pandas scipy matplotlib seaborn

Methodology:
- Shift values are ordinal integers (-2 to +2), not continuous
- Data fails Shapiro-Wilk normality test
- Primary: Kruskal-Wallis H + Mann-Whitney U (non-parametric)
- Supplementary: ANOVA + t-test (parametric, for reference)
- Both raw shift (direction) and absolute shift (magnitude) analyzed
- Initial belief assumed neutral (2) for all - no pre-test measurement
"""

import pandas as pd
import numpy as np
from scipy import stats as sp
from itertools import combinations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, glob, warnings
warnings.filterwarnings('ignore')

FIG_DIR = "figures"
ALPHA = 0.05
VARIABLES = ["text_type", "technique", "sentiment", "lean"]


# ============================================================
# DATA LOADING
# ============================================================

def load_data(results_path, stats_path):
    r = pd.read_csv(results_path)
    s = pd.read_csv(stats_path)
    r["abs_shift"] = r["shift"].abs()
    # Map claim types if claims.csv available
    for path in ["claims.csv", "beliefshiftexperiment/claims.csv"]:
        if os.path.exists(path):
            cdf = pd.read_csv(path)
            tmap = dict(zip(cdf["claim"], cdf["type"]))
            r["claim_type"] = r["claim"].map(tmap).fillna("unknown")
            s["claim_type"] = s["claim"].map(tmap).fillna("unknown")
            return r, s
    r["claim_type"] = "unknown"
    s["claim_type"] = "unknown"
    return r, s


# ============================================================
# STATISTICAL TESTS
# ============================================================

def kruskal_wallis_test(df, groupby, value_col):
    """Kruskal-Wallis H test with epsilon-squared effect size."""
    groups = [g[value_col].dropna().values for _, g in df.groupby(groupby)]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) < 2:
        return None
    H, p = sp.kruskal(*groups)
    n = sum(len(g) for g in groups)
    eps_sq = H / (n - 1) if n > 1 else 0
    return {"H": H, "p": p, "eps_sq": eps_sq,
            "effect": "large" if eps_sq > 0.14 else "medium" if eps_sq > 0.06 else "small",
            "sig": p < ALPHA}

def anova_test(df, groupby, value_col):
    """One-way ANOVA with eta-squared (supplementary)."""
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
            "effect": "large" if eta_sq > 0.14 else "medium" if eta_sq > 0.06 else "small",
            "sig": p < ALPHA}

def mann_whitney_pairwise(df, groupby, value_col):
    """Pairwise Mann-Whitney U with Bonferroni correction and rank-biserial r."""
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
            "median_a": np.median(grouped[a]), "median_b": np.median(grouped[b]),
            "diff": np.mean(grouped[a]) - np.mean(grouped[b]),
            "U": U, "p": p, "p_adj": p_adj,
            "rank_biserial_r": r_rb,
            "sig": p_adj < ALPHA
        })
    return pd.DataFrame(results)


# ============================================================
# ANALYSIS
# ============================================================

def descriptive_stats(results_df):
    print("\n" + "="*60)
    print("DESCRIPTIVE STATISTICS")
    print("="*60)
    for var in VARIABLES:
        print(f"\n--- {var} ---")
        desc = results_df.groupby(var).agg(
            n=("shift", "count"),
            mean_shift=("shift", "mean"),
            median_shift=("shift", "median"),
            std_shift=("shift", "std"),
            mean_abs_shift=("abs_shift", "mean"),
            median_abs_shift=("abs_shift", "median"),
        ).round(3)
        print(desc.to_string())


def main_effects(results_df):
    """Test each variable's main effect. Returns report DataFrame and dict of significance."""
    print("\n" + "="*60)
    print("MAIN EFFECTS")
    print("="*60)

    report = []
    sig_vars = {}  # track which variables are significant for plotting

    for measure, label in [("shift", "Raw Shift"), ("abs_shift", "Absolute Shift")]:
        print(f"\n--- {label} ---")
        for var in VARIABLES:
            kw = kruskal_wallis_test(results_df, var, measure)
            av = anova_test(results_df, var, measure)
            if not kw:
                continue

            sig_str = "***" if kw["p"] < 0.001 else "**" if kw["p"] < 0.01 else "*" if kw["sig"] else "ns"
            print(f"\n  {var}:")
            print(f"    Kruskal-Wallis: H={kw['H']:.2f}, p={kw['p']:.2e}, ε²={kw['eps_sq']:.4f} ({kw['effect']}) {sig_str}")
            if av:
                print(f"    ANOVA (suppl.): F={av['F']:.2f}, p={av['p']:.2e}, η²={av['eta_sq']:.4f} ({av['effect']})")

            if measure == "shift":
                sig_vars[var] = {"sig": kw["sig"], "p": kw["p"], "eps_sq": kw["eps_sq"]}

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


def pairwise_comparisons(results_df):
    print("\n" + "="*60)
    print("PAIRWISE COMPARISONS (Mann-Whitney U, Bonferroni corrected)")
    print("="*60)

    all_pairs = []
    for measure, label in [("shift", "Raw"), ("abs_shift", "Absolute")]:
        for var in VARIABLES:
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


def interaction_effects(stats_df):
    print("\n" + "="*60)
    print("INTERACTION: Lean × Sentiment")
    print("="*60)

    print("\n  Mean shift:")
    pivot = stats_df.pivot_table(values="mean", index="lean", columns="sentiment", aggfunc="mean")
    print("  " + pivot.round(3).to_string().replace("\n", "\n  "))

    print("\n  Absolute mean shift:")
    pivot_abs = stats_df.pivot_table(values="abs_mean", index="lean", columns="sentiment", aggfunc="mean")
    print("  " + pivot_abs.round(3).to_string().replace("\n", "\n  "))


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
        print(f"  {ct}: mean={sub['shift'].mean():+.3f} abs_mean={sub['abs_shift'].mean():.3f} n={len(sub)}")

    facts = results_df[results_df["claim_type"] == "fact"]["shift"].dropna()
    opinions = results_df[results_df["claim_type"] == "opinion"]["shift"].dropna()
    if len(facts) > 0 and len(opinions) > 0:
        U, p = sp.mannwhitneyu(facts, opinions, alternative='two-sided')
        print(f"\n  Mann-Whitney U={U:.0f}, p={p:.2e}")


def ranking_summary(results_df):
    print("\n" + "="*60)
    print("RANKING: WHICH METRICS WORK BEST")
    print("="*60)

    print("\n  Variables ranked by effect size (Kruskal-Wallis ε² on raw shift):")
    var_effects = []
    for var in VARIABLES:
        kw = kruskal_wallis_test(results_df, var, "shift")
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

def plot_main_effects(results_df, sig_vars):
    """Violin plots with significance annotation per subplot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Belief Shift by Experimental Variable", fontsize=16, fontweight="bold")

    for ax, var in zip(axes.flat, VARIABLES):
        order = results_df.groupby(var)["shift"].mean().sort_values(ascending=False).index.tolist()
        sns.violinplot(data=results_df, x=var, y="shift", order=order,
                       palette="Set2", inner="quartile", ax=ax, cut=0)
        # Mean markers
        means = results_df.groupby(var)["shift"].mean()
        for i, level in enumerate(order):
            ax.scatter(i, means[level], color="black", s=30, zorder=5)

        ax.set_ylabel("Shift")
        ax.set_xlabel("")
        ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.tick_params(axis="x", rotation=20, labelsize=9)

        # Significance annotation
        info = sig_vars.get(var, {})
        if info.get("sig"):
            p = info["p"]
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*"
            ax.set_title(f"{var.replace('_', ' ').title()} {stars}\n(ε²={info['eps_sq']:.4f})",
                         fontweight="bold")
        else:
            ax.set_title(f"{var.replace('_', ' ').title()} (ns)", fontweight="bold", color="gray")

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/main_effects.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_interaction(stats_df):
    """Single Lean × Sentiment heatmap."""
    fig, ax = plt.subplots(figsize=(7, 5))
    pivot = stats_df.pivot_table(values="mean", index="lean", columns="sentiment", aggfunc="mean")
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                ax=ax, linewidths=0.5, annot_kws={"size": 14},
                cbar_kws={"label": "Mean Shift"})
    ax.set_title("Lean × Sentiment Interaction (Mean Shift)", fontweight="bold", fontsize=13)
    ax.set_xlabel("Sentiment", fontsize=12)
    ax.set_ylabel("Lean", fontsize=12)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/interaction.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_distributions(results_df):
    """Shift percentage distributions for lean and sentiment."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Shift Distribution by Significant Variables", fontsize=14, fontweight="bold")

    shift_labels = {-2: "Str.\nDisagree", -1: "Disagree", 0: "Neutral", 1: "Agree", 2: "Str.\nAgree"}
    all_shifts = list(range(-2, 3))

    for ax, var in zip(axes, ["lean", "sentiment"]):
        for level in sorted(results_df[var].unique()):
            sub = results_df[results_df[var] == level]
            counts = sub["shift"].value_counts().reindex(all_shifts, fill_value=0).sort_index()
            pct = counts / len(sub) * 100
            ax.plot(all_shifts, pct.values, marker="o", label=level, linewidth=2)
        ax.set_xlabel("Final Belief")
        ax.set_ylabel("Percentage (%)")
        ax.set_title(var.replace("_", " ").title(), fontweight="bold")
        ax.set_xticks(all_shifts)
        ax.set_xticklabels([shift_labels[s] for s in all_shifts], fontsize=8)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/distributions.png", dpi=150, bbox_inches="tight")
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

    results_df, stats_df = load_data(rp, sp_path)

    reps = results_df.groupby(["claim","text_type","technique","sentiment","lean"]).size()
    print(f"\n--- Data Summary ---")
    print(f"Results: {len(results_df)} rows, {results_df['claim'].nunique()} claims")
    print(f"Conditions: {len(stats_df)} combinations, {reps.min()}-{reps.max()} reps each")
    print(f"Claim types: {results_df['claim_type'].value_counts().to_dict()}")
    print(f"Shift range: {results_df['shift'].min()} to {results_df['shift'].max()}")
    print(f"Note: Ordinal data, non-normal. Primary tests are non-parametric.")

    os.makedirs(FIG_DIR, exist_ok=True)

    # Analysis
    descriptive_stats(results_df)
    effects_report, sig_vars = main_effects(results_df)
    pairs_report = pairwise_comparisons(results_df)
    interaction_effects(stats_df)
    claim_type_comparison(results_df)
    ranking_summary(results_df)

    # CSV output - designed for human reading
    if len(effects_report):
        # Pivot: one row per variable, columns for both tests and measures
        rows = []
        for var in VARIABLES:
            row = {"variable": var}
            for measure in ["shift", "abs_shift"]:
                prefix = "raw" if measure == "shift" else "abs"
                kw = effects_report[(effects_report["variable"]==var) & 
                     (effects_report["measure"]==measure) & 
                     (effects_report["test"]=="Kruskal-Wallis")]
                av = effects_report[(effects_report["variable"]==var) & 
                     (effects_report["measure"]==measure) & 
                     (effects_report["test"]=="ANOVA")]
                if len(kw):
                    k = kw.iloc[0]
                    row[f"{prefix}_H"] = k["statistic"]
                    row[f"{prefix}_p"] = f"{k['p']:.2e}"
                    row[f"{prefix}_effect"] = f"{k['effect_size']} ({k['effect_label']})"
                    row[f"{prefix}_sig"] = "yes" if k["significant"] else "no"
            rows.append(row)
        pd.DataFrame(rows).to_csv("main_effects.csv", index=False)
        print(f"\nSaved: main_effects.csv")

    if len(pairs_report):
        pr = pairs_report[pairs_report["sig"] == True].copy()
        out_rows = []
        for _, r in pr.iterrows():
            out_rows.append({
                "variable": r["variable"],
                "measure": "raw" if r["measure"] == "shift" else "absolute",
                "comparison": f"{r['a']} vs {r['b']}",
                "means": f"{r['mean_a']:.3f} vs {r['mean_b']:.3f}",
                "diff": round(r["diff"], 3),
                "effect_r": round(r["rank_biserial_r"], 3),
                "p_adj": f"{r['p_adj']:.2e}",
            })
        pd.DataFrame(out_rows).to_csv("pairwise_comparisons.csv", index=False)
        print(f"Saved: pairwise_comparisons.csv ({len(out_rows)} significant pairs)")

    # Figures
    print("\nGenerating figures...")
    plot_main_effects(results_df, sig_vars)
    print(f"  {FIG_DIR}/main_effects.png")
    plot_distributions(results_df)
    print(f"  {FIG_DIR}/distributions.png")
    plot_interaction(stats_df)
    print(f"  {FIG_DIR}/interaction.png")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
