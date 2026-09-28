"""
Figures for the KBC case study.
Run from the notebook after master/lab/tgt/oof/out/imp are in scope.

    import sys; sys.path.append("src")
    from plots import *
    make_all_figures(master, lab, tgt, oof, out, imp, PRODUCTS)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

OUT = "../outputs/figures"
COLORS = {"MF": "#4C72B0", "CC": "#DD8452", "CL": "#55A868"}


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png", dpi=150)
    plt.close(fig)


# ---------- FOR THE EXECUTIVE SUMMARY (keep these simple, minimal jargon) ----------


def plot_revenue_comparison():
    """Strategy vs. baselines, from the back-test (§8 of the technical report)."""
    labels = [
        "Full mix\n(strategy)",
        "Random clients,\nall CL",
        "Random clients,\nrandom offer",
    ]
    vals = [6.62, 3.61, 2.70]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(labels, vals, color=["#2E7D32", "#9E9E9E", "#BDBDBD"])
    ax.bar_label(bars, fmt="%.2f")
    ax.set_ylabel("Revenue per contact")
    ax.set_title("Back-tested revenue per contact: strategy vs. baselines")
    _save(fig, "revenue_comparison")


def plot_offer_mix(out):
    """Offer mix among the 97 selected clients."""
    counts = out.Offer.value_counts().reindex(["CL", "CC", "MF"]).fillna(0)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    colors = [COLORS[p] for p in counts.index]
    wedges, _, autotexts = ax.pie(
        counts,
        labels=[f"{p} ({int(n)})" for p, n in counts.items()],
        autopct=lambda pct: f"{pct:.0f}%" if pct > 0 else "",
        colors=colors,
        startangle=90,
    )
    ax.set_title("Offer mix (97 contacted clients)")
    _save(fig, "offer_mix")


def plot_lift(oof, lab, PRODUCTS):
    """Top-15% lift per product — the headline model-quality number."""
    lifts = []
    for p in PRODUCTS:
        y = lab[f"Sale_{p}"].astype(int).to_numpy()
        idx = np.argsort(-oof[p])[: int(len(y) * 0.15)]
        lifts.append(y[idx].mean() / y.mean())
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    bars = ax.bar(PRODUCTS, lifts, color=[COLORS[p] for p in PRODUCTS])
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Random targeting")
    ax.bar_label(bars, fmt="%.2fx")
    ax.set_ylabel("Lift vs. random targeting")
    ax.set_title("Conversion lift in the top 15% of ranked clients")
    ax.legend()
    _save(fig, "lift_by_product")


# ---------- FOR THE TECHNICAL REPORT (more detail, one per finding) ----------


def plot_key_distributions(master):
    """Age, tenure, current-account balance (log) — full client base."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].hist(master.Age, bins=30, color="#607D8B")
    axes[0].set_title("Age")
    axes[1].hist(master.Tenure, bins=30, color="#607D8B")
    axes[1].set_title("Tenure (months)")
    axes[2].hist(
        np.sign(master.ActBal_CA) * np.log1p(np.abs(master.ActBal_CA)),
        bins=30,
        color="#607D8B",
    )
    axes[2].set_title("Current-account balance (signed log)")
    fig.suptitle("Key distributions (full client base, n=1,615)")
    _save(fig, "key_distributions")


def plot_product_ownership(master):
    """Prevalence of each product, full base vs. no-flow-data clients."""
    flags = ["has_SA", "has_MF", "has_OVD", "has_CC", "has_CL"]
    all_rate = master[flags].mean()
    gap_rate = master.loc[master.no_flow_data, flags].mean()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(flags))
    ax.bar(x - 0.2, all_rate, width=0.4, label="Full base", color="#4C72B0")
    ax.bar(x + 0.2, gap_rate, width=0.4, label="No flow data (n=28)", color="#DD8452")
    ax.set_xticks(x, [f.replace("has_", "") for f in flags])
    ax.set_ylabel("Ownership rate")
    ax.set_title("Product ownership (full base vs. no-flow-data clients)")
    ax.legend()
    _save(fig, "ownership_no_flow_gap")


def plot_class_balance(lab, PRODUCTS):
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    rates = [lab[f"Sale_{p}"].mean() for p in PRODUCTS]
    bars = ax.bar(PRODUCTS, rates, color=[COLORS[p] for p in PRODUCTS])
    ax.bar_label(bars, fmt="%.3f")
    ax.set_ylabel("Sale rate")
    ax.set_title("Base sale rate by product (labeled set, n=969)")
    _save(fig, "class_balance")


def plot_revenue_distributions(lab, PRODUCTS):
    """Revenue among buyers only — shows the heavy tail, esp. for CC."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, p in zip(axes, PRODUCTS):
        r = lab.loc[lab[f"Sale_{p}"] == 1, f"Revenue_{p}"]
        ax.hist(r, bins=30, color=COLORS[p])
        ax.axvline(
            r.mean(), color="black", linestyle="--", label=f"mean={r.mean():.1f}"
        )
        ax.axvline(
            r.median(), color="gray", linestyle=":", label=f"median={r.median():.1f}"
        )
        ax.set_title(f"{p} revenue (buyers only)")
        ax.legend(fontsize=8)
    fig.suptitle("Revenue distributions (buyers only)")
    _save(fig, "revenue_distributions")


def plot_calibration(oof, lab, PRODUCTS):
    """Reliability diagram: predicted vs. observed rate, 5 bins per product."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, p in zip(axes, PRODUCTS):
        y = lab[f"Sale_{p}"].astype(int).to_numpy()
        pred = oof[p]
        bins = np.quantile(pred, np.linspace(0, 1, 6))
        bin_id = np.digitize(pred, bins[1:-1])
        pred_means = [pred[bin_id == i].mean() for i in range(5)]
        obs_means = [y[bin_id == i].mean() for i in range(5)]
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
        ax.plot(pred_means, obs_means, "o-", color=COLORS[p])
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed rate")
        ax.set_title(p)
        ax.legend(fontsize=8)
    fig.suptitle("Calibration (oof blend predictions, 5 bins)")
    _save(fig, "calibration")


def plot_feature_importance(imp):
    """imp: DataFrame, index=feature groups, columns=PRODUCTS (from group_importance)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    imp.plot(kind="barh", ax=ax, color=[COLORS[c] for c in imp.columns])
    ax.set_xlabel("Mean AUC drop when group is permuted")
    ax.set_title("Group-wise permutation importance")
    _save(fig, "feature_importance")


def plot_strategy_ablation():
    """Back-test revenue across strategy variants (§8)."""
    labels = ["Full mix", "CC + CL only", "CL-only"]
    vals = [959.8, 1005.6, 1154.6]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(labels, vals, color=["#2E7D32", "#66BB6A", "#A5D6A7"])
    ax.bar_label(bars, fmt="%.0f")
    ax.set_ylabel("Realized revenue (145 contacts)")
    ax.set_title("Strategy ablation")
    _save(fig, "strategy_ablation")


def make_all_figures(master, lab, tgt, oof, out, imp, PRODUCTS):
    plot_revenue_comparison()
    plot_offer_mix(out)
    plot_lift(oof, lab, PRODUCTS)
    plot_key_distributions(master)
    plot_product_ownership(master)
    plot_class_balance(lab, PRODUCTS)
    plot_revenue_distributions(lab, PRODUCTS)
    plot_calibration(oof, lab, PRODUCTS)
    plot_feature_importance(imp)
    plot_strategy_ablation()
    print(f"Saved figures to {OUT}/")
