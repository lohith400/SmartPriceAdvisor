# -*- coding: utf-8 -*-
"""
visualizations.py -- SmartPriceAdvisor Backend
================================================
Stage 6: Generate and save all 8 analysis plots as PNG files.

Run AFTER train.py:
    python visualizations.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.lines import Line2D

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "data")
MODEL_DIR     = os.path.join(BASE_DIR, "models")
PLOTS_DIR     = os.path.join(BASE_DIR, "plots")

PROCESSED_CSV = os.path.join(DATA_DIR, "processed_clothing.csv")
TEST_PRED_CSV = os.path.join(DATA_DIR, "test_predictions.csv")
REGIONAL_CSV  = os.path.join(DATA_DIR, "regional_elasticity.csv")
MODEL_PKL     = os.path.join(MODEL_DIR, "dml_model.pkl")

os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
})
ZONE_PALETTE = {
    "North":   "#4C72B0",
    "South":   "#DD8452",
    "East":    "#55A868",
    "West":    "#C44E52",
    "Central": "#8172B3",
}


def save(fig, name: str):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {name}")


# ── Load data ─────────────────────────────────────────────────────────────────
def load_data():
    for f in [PROCESSED_CSV, TEST_PRED_CSV, REGIONAL_CSV, MODEL_PKL]:
        if not os.path.exists(f):
            print(f"[ERROR] Missing: {f}")
            print("Run train.py first, then re-run visualizations.py")
            sys.exit(1)

    processed = pd.read_csv(PROCESSED_CSV)
    test_pred = pd.read_csv(TEST_PRED_CSV)
    regional  = pd.read_csv(REGIONAL_CSV)
    artifact  = joblib.load(MODEL_PKL)
    return processed, test_pred, regional, artifact


# =============================================================================
# PLOT 1 -- Heatmap: Price Elasticity by Region
# =============================================================================
def plot_elasticity_by_region(regional: pd.DataFrame):
    pivot = regional.pivot(index="zone", columns="sector", values="mean_elasticity")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        pivot, annot=True, fmt=".5f", cmap="RdYlGn_r",
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "Mean CATE (per Rs.1 change in price)"}
    )
    ax.set_title("Price Elasticity by Region\n(Causal Effect via Double ML)")
    ax.set_xlabel("Sector")
    ax.set_ylabel("Zone")
    fig.tight_layout()
    save(fig, "elasticity_by_region.png")


# =============================================================================
# PLOT 2 -- Scatter: Demand vs Price by Zone
# =============================================================================
def plot_demand_vs_price(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 6))
    zones = df["zone"].unique()

    for zone in sorted(zones):
        sub = df[df["zone"] == zone].sample(min(800, len(df[df["zone"] == zone])), random_state=42)
        color = ZONE_PALETTE.get(zone, "#888888")
        ax.scatter(sub["selling_price"], sub["log_demand"],
                   alpha=0.25, s=15, color=color, label=None)

        # Regression line
        m, b = np.polyfit(sub["selling_price"], sub["log_demand"], 1)
        x_line = np.linspace(sub["selling_price"].min(), sub["selling_price"].max(), 100)
        ax.plot(x_line, m * x_line + b, color=color, linewidth=2, label=zone)

    ax.set_title("Demand vs Price by Zone")
    ax.set_xlabel("Selling Price (Rs.)")
    ax.set_ylabel("Log Demand")
    ax.legend(title="Zone", framealpha=0.9)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"Rs.{x:,.0f}"))
    fig.tight_layout()
    save(fig, "demand_vs_price.png")


# =============================================================================
# PLOT 3 -- Grouped Bar: Optimized Price vs Flat 30% vs Original
# =============================================================================
def plot_optimization_waterfall(artifact: dict):
    from optimizer import optimize_price

    samples = [
        dict(label="Budget\nEast Rural",    original_price=400,  zone="East",  sector="Rural",  price_tier="budget"),
        dict(label="Mid\nSouth Urban",      original_price=1000, zone="South", sector="Urban",  price_tier="mid"),
        dict(label="Premium\nNorth Urban",  original_price=3000, zone="North", sector="Urban",  price_tier="premium"),
        dict(label="Mid\nWest Rural",       original_price=800,  zone="West",  sector="Rural",  price_tier="mid"),
        dict(label="Premium\nSouth Urban",  original_price=4000, zone="South", sector="Urban",  price_tier="premium"),
    ]

    labels        = []
    orig_prices   = []
    flat_prices   = []
    opt_prices    = []

    for s in samples:
        res = optimize_price(artifact, **{k: v for k, v in s.items() if k != "label"})
        labels.append(s["label"])
        orig_prices.append(s["original_price"])
        flat_prices.append(res["flat_price"])
        opt_prices.append(res["recommended_price"])

    x    = np.arange(len(labels))
    w    = 0.25
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.bar(x - w,   orig_prices,  w, label="Original Price",    color="#4C72B0", alpha=0.85)
    ax.bar(x,       flat_prices,  w, label="Flat 30% Discount", color="#DD8452", alpha=0.85)
    ax.bar(x + w,   opt_prices,   w, label="Optimized Price",   color="#55A868", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Price (Rs.)")
    ax.set_title("Optimized Price vs Flat 30% Discount vs Original Price")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"Rs.{x:,.0f}"))
    fig.tight_layout()
    save(fig, "optimization_waterfall.png")


# =============================================================================
# PLOT 4 -- Bar: Profit Improvement %
# =============================================================================
def plot_profit_improvement(artifact: dict):
    from optimizer import optimize_price

    samples = [
        dict(label="Budget/East Rural",    original_price=400,  zone="East",  sector="Rural",  price_tier="budget"),
        dict(label="Mid/South Urban",      original_price=1000, zone="South", sector="Urban",  price_tier="mid"),
        dict(label="Premium/North Urban",  original_price=3000, zone="North", sector="Urban",  price_tier="premium"),
        dict(label="Mid/West Rural",       original_price=800,  zone="West",  sector="Rural",  price_tier="mid"),
        dict(label="Premium/South Urban",  original_price=4000, zone="South", sector="Urban",  price_tier="premium"),
    ]

    labels     = []
    profits    = []
    for s in samples:
        res = optimize_price(artifact, **{k: v for k, v in s.items() if k != "label"})
        labels.append(s["label"])
        profits.append(res["profit_improvement_pct"])

    colors = ["#2ecc71" if p >= 0 else "#e74c3c" for p in profits]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, profits, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, profits):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.3 if val >= 0 else -1.5),
                f"{val:+.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Profit Improvement (%)")
    ax.set_title("Profit Improvement: Smart Pricing vs Flat 30% Discount")
    ax.set_xlabel("Product Segment")
    fig.tight_layout()
    save(fig, "profit_improvement.png")


# =============================================================================
# PLOT 5 -- Histogram: Elasticity Distribution
# =============================================================================
def plot_elasticity_distribution(test_pred: pd.DataFrame):
    cate = test_pred["cate"].dropna()
    mean_val = cate.mean()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(cate, bins=60, color="#4C72B0", alpha=0.75, edgecolor="white", linewidth=0.5)
    ax.axvline(mean_val, color="#C44E52", linewidth=2,
               label=f"Mean = {mean_val:.5f}")
    ax.set_xlabel("CATE  (causal elasticity per Rs.1 price change)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Price Elasticity Across Products\n(Conditional Average Treatment Effects)")
    ax.legend()
    fig.tight_layout()
    save(fig, "elasticity_distribution.png")


# =============================================================================
# PLOT 6 -- Scatter: Purchasing Power vs Price Sensitivity
# =============================================================================
def plot_mpce_vs_elasticity(regional: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 6))

    for _, row in regional.iterrows():
        color = ZONE_PALETTE.get(row["zone"], "#888")
        ax.scatter(row["avg_mpce_total"], row["mean_elasticity"],
                   s=row["sample_count"] / 5 + 50,
                   color=color, alpha=0.85, edgecolors="white", linewidth=0.8, zorder=5)
        ax.annotate(
            f"{row['zone']}-{row['sector']}",
            (row["avg_mpce_total"], row["mean_elasticity"]),
            textcoords="offset points", xytext=(6, 4), fontsize=8.5,
        )

    # Legend for zones
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
               markersize=8, label=z)
        for z, c in ZONE_PALETTE.items()
    ]
    ax.legend(handles=handles, title="Zone", loc="upper right")
    ax.set_xlabel("Avg Monthly Household Spend (Rs.)")
    ax.set_ylabel("Mean Causal Elasticity (CATE)")
    ax.set_title("Purchasing Power vs Price Sensitivity\n(Size = sample count)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"Rs.{x:,.0f}"))
    fig.tight_layout()
    save(fig, "mpce_vs_elasticity.png")


# =============================================================================
# PLOT 7 -- Model Residuals
# =============================================================================
def plot_model_residuals(test_pred: pd.DataFrame):
    y_actual = test_pred["log_demand"].dropna()
    y_pred   = test_pred.loc[y_actual.index, "y_pred"]
    residuals = test_pred.loc[y_actual.index, "residuals"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Predicted vs Actual
    lims = [min(y_actual.min(), y_pred.min()),
            max(y_actual.max(), y_pred.max())]
    ax1.scatter(y_pred, y_actual, alpha=0.15, s=10, color="#4C72B0")
    ax1.plot(lims, lims, color="#C44E52", linewidth=1.5, linestyle="--", label="Perfect fit")
    ax1.set_xlabel("Predicted log_demand")
    ax1.set_ylabel("Actual log_demand")
    ax1.set_title("Predicted vs Actual (First-Stage Model Y)")
    ax1.legend()

    r2 = 1 - np.sum(residuals**2) / np.sum((y_actual - y_actual.mean())**2)
    ax1.text(0.05, 0.92, f"R\u00b2 = {r2:.4f}", transform=ax1.transAxes,
             fontsize=10, bbox=dict(facecolor="white", alpha=0.7))

    # Residual distribution
    ax2.hist(residuals, bins=60, color="#55A868", alpha=0.75, edgecolor="white", linewidth=0.4)
    ax2.axvline(0, color="#C44E52", linewidth=1.5, linestyle="--")
    ax2.set_xlabel("Residual (Actual - Predicted)")
    ax2.set_ylabel("Count")
    ax2.set_title("Residual Distribution")

    fig.suptitle("Double ML -- Model Y Residuals (Demand Prediction Quality)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "model_residuals.png")


# =============================================================================
# PLOT 8 -- Box Plot: Discount Distribution by Price Tier
# =============================================================================
def plot_price_tier_discount(processed: pd.DataFrame):
    tier_order = ["budget", "mid", "premium", "luxury"]
    tier_colors = ["#55A868", "#4C72B0", "#DD8452", "#8172B3"]

    fig, ax = plt.subplots(figsize=(9, 6))
    data_by_tier = [
        processed.loc[processed["price_tier"] == t, "discount_pct"].dropna()
        for t in tier_order
    ]

    bp = ax.boxplot(
        data_by_tier, tick_labels=tier_order, patch_artist=True,
        medianprops=dict(color="white", linewidth=2),
        flierprops=dict(marker="o", alpha=0.3, markersize=3),
    )
    for patch, color in zip(bp["boxes"], tier_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xlabel("Price Tier")
    ax.set_ylabel("Discount %")
    ax.set_title("Discount Distribution by Price Tier")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    # Annotate medians
    for i, tier in enumerate(tier_order):
        med = data_by_tier[i].median()
        ax.text(i + 1, med + 1, f"{med:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color="black")

    fig.tight_layout()
    save(fig, "price_tier_discount.png")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("  SmartPriceAdvisor -- Visualizations")
    print("=" * 60)

    processed, test_pred, regional, artifact = load_data()
    print(f"  Loaded processed_clothing.csv : {len(processed):,} rows")
    print(f"  Loaded test_predictions.csv   : {len(test_pred):,} rows")
    print(f"  Loaded regional_elasticity.csv: {len(regional)} groups")
    print(f"\n  Generating 8 plots into: {PLOTS_DIR}\n")

    plot_elasticity_by_region(regional)
    plot_demand_vs_price(processed)
    plot_optimization_waterfall(artifact)
    plot_profit_improvement(artifact)
    plot_elasticity_distribution(test_pred)
    plot_mpce_vs_elasticity(regional)
    plot_model_residuals(test_pred)
    plot_price_tier_discount(processed)

    print(f"\n  All 8 plots saved to: {PLOTS_DIR}")
    print("  Done!")


if __name__ == "__main__":
    main()
