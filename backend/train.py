# -*- coding: utf-8 -*-
"""
train.py -- SmartPriceAdvisor Backend
======================================
Stages 1-3: Data preparation, Double ML training, Regional elasticity table.

Run:
    python train.py
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from econml.dml import LinearDML

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
MODEL_DIR       = os.path.join(BASE_DIR, "models")
RAW_CSV         = os.path.join(DATA_DIR, "model_ready_v2.csv")
PROCESSED_CSV   = os.path.join(DATA_DIR, "processed_clothing.csv")
REGIONAL_CSV    = os.path.join(DATA_DIR, "regional_elasticity.csv")
TEST_PRED_CSV   = os.path.join(DATA_DIR, "test_predictions.csv")
MODEL_PKL       = os.path.join(MODEL_DIR, "dml_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# =============================================================================
# STAGE 1 -- DATA PREPARATION
# =============================================================================
def stage1_prepare() -> tuple:
    print("\n" + "=" * 65)
    print("STAGE 1 -- Data Preparation")
    print("=" * 65)

    df = pd.read_csv(RAW_CSV)
    print(f"  Loaded         : {len(df):,} rows  |  {df.shape[1]} columns")

    # Filter clothing only
    df = df[df["category_std"] == "clothing"].copy()
    print(f"  After clothing filter: {len(df):,} rows")

    # Fill null brand
    df["brand"] = df["brand"].fillna("unknown")

    # One-hot encode: zone, sector, price_tier
    # Define all expected categories so encoding is deterministic
    zone_cats     = ["Central", "East", "North", "South", "West"]
    sector_cats   = ["Rural", "Urban"]
    tier_cats     = ["budget", "luxury", "mid", "premium"]

    df["zone"]       = pd.Categorical(df["zone"],       categories=zone_cats)
    df["sector"]     = pd.Categorical(df["sector"],     categories=sector_cats)
    df["price_tier"] = pd.Categorical(df["price_tier"], categories=tier_cats)

    zone_dummies   = pd.get_dummies(df["zone"],       prefix="zone",       drop_first=False)
    sector_dummies = pd.get_dummies(df["sector"],     prefix="sector",     drop_first=False)
    tier_dummies   = pd.get_dummies(df["price_tier"], prefix="price_tier", drop_first=False)

    # Drop one category per group to avoid perfect multicollinearity
    # Baselines: Central, Rural, budget
    for col in ["zone_Central", "sector_Rural", "price_tier_budget"]:
        if col in zone_dummies.columns:   zone_dummies.drop(columns=[col], inplace=True)
        if col in sector_dummies.columns: sector_dummies.drop(columns=[col], inplace=True)
        if col in tier_dummies.columns:   tier_dummies.drop(columns=[col], inplace=True)

    df = pd.concat([df.reset_index(drop=True),
                    zone_dummies.reset_index(drop=True),
                    sector_dummies.reset_index(drop=True),
                    tier_dummies.reset_index(drop=True)], axis=1)

    # Column groups
    zone_cols   = sorted([c for c in df.columns if c.startswith("zone_")])
    sector_cols = sorted([c for c in df.columns if c.startswith("sector_")])
    tier_cols   = sorted([c for c in df.columns if c.startswith("price_tier_")])

    # X: effect modifiers -- capture heterogeneous elasticity
    x_cols = zone_cols + sector_cols + tier_cols + ["elasticity_weight"]

    # W: pure confounders -- control for but don't estimate heterogeneity
    w_cols = ["avg_mpce_total", "avg_mpce_clothing", "seller_rating"]

    y_col = "log_demand"    # outcome
    t_col = "selling_price" # treatment

    # Drop rows with nulls in key columns
    key_cols = [y_col, t_col] + x_cols + w_cols + ["original_price"]
    df = df.dropna(subset=key_cols)
    print(f"  After null drop : {len(df):,} rows")

    # Save processed data
    df.to_csv(PROCESSED_CSV, index=False)
    print(f"  Saved processed_clothing.csv  ({len(df):,} rows)")

    # 80/20 split
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    print(f"  Train: {len(df_train):,}  |  Test: {len(df_test):,}")
    print(f"\n  X effect modifiers ({len(x_cols)}): {x_cols}")
    print(f"  W confounders     ({len(w_cols)}): {w_cols}")
    print(f"  Y (outcome)  : {y_col}")
    print(f"  T (treatment): {t_col}")

    return df_train, df_test, x_cols, w_cols, y_col, t_col


# =============================================================================
# STAGE 2 -- DOUBLE ML MODEL
# =============================================================================
def stage2_train(df_train, df_test, x_cols, w_cols, y_col, t_col) -> tuple:
    print("\n" + "=" * 65)
    print("STAGE 2 -- Double ML Model (EconML LinearDML)")
    print("=" * 65)

    Y_train = df_train[y_col].values.astype(float)
    T_train = df_train[t_col].values.astype(float)
    X_train = df_train[x_cols].values.astype(float)
    W_train = df_train[w_cols].values.astype(float)

    Y_test  = df_test[y_col].values.astype(float)
    T_test  = df_test[t_col].values.astype(float)
    X_test  = df_test[x_cols].values.astype(float)
    W_test  = df_test[w_cols].values.astype(float)

    model_y = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model_t = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)

    dml = LinearDML(
        model_y=model_y,
        model_t=model_t,
        random_state=42,
    )

    print("  Fitting LinearDML (nuisance RF models + causal final stage)...")
    print("  [This may take 3-5 minutes on first run]")
    sys.stdout.flush()
    dml.fit(Y=Y_train, T=T_train, X=X_train, W=W_train)
    print("  Fitting complete!")

    # Print causal coefficients
    try:
        coefs = dml.coef_
        intercept = float(dml.intercept_)
        print("\n  Causal treatment-effect coefficients theta(X):")
        for col, c in zip(x_cols, coefs):
            print(f"    {col:35s}: {c:+.6f}")
        print(f"    {'intercept':35s}: {intercept:+.6f}")
    except Exception as exc:
        print(f"  (Coefficient display skipped: {exc})")

    # CATE on test set: effect(X) = theta(X) = per-rupee causal elasticity
    cate_test = dml.effect(X_test)
    print(f"\n  CATE stats on test set:")
    print(f"    mean  : {cate_test.mean():.6f}  (per Rs.1 change in selling_price)")
    print(f"    std   : {cate_test.std():.6f}")
    print(f"    min   : {cate_test.min():.6f}")
    print(f"    max   : {cate_test.max():.6f}")

    # Standalone first-stage RF for visualization residuals
    print("\n  Fitting standalone RF for residual visualization...")
    XW_train = np.hstack([X_train, W_train])
    XW_test  = np.hstack([X_test,  W_test])
    vis_model_y = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    vis_model_y.fit(XW_train, Y_train)
    y_pred_test  = vis_model_y.predict(XW_test)
    residuals    = Y_test - y_pred_test
    r2_score     = 1 - np.sum(residuals**2) / np.sum((Y_test - Y_test.mean())**2)
    print(f"  First-stage model_y  R2 on test : {r2_score:.4f}")

    # Save test predictions
    df_test = df_test.copy()
    df_test["cate"]       = cate_test
    df_test["y_pred"]     = y_pred_test
    df_test["residuals"]  = residuals
    df_test.to_csv(TEST_PRED_CSV, index=False)
    print(f"  Saved test_predictions.csv  ({len(df_test):,} rows)")

    # Compute zone+sector mean log_demand for baseline lookups
    zone_sector_baselines = (
        df_train.groupby(["zone", "sector"])[y_col].mean().to_dict()
    )

    # Save model artifact
    artifact = {
        "model":                  dml,
        "x_cols":                 x_cols,
        "w_cols":                 w_cols,
        "y_col":                  y_col,
        "t_col":                  t_col,
        "baseline_log_demand":    float(Y_train.mean()),
        "zone_sector_baselines":  zone_sector_baselines,
    }
    joblib.dump(artifact, MODEL_PKL)
    print(f"  Saved dml_model.pkl")

    return dml, df_test, cate_test


# =============================================================================
# STAGE 3 -- REGIONAL ELASTICITY TABLE
# =============================================================================
def stage3_regional(df_test: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 65)
    print("STAGE 3 -- Regional Elasticity Table")
    print("=" * 65)

    regional = (
        df_test
        .groupby(["zone", "sector"])
        .agg(
            mean_elasticity    = ("cate",           "mean"),
            std_elasticity     = ("cate",           "std"),
            sample_count       = ("cate",           "count"),
            avg_mpce_total     = ("avg_mpce_total", "mean"),
            avg_original_price = ("original_price", "mean"),
            avg_selling_price  = ("selling_price",  "mean"),
        )
        .reset_index()
    )

    # Round for readability
    for col in ["mean_elasticity", "std_elasticity", "avg_mpce_total",
                "avg_original_price", "avg_selling_price"]:
        regional[col] = regional[col].round(6)

    print("\n  Regional Elasticity by Zone x Sector:")
    print(regional.to_string(index=False))

    regional.to_csv(REGIONAL_CSV, index=False)
    print(f"\n  Saved regional_elasticity.csv  ({len(regional)} zone-sector groups)")

    return regional


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 65)
    print("  SmartPriceAdvisor -- Training Pipeline")
    print("=" * 65)

    df_train, df_test, x_cols, w_cols, y_col, t_col = stage1_prepare()
    dml, df_test, cate_test = stage2_train(
        df_train, df_test, x_cols, w_cols, y_col, t_col
    )
    regional = stage3_regional(df_test)

    print("\n" + "=" * 65)
    print("  All stages complete!")
    print(f"  Model    : backend/models/dml_model.pkl")
    print(f"  Data     : backend/data/processed_clothing.csv")
    print(f"             backend/data/test_predictions.csv")
    print(f"             backend/data/regional_elasticity.csv")
    print("=" * 65)


if __name__ == "__main__":
    main()
