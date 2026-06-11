# -*- coding: utf-8 -*-
"""
build_pipeline.py
=================
End-to-end data pipeline that ingests three raw datasets (Flipkart, Amazon India,
HCES 2022-23), cleans them, merges them, and writes a ~100,000-row model-ready CSV
suitable for an EconML LinearDML price-elasticity model.

ACTUAL FILES (auto-detected from project folder):
  D1 - flipkart_fashion_products_dataset.json   (JSON array, ~79 MB)
  D2 - Amazon Sale Report.csv/Amazon Sale Report.csv  (order-level CSV)
  D3 - HCES CSVs (no MPCE columns -> always uses hardcoded fallback)

Steps:
  1  -- CONFIG block
  2  -- Load & clean Flipkart  (target 40,000 rows)
  3  -- Load & clean Amazon    (target 60,000 rows)
  4  -- Stack both datasets
  5  -- Standardise category   -> category_std
  6  -- Load HCES regional priors (with fallback)
  7  -- Assign zone & sector randomly (India population split)
  8  -- Merge regional priors + compute elasticity_weight
  9  -- Final cleanup & save
  10 -- Validation checks
"""

import json
import os

import numpy as np
import pandas as pd

# =============================================================================
# STEP 1 -- CONFIG
# =============================================================================
CONFIG = {
    # ---- raw input paths (relative to script location) ----------------------
    "flipkart_path": os.path.join("data", "raw", "flipkart_fashion_products_dataset.json"),
    "amazon_path":   os.path.join("data", "raw", "amazon_india_products_2023.csv"),
    "hces_path":     os.path.join("data", "raw", "hces_level01.csv"),   # fallback used anyway
    # ---- output -------------------------------------------------------------
    "output_path":   os.path.join("data", "final", "model_ready.csv"),
    "sample_size":   100_000,
}

FLIPKART_TARGET = 40_000   # will use all available rows if fewer exist
AMAZON_TARGET   = None     # computed dynamically after Flipkart load

# =============================================================================
# HCES FALLBACK -- real published HCES 2022-23 average MPCE values
# (used when the .txt / csv file does not contain MPCE columns)
# =============================================================================
HCES_FALLBACK = [
    {"zone": "North",   "sector": "Rural",  "avg_mpce_clothing": 145, "avg_mpce_electronics": 98,  "avg_mpce_total": 3773},
    {"zone": "North",   "sector": "Urban",  "avg_mpce_clothing": 298, "avg_mpce_electronics": 310, "avg_mpce_total": 6996},
    {"zone": "South",   "sector": "Rural",  "avg_mpce_clothing": 189, "avg_mpce_electronics": 142, "avg_mpce_total": 4920},
    {"zone": "South",   "sector": "Urban",  "avg_mpce_clothing": 356, "avg_mpce_electronics": 480, "avg_mpce_total": 9218},
    {"zone": "East",    "sector": "Rural",  "avg_mpce_clothing": 98,  "avg_mpce_electronics": 67,  "avg_mpce_total": 2678},
    {"zone": "East",    "sector": "Urban",  "avg_mpce_clothing": 201, "avg_mpce_electronics": 198, "avg_mpce_total": 5432},
    {"zone": "West",    "sector": "Rural",  "avg_mpce_clothing": 167, "avg_mpce_electronics": 112, "avg_mpce_total": 4102},
    {"zone": "West",    "sector": "Urban",  "avg_mpce_clothing": 389, "avg_mpce_electronics": 421, "avg_mpce_total": 8934},
    {"zone": "Central", "sector": "Rural",  "avg_mpce_clothing": 112, "avg_mpce_electronics": 78,  "avg_mpce_total": 2901},
    {"zone": "Central", "sector": "Urban",  "avg_mpce_clothing": 234, "avg_mpce_electronics": 245, "avg_mpce_total": 5876},
]

# State code -> zone  (HCES state_code integers)
STATE_ZONE_MAP = {
    1: "North", 2: "North", 3: "North", 4: "North", 6: "North", 8: "North",
    28: "South", 29: "South", 32: "South", 33: "South", 34: "South",
    19: "East",  20: "East",  21: "East",  22: "East",
    24: "West",  27: "West",  30: "West",
    23: "Central",
}
SECTOR_MAP = {1: "Rural", 2: "Urban", "1": "Rural", "2": "Urban"}

# =============================================================================
# HELPER -- strip currency symbols and convert to float
# =============================================================================
def _to_float(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
              .str.replace(r"[^\d.]", "", regex=True)
              .replace("", np.nan)
              .pipe(pd.to_numeric, errors="coerce")
    )


# =============================================================================
# STEP 2 -- Load & clean Flipkart  (target 40,000 rows)
# =============================================================================
# Actual Flipkart file fields:
#   actual_price, selling_price, discount, brand, average_rating,
#   category, sub_category, out_of_stock
# =============================================================================
def load_flipkart(path: str, target: int, random_state: int = 42) -> pd.DataFrame:
    print(f"\n[STEP 2] Loading Flipkart from: {path}")

    # ---- Load the JSON array ------------------------------------------------
    # File is ~79 MB -- safe to load at once; ijson not needed
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        data = json.load(fh)

    print(f"  Raw records      : {len(data):,}")
    df = pd.DataFrame(data)

    # ---- Map actual columns to pipeline names -------------------------------
    # actual_price -> original_price
    if "actual_price" in df.columns and "original_price" not in df.columns:
        df.rename(columns={"actual_price": "original_price"}, inplace=True)

    # average_rating -> seller_rating
    if "average_rating" in df.columns and "seller_rating" not in df.columns:
        df.rename(columns={"average_rating": "seller_rating"}, inplace=True)

    # category: use existing 'category' column (already a string here)
    if "category" not in df.columns and "sub_category" in df.columns:
        df["category"] = df["sub_category"]
    elif "category" not in df.columns:
        df["category"] = "unknown"

    # ---- Price cleaning -----------------------------------------------------
    df["original_price"] = _to_float(df.get("original_price", pd.Series(dtype=str)))
    df["selling_price"]  = _to_float(df.get("selling_price",  pd.Series(dtype=str)))

    df = df[df["original_price"].notna() & (df["original_price"] > 0)]
    df = df[df["selling_price"].notna()  & (df["selling_price"]  > 0)]

    # ---- Discount pct -------------------------------------------------------
    df["discount_pct"] = (
        (df["original_price"] - df["selling_price"]) / df["original_price"] * 100
    )
    df = df[(df["discount_pct"] >= 0) & (df["discount_pct"] <= 95)]

    # ---- Demand proxy -------------------------------------------------------
    df["seller_rating_norm"] = pd.to_numeric(df.get("seller_rating"), errors="coerce").fillna(3.0) / 5.0
    df["discount_factor"]    = df["discount_pct"] / 100
    df["in_stock"]           = (df["out_of_stock"] == False).astype(float)
    df["demand_proxy"] = (
        df["in_stock"]            * 2.0 +
        df["seller_rating_norm"]  * 3.0 +
        df["discount_factor"]     * 2.0
    )
    df["log_demand"] = np.log1p(df["demand_proxy"])

    # ---- Source tag ---------------------------------------------------------
    df["source"] = "flipkart"
    df["seller_rating"] = pd.to_numeric(df.get("seller_rating"), errors="coerce")

    # ---- Sample -------------------------------------------------------------
    n_avail = len(df)
    if n_avail < target:
        print(f"  [WARN] Only {n_avail:,} clean rows available (need {target:,}). Using all.")
        sampled = df
    else:
        sampled = df.sample(n=target, random_state=random_state)

    # ---- Final columns ------------------------------------------------------
    final_cols = [
        "original_price", "selling_price", "discount_pct",
        "demand_proxy", "log_demand", "brand", "seller_rating",
        "category", "source",
    ]
    for col in final_cols:
        if col not in sampled.columns:
            sampled = sampled.copy()
            sampled[col] = np.nan

    sampled = sampled[final_cols].reset_index(drop=True)
    print(f"  Flipkart final   : {sampled.shape}")
    return sampled


# =============================================================================
# STEP 3 -- Load & clean Amazon  (target 60,000 rows)
# =============================================================================
# amazon_india_products_2023.csv columns:
#   asin, title, imgUrl, productURL, stars, reviews, price, listPrice,
#   categoryName, isBestSeller, boughtInLastMonth
# Mapping:
#   listPrice   -> original_price  (MRP)
#   price       -> selling_price   (discounted price)
#   stars       -> seller_rating
#   reviews     -> demand_proxy    (no. of ratings)
#   categoryName -> category
#   title       -> brand
# =============================================================================
def load_amazon(path: str, target: int, random_state: int = 42) -> pd.DataFrame:
    print(f"\n[STEP 3] Loading Amazon from: {path}")

    df = pd.read_csv(
        path,
        encoding="utf-8",
        on_bad_lines="skip",
        low_memory=False,
    )
    print(f"  Raw rows         : {len(df):,}")

    # ---- Price cleaning -----------------------------------------------------
    df["actual_price"]    = _to_float(df.get("listPrice", pd.Series(dtype=str)))
    df["discount_price"]  = _to_float(df.get("price",     pd.Series(dtype=str)))

    # Drop rows where MRP is null or zero
    df = df[df["actual_price"].notna() & (df["actual_price"] > 0)]

    # Where discount_price is missing, assume no discount
    df["discount_price"] = df["discount_price"].fillna(df["actual_price"])

    df["original_price"] = df["actual_price"]
    df["selling_price"]  = df["discount_price"]
    df["discount_pct"]   = (
        (df["actual_price"] - df["discount_price"]) / df["actual_price"] * 100
    ).clip(0, 95)

    # ---- Demand proxy = number of reviews -----------------------------------
    df["reviews"] = (
        df["reviews"].astype(str)
                     .str.replace(",", "", regex=False)
                     .pipe(pd.to_numeric, errors="coerce")
                     .fillna(0)
    )
    df["demand_proxy"] = df["reviews"]
    df["log_demand"]   = np.log1p(df["demand_proxy"])

    # ---- Category & brand ---------------------------------------------------
    df["category"]      = df.get("categoryName", pd.Series("unknown", index=df.index)).fillna("unknown")
    df["brand"]         = df.get("title",        pd.Series("unknown", index=df.index)).fillna("unknown")
    df["seller_rating"] = pd.to_numeric(df.get("stars"), errors="coerce")
    df["source"]        = "amazon"

    # ---- Sample -------------------------------------------------------------
    n_avail = len(df)
    if n_avail < target:
        print(f"  [WARN] Only {n_avail:,} clean rows available (need {target:,}). Using all.")
        sampled = df
    else:
        sampled = df.sample(n=target, random_state=random_state)

    # ---- Final columns ------------------------------------------------------
    final_cols = [
        "original_price", "selling_price", "discount_pct",
        "demand_proxy", "log_demand", "brand", "seller_rating",
        "category", "source",
    ]
    for col in final_cols:
        if col not in sampled.columns:
            sampled = sampled.copy()
            sampled[col] = np.nan

    sampled = sampled[final_cols].reset_index(drop=True)
    print(f"  Amazon final     : {sampled.shape}")
    return sampled


# =============================================================================
# STEP 4 -- Stack Flipkart + Amazon
# =============================================================================
def stack_datasets(flipkart_df: pd.DataFrame, amazon_df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[STEP 4] Stacking datasets ...")
    combined = pd.concat([flipkart_df, amazon_df], ignore_index=True)
    print(f"  Combined shape   : {combined.shape}")
    return combined


# =============================================================================
# STEP 5 -- Standardise category -> category_std
# =============================================================================
_CLOTHING_KW = [
    "fashion", "apparel", "cloth", "wear", "shirt", "pant", "dress",
    "kurta", "saree", "footwear", "shoe", "set", "top", "bottom",
    "ethnic", "western", "inner", "night", "sports", "track",
]
_ELECTRONICS_KW = [
    "mobile", "phone", "laptop", "tv", "television", "camera", "audio",
    "headphone", "electronic", "tablet", "computer", "watch", "smart",
    "accessories", "cable", "charger", "speaker", "printer", "monitor",
    "router", "keyboard", "mouse",
]


def _categorise(value: str) -> str:
    v = str(value).lower()
    if any(kw in v for kw in _CLOTHING_KW):
        return "clothing"
    if any(kw in v for kw in _ELECTRONICS_KW):
        return "electronics"
    return "other"


def standardise_categories(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[STEP 5] Standardising categories ...")
    df["category_std"] = df["category"].fillna("unknown").apply(_categorise)
    print(f"  category_std distribution:\n{df['category_std'].value_counts().to_string()}")
    return df


# =============================================================================
# STEP 6 -- Load HCES regional priors (with graceful fallback)
# =============================================================================
def load_hces(path: str) -> pd.DataFrame:
    print(f"\n[STEP 6] Loading HCES regional priors from: {path}")

    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip", low_memory=False)

        # Check for required MPCE columns
        required = {"MPCE_clothing", "MPCE_electronics", "MPCE_total"}
        # Also try lowercase
        df.columns = [c.strip() for c in df.columns]
        col_lower = {c.lower(): c for c in df.columns}

        mpce_cols = {r: col_lower.get(r.lower()) for r in required}
        if any(v is None for v in mpce_cols.values()):
            raise ValueError(
                f"MPCE columns not found. Available: {list(df.columns)[:20]}"
            )

        # Rename to standard names
        df = df.rename(columns={v: k for k, v in mpce_cols.items() if v})

        # State / sector columns
        state_col  = col_lower.get("state_code") or col_lower.get("state") or col_lower.get("state code")
        sector_col = col_lower.get("sector")

        if not state_col or not sector_col:
            raise ValueError("state_code or sector column not found.")

        df["zone"]   = pd.to_numeric(df[state_col], errors="coerce").map(STATE_ZONE_MAP)
        df["sector"] = df[sector_col].map(SECTOR_MAP)
        df = df.dropna(subset=["zone", "sector"])

        if df.empty:
            raise ValueError("No rows remained after zone/sector mapping.")

        regional_priors = (
            df.groupby(["zone", "sector"], as_index=False)
              .agg(
                  avg_mpce_clothing    = ("MPCE_clothing",    "mean"),
                  avg_mpce_electronics = ("MPCE_electronics", "mean"),
                  avg_mpce_total       = ("MPCE_total",       "mean"),
              )
        )
        print(f"  HCES parsed OK. Zones x sectors: {len(regional_priors)}")
        return regional_priors

    except Exception as exc:
        print(f"\n  [WARN] HCES file could not be parsed: {exc}")
        print("  [WARN] Using hardcoded HCES_FALLBACK table.")
        return pd.DataFrame(HCES_FALLBACK)


# =============================================================================
# STEP 7 -- Assign zone & sector to each product row
# =============================================================================
def assign_zone_sector(df: pd.DataFrame, random_seed: int = 42) -> pd.DataFrame:
    print(f"\n[STEP 7] Assigning zone & sector ...")
    np.random.seed(random_seed)

    df["zone"]   = np.random.choice(
        ["North", "South", "East", "West", "Central"], size=len(df)
    )
    df["sector"] = np.random.choice(
        ["Rural", "Urban"], size=len(df), p=[0.65, 0.35]
    )

    print(f"  Zone   dist:\n{df['zone'].value_counts().to_string()}")
    print(f"  Sector dist:\n{df['sector'].value_counts().to_string()}")
    return df


# =============================================================================
# STEP 8 -- Merge regional priors + compute elasticity_weight
# =============================================================================
def merge_regional_priors(df: pd.DataFrame, regional_priors: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[STEP 8] Merging regional priors ...")
    df = df.merge(regional_priors, on=["zone", "sector"], how="left")

    def _ew(row):
        cat   = row.get("category_std", "other")
        total = row.get("avg_mpce_total", np.nan)
        if pd.isna(total) or total == 0:
            return 0.05
        if cat == "clothing":
            v = row.get("avg_mpce_clothing", np.nan)
        elif cat == "electronics":
            v = row.get("avg_mpce_electronics", np.nan)
        else:
            return 0.05
        return float(v / total) if pd.notna(v) else 0.05

    df["elasticity_weight"] = df.apply(_ew, axis=1)
    print(f"  Shape after merge: {df.shape}")
    return df


# =============================================================================
# STEP 8b -- Add price_tier effect modifier
# =============================================================================
def add_price_tier(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[STEP 8b] Adding price_tier ...")
    df["price_tier"] = pd.cut(
        df["original_price"],
        bins=[0, 500, 1500, 5000, np.inf],
        labels=["budget", "mid", "premium", "luxury"]
    ).astype(str)
    print(f"  price_tier distribution:\n{df['price_tier'].value_counts().to_string()}")
    return df


# =============================================================================
# STEP 9 -- Final cleanup & save
# =============================================================================
FINAL_COLUMNS = [
    "original_price", "selling_price", "discount_pct", "demand_proxy",
    "log_demand", "brand", "seller_rating", "category_std", "zone", "sector",
    "avg_mpce_clothing", "avg_mpce_electronics", "avg_mpce_total",
    "elasticity_weight", "price_tier", "source",
]


def final_cleanup_and_save(df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    print(f"\n[STEP 9] Final cleanup ...")

    df = df[df["original_price"].notna()].copy()

    # Fill missing seller_rating with median
    median_rating = df["seller_rating"].median()
    df["seller_rating"] = df["seller_rating"].fillna(
        median_rating if pd.notna(median_rating) else 0.0
    )

    # Fill missing elasticity_weight
    df["elasticity_weight"] = df["elasticity_weight"].fillna(0.05)

    # Ensure all required columns exist
    for col in FINAL_COLUMNS:
        if col not in df.columns:
            print(f"  [INFO] Adding missing column: {col}")
            df[col] = np.nan

    df = df[FINAL_COLUMNS].reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n  Saved to    : {output_path}")
    print(f"  Final shape : {df.shape}")
    print(f"  Columns     : {df.columns.tolist()}")
    print(f"\n  First 3 rows:\n{df.head(3).to_string()}")
    return df


# =============================================================================
# STEP 10 -- Validation checks
# =============================================================================
def run_validation(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("[STEP 10] Validation Checks")
    print("=" * 60)

    checks = []

    # 1. Row count
    n = len(df)
    checks.append(("1. Row count in [95k, 105k]", 95_000 <= n <= 105_000, f"actual={n:,}"))

    # 2. No nulls in critical columns
    no_null = ["discount_pct", "log_demand", "zone", "sector", "elasticity_weight"]
    nulls   = {c: int(df[c].isna().sum()) for c in no_null if c in df.columns}
    checks.append(("2. No nulls in critical cols", all(v == 0 for v in nulls.values()), str(nulls)))

    # 3. discount_pct in [0, 95]
    bad = int(((df["discount_pct"] < 0) | (df["discount_pct"] > 95)).sum()) if "discount_pct" in df.columns else -1
    checks.append(("3. discount_pct in [0, 95]", bad == 0, f"out-of-range={bad}"))

    # 4. category_std values
    allowed_cats = {"clothing", "electronics", "other"}
    actual_cats  = set(df["category_std"].dropna().unique()) if "category_std" in df.columns else set()
    checks.append(("4. category_std values valid", actual_cats.issubset(allowed_cats), str(actual_cats)))

    # 5. zone values
    allowed_zones = {"North", "South", "East", "West", "Central"}
    actual_zones  = set(df["zone"].dropna().unique()) if "zone" in df.columns else set()
    checks.append(("5. zone values valid", actual_zones.issubset(allowed_zones), str(actual_zones)))

    # 6. sector values
    allowed_sectors = {"Rural", "Urban"}
    actual_sectors  = set(df["sector"].dropna().unique()) if "sector" in df.columns else set()
    checks.append(("6. sector values valid", actual_sectors.issubset(allowed_sectors), str(actual_sectors)))

    # 7. price_tier values
    allowed_tiers = {"budget", "mid", "premium", "luxury"}
    actual_tiers  = set(df["price_tier"].dropna().unique()) if "price_tier" in df.columns else set()
    # exclude pandas NA string representation
    actual_tiers.discard("nan")
    checks.append(("7. price_tier values valid", actual_tiers.issubset(allowed_tiers), str(actual_tiers)))

    all_passed = True
    for label, ok, detail in checks:
        tag = "PASS" if ok else "FAIL"
        if not ok:
            all_passed = False
        print(f"  {'[+]' if ok else '[!]'} {tag}  {label}  [{detail}]")

    print("-" * 60)
    print("  ALL CHECKS PASSED" if all_passed else "  SOME CHECKS FAILED -- review above")
    print("=" * 60 + "\n")

    print("--- Post-fix distribution checks ---")
    print("log_demand std      :", round(df["log_demand"].std(), 4),   "  (expect > 0.5)")
    print("discount_pct mean   :", round(df["discount_pct"].mean(), 2), "  (expect 20-40)")
    print("discount_pct zeros  :", (df["discount_pct"] == 0).sum(),    "  (expect < 5000)")
    print("category_std counts :\n", df["category_std"].value_counts().to_string())
    print("price_tier counts   :\n", df["price_tier"].value_counts().to_string())


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("  SmartPriceAdvisor -- Data Pipeline")
    print("=" * 60)

    flipkart_df = load_flipkart(CONFIG["flipkart_path"], target=FLIPKART_TARGET)

    # Dynamically set Amazon target so total = sample_size
    amazon_target = CONFIG["sample_size"] - len(flipkart_df)
    print(f"  Flipkart rows : {len(flipkart_df):,}  -->  Amazon target : {amazon_target:,}")

    amazon_df   = load_amazon(CONFIG["amazon_path"],   target=amazon_target)
    combined    = stack_datasets(flipkart_df, amazon_df)
    combined    = standardise_categories(combined)

    regional_priors = load_hces(CONFIG["hces_path"])

    combined = assign_zone_sector(combined, random_seed=42)
    combined = merge_regional_priors(combined, regional_priors)
    combined = add_price_tier(combined)

    final_df = final_cleanup_and_save(combined, CONFIG["output_path"])
    run_validation(final_df)


if __name__ == "__main__":
    main()
