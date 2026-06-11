# -*- coding: utf-8 -*-
"""
optimizer.py -- SmartPriceAdvisor Backend
==========================================
Stage 4: Price optimization engine using the trained LinearDML model.

Usage (standalone):
    from optimizer import optimize_price
    result = optimize_price("path/to/dml_model.pkl", original_price=1500,
                             zone="South", sector="Urban", ...)
"""

import os
import numpy as np
import joblib
from typing import Optional

# ── HCES 2022-23 Lookup Table ─────────────────────────────────────────────────
# Fallback when user does not provide avg_mpce_total / avg_mpce_clothing
HCES_TABLE = {
    ("North",   "Rural"): {"avg_mpce_total": 3773, "avg_mpce_clothing": 145},
    ("North",   "Urban"): {"avg_mpce_total": 6996, "avg_mpce_clothing": 298},
    ("South",   "Rural"): {"avg_mpce_total": 4920, "avg_mpce_clothing": 189},
    ("South",   "Urban"): {"avg_mpce_total": 9218, "avg_mpce_clothing": 356},
    ("East",    "Rural"): {"avg_mpce_total": 2678, "avg_mpce_clothing":  98},
    ("East",    "Urban"): {"avg_mpce_total": 5432, "avg_mpce_clothing": 201},
    ("West",    "Rural"): {"avg_mpce_total": 4102, "avg_mpce_clothing": 167},
    ("West",    "Urban"): {"avg_mpce_total": 8934, "avg_mpce_clothing": 389},
    ("Central", "Rural"): {"avg_mpce_total": 2901, "avg_mpce_clothing": 112},
    ("Central", "Urban"): {"avg_mpce_total": 5876, "avg_mpce_clothing": 234},
}

# OHE column definitions (must match train.py exactly)
# Baselines dropped: zone_Central, sector_Rural, price_tier_budget
_ZONE_COLS   = ["zone_East", "zone_North", "zone_South", "zone_West"]
_SECTOR_COLS = ["sector_Urban"]
_TIER_COLS   = ["price_tier_luxury", "price_tier_mid", "price_tier_premium"]
_X_COLS      = _ZONE_COLS + _SECTOR_COLS + _TIER_COLS + ["elasticity_weight"]


def _build_x_row(zone: str, sector: str, price_tier: str, elasticity_weight: float) -> np.ndarray:
    """
    Build the X feature vector matching the training column order.
    Baselines: zone=Central, sector=Rural, price_tier=budget (all zeros).
    """
    row = np.zeros(len(_X_COLS), dtype=float)

    zone_key   = f"zone_{zone}"
    sector_key = f"sector_{sector}"
    tier_key   = f"price_tier_{price_tier}"

    for i, col in enumerate(_X_COLS):
        if col == "elasticity_weight":
            row[i] = elasticity_weight
        elif col in (zone_key, sector_key, tier_key):
            row[i] = 1.0

    return row


def _infer_price_tier(original_price: float) -> str:
    """Infer price_tier from original_price using the same bins as the pipeline."""
    if original_price <= 500:
        return "budget"
    elif original_price <= 1500:
        return "mid"
    elif original_price <= 5000:
        return "premium"
    else:
        return "luxury"


def optimize_price(
    artifact_or_path,
    original_price: float,
    zone: str,
    sector: str,
    seller_rating: float = 4.0,
    avg_mpce_total: Optional[int] = None,
    avg_mpce_clothing: Optional[int] = None,
    price_tier: Optional[str] = None,
    n_steps: int = 50,
) -> dict:
    """
    Find the profit-maximizing selling price using the trained LinearDML model.

    Parameters
    ----------
    artifact_or_path : str or dict
        Path to dml_model.pkl OR the already-loaded artifact dict.
    original_price   : float   MRP / listed price
    zone             : str     North / South / East / West / Central
    sector           : str     Rural / Urban
    seller_rating    : float   Product rating (default 4.0)
    avg_mpce_total   : int     Monthly household spend; if None, looked up from HCES table
    avg_mpce_clothing: int     Monthly clothing spend; if None, looked up from HCES table
    price_tier       : str     budget / mid / premium / luxury; if None, inferred from price
    n_steps          : int     Number of candidate prices to evaluate (default 50)

    Returns
    -------
    dict with recommended price and comparison against flat 30% discount.
    """
    # Load artifact if path is given
    if isinstance(artifact_or_path, str):
        artifact = joblib.load(artifact_or_path)
    else:
        artifact = artifact_or_path

    dml                  = artifact["model"]
    baseline_log_demand  = artifact["baseline_log_demand"]
    zone_sector_baselines = artifact["zone_sector_baselines"]

    # Fill optional params
    if avg_mpce_total is None or avg_mpce_clothing is None:
        hces = HCES_TABLE.get((zone, sector), {"avg_mpce_total": 5000, "avg_mpce_clothing": 200})
        avg_mpce_total    = avg_mpce_total    or hces["avg_mpce_total"]
        avg_mpce_clothing = avg_mpce_clothing or hces["avg_mpce_clothing"]

    if price_tier is None:
        price_tier = _infer_price_tier(original_price)

    # Use zone+sector baseline log_demand if available
    base_ld = zone_sector_baselines.get((zone, sector), baseline_log_demand)

    # Cost floor = 55% of original price
    cost_floor = original_price * 0.55

    # Candidate selling prices: 40% to 100% of original price
    candidates = np.linspace(original_price * 0.40, original_price * 1.0, n_steps)

    # Build all X rows at once (one per candidate price) for vectorized scoring
    x_rows = np.array([
        _build_x_row(
            zone, sector, price_tier,
            elasticity_weight=float(np.clip(p / avg_mpce_total, 0.0, 1.0))
        )
        for p in candidates
    ])

    # Get CATE for all candidates at once
    effects = dml.effect(x_rows)  # shape (n_steps,)

    best_profit  = -np.inf
    best_result  = None
    results      = []

    for i, (cand_price, effect) in enumerate(zip(candidates, effects)):
        pred_log_demand = base_ld + effect * (cand_price - original_price)
        pred_demand     = float(np.exp(pred_log_demand))
        revenue         = cand_price * pred_demand
        profit          = revenue - (cost_floor * pred_demand)
        margin_pct      = (profit / revenue * 100) if revenue > 0 else 0.0

        if margin_pct < 15.0:
            continue

        results.append({
            "price":       cand_price,
            "profit":      profit,
            "demand":      pred_demand,
            "margin_pct":  margin_pct,
            "log_demand":  pred_log_demand,
        })

        if profit > best_profit:
            best_profit = profit
            best_result = {
                "price":      cand_price,
                "profit":     profit,
                "demand":     pred_demand,
                "margin_pct": margin_pct,
            }

    # If no candidate passes margin filter, take max-revenue among all
    if best_result is None:
        revenues = np.array([
            candidates[i] * float(np.exp(base_ld + effects[i] * (candidates[i] - original_price)))
            for i in range(n_steps)
        ])
        best_idx = int(np.argmax(revenues))
        cand_price  = candidates[best_idx]
        effect_val  = effects[best_idx]
        pred_log_d  = base_ld + effect_val * (cand_price - original_price)
        pred_demand = float(np.exp(pred_log_d))
        revenue     = cand_price * pred_demand
        profit      = revenue - cost_floor * pred_demand
        best_result = {
            "price":      cand_price,
            "profit":     profit,
            "demand":     pred_demand,
            "margin_pct": (profit / revenue * 100) if revenue > 0 else 0.0,
        }

    # Flat 30% discount benchmark
    flat_price      = original_price * 0.70
    flat_ew         = float(np.clip(flat_price / avg_mpce_total, 0.0, 1.0))
    flat_x          = _build_x_row(zone, sector, price_tier, flat_ew).reshape(1, -1)
    flat_effect     = float(dml.effect(flat_x)[0])
    flat_log_d      = base_ld + flat_effect * (flat_price - original_price)
    flat_demand     = float(np.exp(flat_log_d))
    flat_revenue    = flat_price * flat_demand
    flat_profit     = flat_revenue - cost_floor * flat_demand
    flat_margin_pct = (flat_profit / flat_revenue * 100) if flat_revenue > 0 else 0.0

    rec_price       = float(best_result["price"])
    rec_discount    = float((original_price - rec_price) / original_price * 100)
    profit_impr     = float(
        (best_result["profit"] - flat_profit) / abs(flat_profit) * 100
    ) if flat_profit != 0 else 0.0

    return {
        "recommended_price":        round(rec_price, 2),
        "recommended_discount_pct": round(rec_discount, 2),
        "predicted_demand_index":   round(float(best_result["demand"]), 4),
        "predicted_margin_pct":     round(float(best_result["margin_pct"]), 2),
        "flat_price":               round(flat_price, 2),
        "flat_discount_pct":        30.0,
        "flat_demand_index":        round(flat_demand, 4),
        "flat_margin_pct":          round(flat_margin_pct, 2),
        "profit_improvement_pct":   round(profit_impr, 2),
        # ── metadata ──
        "zone":                     zone,
        "sector":                   sector,
        "price_tier":               price_tier,
        "original_price":           original_price,
        "avg_mpce_total":           avg_mpce_total,
        "avg_mpce_clothing":        avg_mpce_clothing,
    }


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    pkl_path = os.path.join(os.path.dirname(__file__), "models", "dml_model.pkl")
    if not os.path.exists(pkl_path):
        print("dml_model.pkl not found. Run train.py first.")
        sys.exit(1)

    print("Loading model...")
    artifact = joblib.load(pkl_path)

    test_cases = [
        dict(original_price=400,  zone="East",  sector="Rural",  price_tier="budget"),
        dict(original_price=1000, zone="South",  sector="Urban",  price_tier="mid"),
        dict(original_price=3000, zone="North",  sector="Urban",  price_tier="premium"),
        dict(original_price=800,  zone="West",   sector="Rural",  price_tier="mid"),
        dict(original_price=4000, zone="South",  sector="Urban",  price_tier="premium"),
    ]

    print("\n{'='*70}")
    print(" Price Optimization Results")
    print("{'='*70}")
    for tc in test_cases:
        res = optimize_price(artifact, **tc)
        print(f"\n  Input  : Rs.{tc['original_price']} | {tc['zone']} {tc['sector']} | {tc['price_tier']}")
        print(f"  Optimal: Rs.{res['recommended_price']:.0f} ({res['recommended_discount_pct']:.1f}% off)  "
              f"margin={res['predicted_margin_pct']:.1f}%  demand={res['predicted_demand_index']:.4f}")
        print(f"  Flat30 : Rs.{res['flat_price']:.0f}  "
              f"margin={res['flat_margin_pct']:.1f}%  demand={res['flat_demand_index']:.4f}")
        print(f"  Profit improvement: {res['profit_improvement_pct']:+.1f}%")
