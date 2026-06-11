# -*- coding: utf-8 -*-
"""
main.py -- SmartPriceAdvisor FastAPI Backend
=============================================
Stage 5: REST API for price optimization and regional elasticity lookup.

Run:
    uvicorn main:app --reload --port 8000
"""

import os
import json
import joblib
import pandas as pd
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from optimizer import optimize_price, HCES_TABLE, _infer_price_tier

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_PKL      = os.path.join(BASE_DIR, "models", "dml_model.pkl")
REGIONAL_CSV   = os.path.join(BASE_DIR, "data",   "regional_elasticity.csv")

# ── Global model state ────────────────────────────────────────────────────────
_artifact: dict = {}


# ── Lifespan (startup + shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup; release on shutdown."""
    global _artifact
    if not os.path.exists(MODEL_PKL):
        raise RuntimeError(
            f"dml_model.pkl not found at {MODEL_PKL}. "
            "Run backend/train.py first."
        )
    print(f"[startup] Loading model from {MODEL_PKL} ...")
    _artifact = joblib.load(MODEL_PKL)
    print("[startup] Model loaded successfully.")
    yield
    # Shutdown
    _artifact.clear()
    print("[shutdown] Model released.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SmartPriceAdvisor API",
    description=(
        "Hyper-personalized markdown & pricing optimization for Indian fashion retail. "
        "Uses EconML LinearDML to estimate regional price elasticity and find the "
        "profit-maximizing discount per zone and sector."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────
class PredictRequest(BaseModel):
    original_price:    float             = Field(...,   gt=0,   description="MRP / listed price in Rs.")
    zone:              str               = Field(...,           description="North / South / East / West / Central")
    sector:            str               = Field(...,           description="Rural / Urban")
    seller_rating:     float             = Field(4.0,   ge=0,   le=5, description="Product rating (0-5)")
    avg_mpce_total:    Optional[int]     = Field(None,          description="Avg monthly household spend (auto-filled from HCES if None)")
    avg_mpce_clothing: Optional[int]     = Field(None,          description="Avg monthly clothing spend (auto-filled from HCES if None)")
    price_tier:        Optional[str]     = Field(None,          description="budget / mid / premium / luxury (auto-inferred if None)")

    model_config = {"json_schema_extra": {
        "example": {
            "original_price": 1500,
            "zone": "South",
            "sector": "Urban",
            "seller_rating": 4.2,
            "price_tier": "mid"
        }
    }}


class PredictResponse(BaseModel):
    recommended_price:        float
    recommended_discount_pct: float
    predicted_demand_index:   float
    predicted_margin_pct:     float
    flat_price:               float
    flat_discount_pct:        float
    flat_demand_index:        float
    flat_margin_pct:          float
    profit_improvement_pct:   float
    zone:                     str
    sector:                   str
    price_tier:               str
    original_price:           float
    avg_mpce_total:           int
    avg_mpce_clothing:        int


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health_check():
    """Check if the API is running and the model is loaded."""
    model_status = "loaded" if _artifact else "not_loaded"
    return {"status": "ok", "model": model_status}


@app.post("/predict", response_model=PredictResponse, tags=["Pricing"])
def predict(req: PredictRequest):
    """
    Recommend the profit-maximizing selling price for a product.

    Uses the trained LinearDML causal model to estimate regional price
    elasticity and find the optimal discount versus a flat 30% markdown.
    """
    if not _artifact:
        raise HTTPException(status_code=503, detail="Model not loaded. Retry shortly.")

    # Validate zone / sector
    valid_zones   = {"North", "South", "East", "West", "Central"}
    valid_sectors = {"Rural", "Urban"}
    valid_tiers   = {"budget", "mid", "premium", "luxury", None}

    if req.zone not in valid_zones:
        raise HTTPException(status_code=422, detail=f"zone must be one of {valid_zones}")
    if req.sector not in valid_sectors:
        raise HTTPException(status_code=422, detail=f"sector must be one of {valid_sectors}")
    if req.price_tier not in valid_tiers:
        raise HTTPException(status_code=422, detail=f"price_tier must be one of {valid_tiers - {None}}")

    # Fill HCES defaults
    hces = HCES_TABLE.get((req.zone, req.sector), {"avg_mpce_total": 5000, "avg_mpce_clothing": 200})
    avg_mpce_total    = req.avg_mpce_total    or hces["avg_mpce_total"]
    avg_mpce_clothing = req.avg_mpce_clothing or hces["avg_mpce_clothing"]
    price_tier        = req.price_tier        or _infer_price_tier(req.original_price)

    try:
        result = optimize_price(
            artifact_or_path   = _artifact,
            original_price     = req.original_price,
            zone               = req.zone,
            sector             = req.sector,
            seller_rating      = req.seller_rating,
            avg_mpce_total     = avg_mpce_total,
            avg_mpce_clothing  = avg_mpce_clothing,
            price_tier         = price_tier,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(exc)}")

    return result


@app.get("/regional_elasticity", tags=["Analytics"])
def get_regional_elasticity():
    """
    Return the regional price elasticity table as JSON.

    Each row represents a zone x sector combination with:
    - mean_elasticity: causal effect of Rs.1 price change on log_demand
    - std_elasticity:  uncertainty
    - sample_count
    - avg_mpce_total
    - avg_original_price / avg_selling_price
    """
    if not os.path.exists(REGIONAL_CSV):
        raise HTTPException(
            status_code=404,
            detail="regional_elasticity.csv not found. Run train.py first."
        )
    df = pd.read_csv(REGIONAL_CSV)
    return df.to_dict(orient="records")


@app.get("/hces_table", tags=["Analytics"])
def get_hces_table():
    """Return the HCES 2022-23 regional spending table used as defaults."""
    records = [
        {
            "zone":              zone,
            "sector":            sector,
            "avg_mpce_total":    vals["avg_mpce_total"],
            "avg_mpce_clothing": vals["avg_mpce_clothing"],
        }
        for (zone, sector), vals in HCES_TABLE.items()
    ]
    return records


# ── Dev runner ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
