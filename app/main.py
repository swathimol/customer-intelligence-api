"""Customer Intelligence API: /predict, /segment, /recommend"""
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# STEP 8.1: Where the model files live (override with MODEL_DIR)
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
ART: dict = {}          # loaded once at startup, shared by all requests


def logit(p):
    """Same helper used when the Platt calibrator was fitted."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


# STEP 8.2: Load every artifact once, when the server starts
@asynccontextmanager
async def lifespan(app: FastAPI):
    ART["features"] = json.load(open(MODEL_DIR / "features.json"))
    ART["model"] = joblib.load(MODEL_DIR / "best_model.pkl")
    ART["platt"] = joblib.load(MODEL_DIR / "calibrator.pkl")
    ART["seg"] = joblib.load(MODEL_DIR / "segmenter.pkl")
    ART["seg_names"] = {int(k): v for k, v in
                        json.load(open(MODEL_DIR / "segment_names.json")).items()}
    ART["rec"] = joblib.load(MODEL_DIR / "recommender.pkl")
    meta = pd.read_parquet(MODEL_DIR / "items_meta.parquet")
    ART["meta"] = meta.set_index("article_id").to_dict("index")
    yield
    ART.clear()


app = FastAPI(title="Customer Intelligence API", version="1.0.0", lifespan=lifespan)


# STEP 8.3: Request schemas
class PredictRequest(BaseModel):
    features: Dict[str, Optional[float]] = Field(
        ..., description="All model features by name (see GET /features). "
                         "Use null for avg_days_between if the customer has one order.")


class SegmentRequest(BaseModel):
    recency: float = Field(..., ge=0, description="Days since last order")
    frequency: float = Field(..., ge=0, description="Number of orders")
    monetary: float = Field(..., ge=0, description="Total spend (relative units)")
    avg_basket_value: float = Field(..., ge=0)
    tenure: float = Field(..., ge=0, description="Days since first order")


# STEP 8.4: Health and feature list
@app.get("/health")
def health():
    return {"status": "ok",
            "n_features": len(ART["features"]),
            "recommender_cutoff": ART["rec"]["cutoff"],
            "recommender_users": len(ART["rec"]["u_ix"])}


@app.get("/features")
def features():
    return {"features": ART["features"]}


# STEP 8.5: /predict - 90-day repurchase probability
@app.post("/predict")
def predict(req: PredictRequest):
    feats = ART["features"]
    missing = [f for f in feats if f not in req.features]
    unknown = [k for k in req.features if k not in feats]
    if missing or unknown:
        raise HTTPException(422, {"missing": missing, "unknown": unknown})

    # build one row in the exact training column order; null -> NaN
    row = {f: (np.nan if req.features[f] is None else req.features[f]) for f in feats}
    X = pd.DataFrame([row], columns=feats).astype(float)

    raw = float(ART["model"].predict_proba(X)[:, 1][0])
    prob = float(ART["platt"].predict_proba(logit(np.array([raw])))[:, 1][0])
    return {"raw_score": round(raw, 4),
            "repurchase_probability_90d": round(prob, 4),
            "note": "Platt-calibrated on 2019-09 data; may still over-predict "
                    "if the base repurchase rate keeps falling."}


# STEP 8.6: /segment - RFM segment (same transform as training)
@app.post("/segment")
def segment(req: SegmentRequest):
    seg = ART["seg"]
    X = pd.DataFrame([req.model_dump()])[seg["features"]]
    Z = seg["scaler"].transform(np.log1p(X.clip(lower=0)))
    sid = int(seg["kmeans"].predict(Z)[0])
    return {"segment_id": sid, "segment": ART["seg_names"][sid]}


# STEP 8.7: /recommend - hybrid recommender with popular fallback
@app.get("/recommend/{cust_idx}")
def recommend(cust_idx: int, n: int = Query(12, ge=1, le=50)):
    rec = ART["rec"]
    ix = rec["u_ix"].get(cust_idx)

    if ix is None:                           # cold start / no recent history
        scores, source = rec["pop_vec"].copy(), "popular_fallback"
    else:                                    # same scoring as the evaluated version
        scores = (rec["M"][ix] @ rec["SIM"]).toarray().ravel()
        if scores.max() > 0:
            scores = scores / scores.max()
        scores = scores + rec["alpha"] * rec["pop_vec"]
        source = "hybrid"

    top = np.argpartition(-scores, n)[:n]
    top = top[np.argsort(-scores[top])]
    items = [{"article_id": int(rec["items"][i]),
              "score": round(float(scores[i]), 4),
              **ART["meta"].get(int(rec["items"][i]), {})} for i in top]
    return {"cust_idx": cust_idx, "source": source,
            "cutoff": rec["cutoff"], "items": items}