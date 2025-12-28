import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------ CONFIG ------------------
LEAKFREE_DIR = "./data/finalParquet_leakfree"
RAW_COMP_PATH = "./data/raw/updated_compustat_data.csv"   # or ./data/raw/compustat_data.csv
OUT_DIR = "./data/signals"
os.makedirs(OUT_DIR, exist_ok=True)

FEATURE_COLS = [
    "gross_profitability","operating_margin","sales_growth","asset_growth",
    "debt_to_assets","log_assets","asset_turnover","roe","roa",
    "innovation","integrity","quality","respect","teamwork","MEANREC"
]
ID_COLS = ["gvkey","conm","tic","fyear"]
TARGET = "ep"   # E/P label (already present in the leak-free parquet)

# ------------------ LOAD LEAK-FREE DATA ------------------
train = pd.read_parquet(f"{LEAKFREE_DIR}/train.parquet")
test  = pd.read_parquet(f"{LEAKFREE_DIR}/test.parquet")

X_train, y_train = train[FEATURE_COLS], train[TARGET]
X_test,  y_test  = test[FEATURE_COLS],  test[TARGET]

# ------------------ TRAIN BEST OVERALL MODEL (GBDT) ------------------
SEED = 42
gb = GradientBoostingRegressor(
    loss="absolute_error",
    n_estimators=300,
    learning_rate=0.1,
    max_depth=3,
    random_state=SEED
)
gb.fit(X_train, y_train)

# Eval (sanity check)
pred_train = gb.predict(X_train)
pred_test  = gb.predict(X_test)

mae_tr = mean_absolute_error(y_train, pred_train)
rmse_tr = np.sqrt(mean_squared_error(y_train, pred_train))
r2_tr = r2_score(y_train, pred_train)

mae_te = mean_absolute_error(y_test, pred_test)
rmse_te = np.sqrt(mean_squared_error(y_test, pred_test))
r2_te = r2_score(y_test, pred_test)

print(f"[GBDT] Train  MAE={mae_tr:.4f} RMSE={rmse_tr:.4f} R2={r2_tr:.3f}")
print(f"[GBDT] Test   MAE={mae_te:.4f} RMSE={rmse_te:.4f} R2={r2_te:.3f}")

# ------------------ BRING IN ACTUAL PRICE & EPS/SHARE ------------------
# We need EPS per share to convert predicted EP -> implied price.
# EPS_ps = (nits_clean / csho), with nits_clean fallback to oibdp if nits missing (consistent with your preprocessing).
raw = pd.read_csv(RAW_COMP_PATH)
for c in ["nits","oibdp","csho","prcc_f","gvkey","tic","conm","fyear"]:
    if c not in raw.columns:
        raw[c] = np.nan

# ensure numeric
for c in ["nits","oibdp","csho","prcc_f","fyear","gvkey"]:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")

raw = raw.sort_values(["gvkey","fyear"]).reset_index(drop=True)
raw["nits_clean"] = np.where(raw["nits"].notna(), raw["nits"], raw["oibdp"])
eps_ps = raw["nits_clean"] / raw["csho"]
raw["eps_ps"] = eps_ps.replace([np.inf,-np.inf], np.nan)

# de-dupe to one row per (gvkey,fyear,tic) with the *latest* record in year if duplicates
raw_key = (raw
    .sort_values(["gvkey","fyear"])
    .dropna(subset=["fyear","gvkey"])
    .drop_duplicates(subset=["gvkey","fyear","tic"], keep="last")
    [["gvkey","tic","conm","fyear","prcc_f","eps_ps"]]
)

# ------------------ MAKE INDICATORS ON TEST SPLIT ONLY ------------------
out = test[ID_COLS].copy()
out["ep_actual"] = y_test.values
out["ep_pred"]   = pred_test

# merge price & EPS/ps
out = out.merge(raw_key, on=["gvkey","tic","conm","fyear"], how="left")

# implied price = EPS / predicted_EP (only valid when both >0)
out["implied_price"] = np.where(
    (out["eps_ps"].notna()) & (out["ep_pred"] > 0),
    out["eps_ps"] / out["ep_pred"],
    np.nan
)

# mispricing = (implied - actual) / actual
out["mispricing_pct"] = (out["implied_price"] - out["prcc_f"]) / out["prcc_f"]

# Simple indicator logic:
#   If implied > actual => model thinks the stock should be higher => potentially UNDERVALUED => LONG
#   If implied < actual => potentially OVERVALUED => SHORT
# Add safety margin with quantile buckets.
valid = out["mispricing_pct"].dropna()
if len(valid) >= 20:
    q_lo = valid.quantile(0.20)
    q_hi = valid.quantile(0.80)
else:
    # fallback thresholds if too few rows
    q_lo, q_hi = -0.05, 0.05

def signal_row(m):
    if pd.isna(m):
        return "HOLD"
    if m >= q_hi:
        return "LONG"
    if m <= q_lo:
        return "SHORT"
    return "HOLD"

out["signal"] = out["mispricing_pct"].apply(signal_row)

# Also provide a continuous score (z-score of mispricing)
mp = out["mispricing_pct"]
out["score"] = (mp - mp.mean()) / (mp.std() if mp.std() not in [0, np.nan] else 1.0)

# Rank for convenience (higher rank = more underpriced by model)
out["rank_underpriced"] = out["mispricing_pct"].rank(ascending=False, method="first")

# Tidy columns
cols = ID_COLS + [
    "prcc_f","eps_ps","ep_actual","ep_pred","implied_price",
    "mispricing_pct","score","signal","rank_underpriced"
]
out = out[cols]

# ------------------ SAVE ------------------
out_path = os.path.join(OUT_DIR, "ep_price_indicator.csv")
out.to_csv(out_path, index=False)
print(f"\nSaved indicator file → {out_path}")
print("Columns:")
print(", ".join(out.columns))
