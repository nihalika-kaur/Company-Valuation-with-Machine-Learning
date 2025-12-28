###prior to this script we compute
#medians 
# winsorization bounds (1%–99%) 
# scaling 
# means 
# scaling 
# standard deviations
#but since we use the entire dataset (2010-2025), training-years data was cleaned/standardized using future-years information.
#since we split into: DATA LEAKAGE
#train (e.g., 2010–2019) and (2020–2021 or 2023–2025)

#in this code we split first, then only preprocess the training dataset

# scripts/final_preprocessing_leakfree.py


import os
import numpy as np
import pandas as pd

# --- CONFIG ---
COMP_PATH    = "./data/raw/updated_compustat_data.csv"
CULTURE_PATH = "./data/raw/culture_scores.csv"
CONS_PATH    = "./data/raw/AnalysisInfo.csv"
OUT_DIR      = "./data/finalParquet_leakfree"

TRAIN_END_YEAR = 2019
TEST_MAX_YEAR  = 2021

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading Compustat...")
df = pd.read_csv(COMP_PATH)

numeric_cols = ["at","ceq","dlc","lt","cogs","nits","oibdp","revt","capx","csho","prcc_f","fyear"]
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.sort_values(["gvkey","fyear"]).reset_index(drop=True)

# earnings fallback
df["nits_clean"] = np.where(df["nits"].notna(), df["nits"], df["oibdp"])

print("Loading Culture...")
culture = pd.read_csv(CULTURE_PATH)
culture.columns = culture.columns.str.lower()
culture = culture.rename(columns={
    "year":"fyear",
    "s_innovation":"innovation",
    "s_integrity":"integrity",
    "s_quality":"quality",
    "s_respect":"respect",
    "s_teamwork":"teamwork"
})
culture["gvkey"] = pd.to_numeric(culture["gvkey"], errors="coerce")
culture["fyear"] = pd.to_numeric(culture["fyear"], errors="coerce")

culture_cols = ["innovation","integrity","quality","respect","teamwork"]
culture_clean = culture.groupby(["gvkey","fyear"])[culture_cols].mean().reset_index()

print("Merging culture into Compustat...")
df = pd.merge(df, culture_clean, on=["gvkey","fyear"], how="left")

# --- FEATURE ENGINEERING ---
def safe_div(a,b):
    return np.where(b != 0, a/b, np.nan)

df["gross_profitability"] = safe_div(df["revt"] - df["cogs"], df["at"])
df["operating_margin"]    = safe_div(df["oibdp"], df["revt"])
# add fill_method=None to avoid forward-fill deprecation warning
df["sales_growth"]        = df.groupby("gvkey")["revt"].pct_change(fill_method=None)
df["asset_growth"]        = df.groupby("gvkey")["at"].pct_change(fill_method=None)
df["debt_to_assets"]      = safe_div(df["lt"], df["at"])
df["log_assets"]          = np.log(df["at"].replace(0, np.nan))
df["asset_turnover"]      = safe_div(df["revt"], df["at"])
df["roe"]                 = safe_div(df["nits_clean"], df["ceq"])
df["roa"]                 = safe_div(df["nits_clean"], df["at"])

# Target: E/P = (earnings per share) / price
df["ep"] = safe_div(df["nits_clean"] / df["csho"], df["prcc_f"])

# Basic hygiene (keep like original)
df.replace([np.inf,-np.inf], np.nan, inplace=True)
df = df[df["ep"].notna()].copy()

# --- TIME SPLIT (do this BEFORE fitting any stats to avoid leakage) ---
train = df[df["fyear"] <= TRAIN_END_YEAR].copy()
test  = df[(df["fyear"] > TRAIN_END_YEAR) & (df["fyear"] <= TEST_MAX_YEAR)].copy()

# ---------------------------------------------------------------------
# NEW: LEAK-FREE TARGET CLEANUP & WINSORIZATION (TRAIN-ONLY BOUNDS)
#   - drop silly prices that blow up ratios (price <= 1.0)
#   - clip extreme EP values using TRAIN 1%/99% quantiles
#   - apply those TRAIN bounds to TEST (no leakage)
# ---------------------------------------------------------------------
# price filter
train = train[train["prcc_f"] > 1.0].copy()
test  = test[test["prcc_f"] > 1.0].copy()

# ensure no infs remain
train.replace([np.inf, -np.inf], np.nan, inplace=True)
test.replace([np.inf, -np.inf], np.nan, inplace=True)

# drop missing ep (safety in case price filter created NaNs)
train = train[train["ep"].notna()].copy()
test  = test[test["ep"].notna()].copy()

# TRAIN-only EP winsor cutoffs
ep_lo = train["ep"].quantile(0.01)
ep_hi = train["ep"].quantile(0.99)

# apply same cutoffs to both splits
train["ep"] = train["ep"].clip(lower=ep_lo, upper=ep_hi)
test["ep"]  = test["ep"].clip(lower=ep_lo, upper=ep_hi)

print(f"Target 'ep' clipped to TRAIN range: {ep_lo:.4f} to {ep_hi:.4f}")
# ---------------------------------------------------------------------

print("Loading Analyst Consensus...")
cons = pd.read_csv(CONS_PATH)
cons["fyear"] = cons["STATPERS"].astype(str).str[:4].astype(int)
cons_yearly = cons.groupby(["oftic","fyear"], as_index=False).agg({"MEANREC":"mean"})

train = pd.merge(train, cons_yearly, left_on=["tic","fyear"], right_on=["oftic","fyear"], how="left")
test  = pd.merge(test,  cons_yearly, left_on=["tic","fyear"], right_on=["oftic","fyear"], how="left")

for d in (train,test):
    if "oftic" in d.columns:
        d.drop(columns=["oftic"], inplace=True)

# --- CLEANING WITH TRAIN-ONLY STATS (unchanged from original) ---
feature_cols = [
    "gross_profitability","operating_margin","sales_growth","asset_growth",
    "debt_to_assets","log_assets","asset_turnover","roe","roa",
    "innovation","integrity","quality","respect","teamwork",
    "MEANREC"
]

# compute training-only bounds & stats
q_lo = train[feature_cols].quantile(0.01)
q_hi = train[feature_cols].quantile(0.99)
meds = train[feature_cols].median()
mean_ = train[feature_cols].mean()
std_  = train[feature_cols].std().replace(0,1.0)

def apply_cleaning(df_part):
    out = df_part.copy()
    out[feature_cols] = out[feature_cols].clip(lower=q_lo, upper=q_hi, axis=1)
    out[feature_cols] = out[feature_cols].fillna(meds)
    out[feature_cols] = (out[feature_cols] - mean_) / std_
    return out

train_clean = apply_cleaning(train)
test_clean  = apply_cleaning(test)

id_cols = ["gvkey","conm","tic","fyear"]
target = "ep"

train_final = pd.concat([train[id_cols].reset_index(drop=True),
                         train_clean[feature_cols].reset_index(drop=True),
                         train[target].reset_index(drop=True)], axis=1)

test_final = pd.concat([test[id_cols].reset_index(drop=True),
                        test_clean[feature_cols].reset_index(drop=True),
                        test[target].reset_index(drop=True)], axis=1)

# --- SAVE ---
train_final.to_parquet(f"{OUT_DIR}/train.parquet", index=False)
test_final.to_parquet(f"{OUT_DIR}/test.parquet", index=False)

print("\nSUCCESS — leak-free preprocessing complete!")
print(f"Train rows: {len(train_final)}, Test rows: {len(test_final)}")

