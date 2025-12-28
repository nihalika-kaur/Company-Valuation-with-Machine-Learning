# import pandas as pd
# import numpy as np
# import os

# # get raw data
# dataset = pd.read_csv('./data/raw/updated_compustat_data.csv')

# # ensure numeric
# numeric_cols = ['at','ceq','dlc','lt','cogs','nits','oibdp','revt','capx','csho','prcc_f','fyear']
# for col in numeric_cols:
#     dataset[col] = pd.to_numeric(dataset[col], errors='coerce')

# # sort by company & fiscal year
# dataset = dataset.sort_values(['gvkey', 'fyear']).reset_index(drop=True)

# #create nits clean
# dataset['nits_clean'] = np.where(
#     ~dataset['nits'].isna(),
#     dataset['nits'],
#     dataset['oibdp']  # fallback if nits is missing
# )

# # FEATURE CALCULATIONS
# dataset['gross_profitability'] = (dataset['revt'] - dataset['cogs']) / dataset['at']
# dataset['operating_margin'] = dataset['oibdp'] / dataset['revt']
# dataset['sales_growth'] = dataset.groupby('gvkey')['revt'].pct_change(fill_method=None)
# dataset['asset_growth'] = dataset.groupby('gvkey')['at'].pct_change(fill_method=None)
# dataset['debt_to_assets'] = dataset['lt'] / dataset['at']
# dataset['log_assets'] = np.log(dataset['at'].replace(0, np.nan))
# dataset['asset_turnover'] = dataset['revt'] / dataset['at']
# dataset['roe'] = np.where(dataset['ceq'] > 0, dataset['nits_clean'] / dataset['ceq'], np.nan)
# dataset['roa'] = np.where(dataset['at'] > 0, dataset['nits_clean'] / dataset['at'], np.nan)
# dataset['ep'] = np.where(dataset['prcc_f'] > 0, (dataset['nits_clean'] / dataset['csho']) / dataset['prcc_f'], np.nan)

# # CLEANING
# # drop invalid target rows
# dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
# dataset = dataset[~dataset['ep'].isna()].copy()

# # fill missing features with median 
# features_to_fill = ['gross_profitability','operating_margin','sales_growth','asset_growth',
#                     'debt_to_assets','log_assets','asset_turnover','roe','roa']
# for col in features_to_fill:
#     dataset[col] = dataset[col].fillna(dataset[col].median())

# # winsorize extreme (1st and 99th percentile)
# q_lo = dataset[features_to_fill].quantile(0.01)
# q_hi = dataset[features_to_fill].quantile(0.99)
# dataset[features_to_fill] = dataset[features_to_fill].clip(lower=q_lo, upper=q_hi, axis=1)

# # TRAIN TEST SPLIT
# cutoff_year = 2021  
# training = dataset[dataset['fyear'] <= cutoff_year].copy()
# testing = dataset[dataset['fyear'] > cutoff_year].copy()

# feature_cols = ['gross_profitability','operating_margin','sales_growth','asset_growth',
#                 'debt_to_assets','log_assets','asset_turnover','roe','roa']

# X_train = training[feature_cols]
# y_train = training['ep']
# X_test = testing[feature_cols]
# y_test = testing['ep']

# # SCALE FEATURES
# train_mean = X_train.mean()
# train_std = X_train.std().replace(0, 1.0)

# X_train_scaled = (X_train - train_mean) / train_std
# X_test_scaled = (X_test - train_mean) / train_std

# # reattach ids and target
# train_out = training[['gvkey','conm','tic','fyear']].copy()
# test_out = testing[['gvkey','conm','tic','fyear']].copy()

# train_out[feature_cols] = X_train_scaled
# test_out[feature_cols] = X_test_scaled

# train_out['ep'] = y_train
# test_out['ep'] = y_test

# # SAVE TO DISK
# os.makedirs('./data/new_processed', exist_ok=True)
# train_out.to_parquet('./data/new_processed/train.parquet', index=False)
# test_out.to_parquet('./data/new_processed/test.parquet', index=False)

# print("Saved processed dataset:")
# print("Train rows:", len(train_out))
# print("Test rows:", len(test_out)



# USING CORP CULTURE DATASET 

import pandas as pd
import numpy as np
import os

# load data 
print("Loading data...")
# load compustat
dataset = pd.read_csv('./data/raw/updated_compustat_data.csv')

# load culture
try:
    culture_df = pd.read_csv('./data/raw/culture_scores.csv')
    print(f"Culture Data Loaded: {len(culture_df)} rows")
except FileNotFoundError:
    print("ERROR: File './data/raw/culture_scores.csv' not found.")
    exit()

# CLEANING CULTURE DATASET

# stardardizing column names to lowercase
culture_df.columns = culture_df.columns.str.lower()


# mapping columns from the csv to the clean names we want
rename_map = {
    'year': 'fyear',            # renaming year to fyear
    's_integrity': 'integrity', # s_integrity to integrity
    's_teamwork': 'teamwork',
    's_innovation': 'innovation',
    's_respect': 'respect',
    's_quality': 'quality'
}
culture_df.rename(columns=rename_map, inplace=True)

# make sure gvkey is numeric in both datasets for merging
dataset['gvkey'] = pd.to_numeric(dataset['gvkey'], errors='coerce')
culture_df['gvkey'] = pd.to_numeric(culture_df['gvkey'], errors='coerce')

# columns we want to keep
culture_cols = ['innovation', 'integrity', 'quality', 'respect', 'teamwork']

# check columns exist now
missing = [c for c in culture_cols if c not in culture_df.columns]
if missing:
    print(f"ERROR: After renaming, still missing columns: {missing}")
    print(f"Columns present: {culture_df.columns.tolist()}")
    exit()

# potential duplicates (take average if multiple rows per gvkey/year)
culture_clean = culture_df.groupby(['gvkey', 'fyear'])[culture_cols].mean().reset_index()

# MERGING THE DATASETS

print("Merging financial data with culture data...")
# left join: keep all rows, if culture missing put NAN
dataset = pd.merge(dataset, culture_clean, on=['gvkey', 'fyear'], how='left')

# FINANCIAL FEATURE CALCULATIONS
numeric_cols = ['at','ceq','dlc','lt','cogs','nits','oibdp','revt','capx','csho','prcc_f','fyear']
for col in numeric_cols:
    dataset[col] = pd.to_numeric(dataset[col], errors='coerce')

dataset = dataset.sort_values(['gvkey', 'fyear']).reset_index(drop=True)

dataset['nits_clean'] = np.where(~dataset['nits'].isna(), dataset['nits'], dataset['oibdp'])

dataset['gross_profitability'] = (dataset['revt'] - dataset['cogs']) / dataset['at']
dataset['operating_margin'] = dataset['oibdp'] / dataset['revt']
dataset['sales_growth'] = dataset.groupby('gvkey')['revt'].pct_change(fill_method=None)
dataset['asset_growth'] = dataset.groupby('gvkey')['at'].pct_change(fill_method=None)
dataset['debt_to_assets'] = dataset['lt'] / dataset['at']
dataset['log_assets'] = np.log(dataset['at'].replace(0, np.nan))
dataset['asset_turnover'] = dataset['revt'] / dataset['at']
dataset['roe'] = np.where(dataset['ceq'] > 0, dataset['nits_clean'] / dataset['ceq'], np.nan)
dataset['roa'] = np.where(dataset['at'] > 0, dataset['nits_clean'] / dataset['at'], np.nan)
dataset['ep'] = np.where(dataset['prcc_f'] > 0, (dataset['nits_clean'] / dataset['csho']) / dataset['prcc_f'], np.nan)

# # drop rows where target ep is missing
# dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
# dataset = dataset[~dataset['ep'].isna()].copy()

# drop penny stocks (< $1)
# companies with stock price < $1 behave erratically and ruin valuation models
dataset = dataset[dataset['prcc_f'] > 1.0].copy()

# drop invalid targets
dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
dataset = dataset[~dataset['ep'].isna()].copy()

# winsorize target 'ep' 
# this clips extreme outliers (like EP = 7000.0) that are destroying model performance
ep_lo = dataset['ep'].quantile(0.01)
ep_hi = dataset['ep'].quantile(0.99)
dataset['ep'] = dataset['ep'].clip(lower=ep_lo, upper=ep_hi)

print(f"Target 'ep' clipped to range: {ep_lo:.4f} to {ep_hi:.4f}")

# HANDLE MISSING VALUES

# list of all features (fin + culture)
feature_cols = ['gross_profitability','operating_margin','sales_growth','asset_growth',
                'debt_to_assets','log_assets','asset_turnover','roe','roa'] + culture_cols

# fill missing values
# for financial ratios used median
# for culture scores median (assuming "avg culture" if data is missing)
for col in feature_cols:
    if col in dataset.columns:
        dataset[col] = dataset[col].fillna(dataset[col].median())

# winsorize
q_lo = dataset[feature_cols].quantile(0.01)
q_hi = dataset[feature_cols].quantile(0.99)
dataset[feature_cols] = dataset[feature_cols].clip(lower=q_lo, upper=q_hi, axis=1)

#TRAIN TEST SPLIT
# cutoff at 2019 to ensure test set (2020-2021) has culture data
cutoff_year = 2019 

training = dataset[dataset['fyear'] <= cutoff_year].copy()
testing = dataset[dataset['fyear'] > cutoff_year].copy()

# don't include years > 2021 in testing (since culture data ends in 2021)
testing = testing[testing['fyear'] <= 2021] 

X_train = training[feature_cols]
y_train = training['ep']
X_test = testing[feature_cols]
y_test = testing['ep']

# scale features
train_mean = X_train.mean()
train_std = X_train.std().replace(0, 1.0)
X_train_scaled = (X_train - train_mean) / train_std
X_test_scaled = (X_test - train_mean) / train_std

# SAVE
train_out = training[['gvkey','conm','tic','fyear']].copy()
test_out = testing[['gvkey','conm','tic','fyear']].copy()

train_out[feature_cols] = X_train_scaled
test_out[feature_cols] = X_test_scaled
train_out['ep'] = y_train
test_out['ep'] = y_test

os.makedirs('./data/new_processed_w_sentiment', exist_ok=True)
train_out.to_parquet('./data/new_processed_w_sentiment/train.parquet', index=False)
test_out.to_parquet('./data/new_processed_w_sentiment/test.parquet', index=False)

print(f"Success!")
print(f"Training Data: {training['fyear'].min()} to {training['fyear'].max()} ({len(train_out)} rows)")
print(f"Testing Data:  {testing['fyear'].min()} to {testing['fyear'].max()} ({len(test_out)} rows)")
print(f"Features: {feature_cols}")