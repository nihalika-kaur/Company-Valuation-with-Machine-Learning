import pandas as pd
import numpy as np

# Gets the Compustat CSV
dataset = pd.read_csv('./data/raw/compustat_data.csv')

## Taking string dates and converting to pandas datetime
preprocessed_dates = dataset['datadate']
dates = pd.to_datetime(preprocessed_dates)
dataset['datadate'] = dates
dataset = dataset.sort_values(['gvkey', 'datadate'])

## --- Clean numeric columns first (makes blanks become NaN) ---
numeric_cols = ['nits','epspi','csho','ceq','at','lt','revt','prcc_f']
for col in numeric_cols:
    dataset[col] = pd.to_numeric(dataset[col], errors='coerce')

# --- build nits_clean ---
# use nits if it's present
# otherwise approximate as epspi * csho (makes units consistent)
dataset['nits_clean'] = np.where(
    ~dataset['nits'].isna(),       # if we have a real nits value
    dataset['nits'],               # use it
    dataset['epspi'] * dataset['csho']  # else approximate earnings
)

## Calculates ROE and adds column
roe = np.where(dataset['ceq'] > 0, dataset['nits_clean'] / dataset['ceq'], np.nan)
dataset['roe'] = roe

## Calculates ROA and adds column
roa = np.where(dataset['at'] != 0, dataset['nits_clean'] / dataset['at'], np.nan)
dataset['roa'] = roa

## Calculates Debt to Equity and adds column
debt_to_eq = np.where(dataset['ceq'] > 0, dataset['lt'] / dataset['ceq'], np.nan)
dataset['debt_to_eq'] = debt_to_eq

## Calculate Sales Growth and adds column
dataset['sales_growth'] = dataset.groupby('gvkey')['revt'].pct_change(fill_method=None)

## Calculate E/P Ratio and adds column
ep = np.where(dataset['prcc_f'] > 0, dataset['epspi'] / dataset['prcc_f'], np.nan)
dataset['ep'] = ep

# Drop Invalid E/P Ratios
dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
dataset = dataset[~dataset['ep'].isna()].copy()

### WE HAVE 4 FEATURES + 1 LABEL ###
# LABEL (for supervised): EP #

## Reduce features down to only relevant ones
keep_cols = ['gvkey','tic','datadate','roe','roa','debt_to_eq','sales_growth','ep']
dataset = dataset[keep_cols]

## Train/Test Split
cutoff_date = pd.Timestamp('2022-12-01')
training = dataset[dataset['datadate'] <= cutoff_date].copy()
testing = dataset[dataset['datadate'] > cutoff_date].copy()

## Winsorize features
winsor_cols = ['roe', 'roa', 'debt_to_eq', 'sales_growth', 'ep']
q_lo = training[winsor_cols].quantile(0.01)
q_hi = training[winsor_cols].quantile(0.99)

training[winsor_cols] = training[winsor_cols].clip(lower=q_lo, upper=q_hi, axis=1)
testing[winsor_cols] = testing[winsor_cols].clip(lower=q_lo, upper=q_hi, axis=1)

## Compute median on training set only
feature_cols = ['roe', 'roa', 'debt_to_eq', 'sales_growth']
medians = training[feature_cols].median()

## Fill missing values in both sets
training[feature_cols] = training[feature_cols].fillna(medians)
testing[feature_cols] = testing[feature_cols].fillna(medians)

## Separating features vs. label
X_train = training[feature_cols]
y_train = training['ep']
X_test = testing[feature_cols]
y_test = testing['ep']

# Standardizing Units by Scaling
train_mean = X_train.mean()
train_std = X_train.std().replace(0, 1.0)

X_train_scaled = (X_train - train_mean) / train_std
X_test_scaled = (X_test - train_mean) / train_std

## Reattach IDs to the scaled features
train_out = training[['gvkey', 'tic', 'datadate']].copy()
test_out = testing[['gvkey', 'tic', 'datadate']].copy()

feature_cols = ['roe', 'roa', 'debt_to_eq', 'sales_growth']

train_out[feature_cols] = X_train_scaled
test_out[feature_cols] = X_test_scaled

# Add label
train_out['ep'] = y_train
test_out['ep'] = y_test

## Save to Disk
import os
os.makedirs('./data/processed', exist_ok=True)

train_out.to_parquet('./data/processed/train.parquet', index=False)
test_out.to_parquet('./data/processed/test.parquet', index=False)

print("Saved:")
print("- data/processed/train.parquet")
print("- data/processed/test.parquet")
print("Rows: ", len(train_out), len(test_out))
