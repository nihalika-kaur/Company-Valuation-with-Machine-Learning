import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the training and testing data
## This model is using the processed data from the finalProcess.py file
train_data = pd.read_parquet('./data/finalParquet/train.parquet')
test_data = pd.read_parquet('./data/finalParquet/test.parquet')

# Define the feats and target
# Features should include financial ratios, culture scores, and consensus data
# features = [
#     'gross_profitability', 'operating_margin', 'sales_growth', 'asset_growth',
#     'debt_to_assets', 'log_assets', 'asset_turnover', 'roe', 'roa',
#     'innovation', 'integrity', 'quality', 'respect', 'teamwork',
#     'MEANREC'
# ]
# target = 'ep'
features = [
    'gross_profitability', 'operating_margin', 'sales_growth', 'asset_growth',
    'debt_to_assets', 'log_assets', 'asset_turnover', 'roe', 'roa',
    'innovation', 'integrity', 'quality', 'respect', 'teamwork',
    'MEANREC'
]
target = 'ep'

X_train = train_data[features]
Y_train = train_data[target]
X_test = test_data[features]
Y_test = test_data[target]

# TODO: Initialize the GradientBoostingRegressor
# You can tune hyperparameters like n_estimators, learning_rate, max_depth, etc.
SEED = 42
gb = GradientBoostingRegressor(
    loss='absolute_error',
    n_estimators=300,
    learning_rate=0.1,
    max_depth=3,
    random_state=SEED
)

# Train the model
gb.fit(X_train, Y_train)

# Make predictions
predictions_train = gb.predict(X_train)
predictions_test = gb.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(Y_train, predictions_train)
rmse = np.sqrt(mean_squared_error(Y_train, predictions_train))
r2 = r2_score(Y_train, predictions_train)

print(f"MAE Train: {mae}")
print(f"RMSE Train: {rmse}")
print(f"R^2 Train: {r2}")

mae = mean_absolute_error(Y_test, predictions_test)
rmse = np.sqrt(mean_squared_error(Y_test, predictions_test))
r2 = r2_score(Y_test, predictions_test)

print(f"MAE Test: {mae}")
print(f"RMSE Test: {rmse}")
print(f"R^2 Test: {r2}")