import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# loads data
train = pd.read_parquet("./data/finalParquet/train.parquet")
test = pd.read_parquet("./data/finalParquet/test.parquet")


# defines target and irrelevant columns that can mess up the model 
target = "ep"
irrelevantCols = ["gvkey", "conm", "tic", "fyear"]

featureCols = [c for c in train.columns if c not in irrelevantCols + [target]]


# splits data
X_train = train[featureCols].copy()
Y_train = train[target].copy()

X_test = test[featureCols].copy()
Y_test = test[target].copy()

# model
rf = RandomForestRegressor(n_estimators=600, max_depth=None, min_samples_split=3, min_samples_leaf=2, random_state=42, n_jobs=-1)

# training
rf.fit(X_train, Y_train)

# metrics evaluation
yPred = rf.predict(X_test)
y_Pred = rf.predict(X_train)

MAE = mean_absolute_error(Y_test, yPred)
MAET = mean_absolute_error(Y_train, y_Pred)
print(f"Mean Absolute Error on test: {MAE}")
print(f"Mean Absolute Error on train: {MAET}")

RMSE = np.sqrt(mean_squared_error(Y_test, yPred))
RMSET = np.sqrt(mean_squared_error(Y_train, y_Pred))
print(f"Root Mean Squared Error on test: {RMSE}")
print(f"Root Mean Squared Error on train: {RMSET}")

r2 = r2_score(Y_test, yPred)
r2T = r2_score(Y_train, y_Pred)
print(f"R squared on test: {r2}")
print(f"R squared on train: {r2T}")

