import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

## This model is using the processed data from the preprocessing.py file

#get data parquets; already done
train_data = pd.read_parquet('./data/finalParquet_leakfree/train.parquet')
test_data = pd.read_parquet('./data/finalParquet_leakfree/test.parquet')

#defining the features & target
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

#defining model
LRModel = LinearRegression()

#training
LRModel.fit(X_train, Y_train)

#checking predictions to evalute accuracy for both the train and test datasets
PredictionOnTrain = LRModel.predict(X_train)
PredictionOnTest = LRModel.predict(X_test)

#MAE on data
MAETrain = mean_absolute_error(Y_train, PredictionOnTrain)
MAETest = mean_absolute_error(Y_test, PredictionOnTest)


print(f"MAE on train: {MAETrain:.3f}")
print(f"MAE on test: {MAETest:.3f}")


#RMSE on data
RMSETrain = np.sqrt(mean_squared_error(Y_train, PredictionOnTrain))
RMSETest = np.sqrt(mean_squared_error(Y_test, PredictionOnTest))

print(f"RMSE on train: {RMSETrain:.3f}")
print(f"RMSE on test: {RMSETest:.3f}")

#R^2 on data

RTrain = r2_score(Y_train, PredictionOnTrain)
RTest = r2_score(Y_test, PredictionOnTest)

print(f"R^2 on train: {RTrain:.3f}")
print(f"R^2 on test: {RTest:.3f}")




