import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

## This model is using the processed data from the preprocessing.py file

#get data parquets; already done
train_data = pd.read_parquet('../../data/processed/train.parquet')
test_data = pd.read_parquet('../../data/processed/test.parquet')

#defining the features & target
features = ['roe', 'roa', 'debt_to_eq', 'sales_growth']
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


print(f"MAE on train: {MAETrain}")
print(f"MAE on test: {MAETest}")

#RMSE on data
RMSETrain = np.sqrt(mean_squared_error(Y_train, PredictionOnTrain))
RMSETest = np.sqrt(mean_squared_error(Y_test, PredictionOnTest))

print(f"RMSE on train: {RMSETrain}")
print(f"RMSE on test: {RMSETest}")

#R^2 on data

RTrain = r2_score(Y_train, PredictionOnTrain)
RTest = r2_score(Y_test, PredictionOnTest)

print(f"R^2 on train: {RTrain}")
print(f"R^2 on test: {RTest}")




