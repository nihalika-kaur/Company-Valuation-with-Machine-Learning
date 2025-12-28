import pandas as pd

# This file is used to view the parquet files
print("Old Data:")
df_train = pd.read_parquet('../data/processed/train.parquet')
df_test  = pd.read_parquet('../data/processed/test.parquet')

print("TRAIN:")
print(df_train.head())
print("\nshape:", df_train.shape)

print("\n\nTEST:")
print(df_test.head())
print("\nshape:", df_test.shape)

print("---------------------------------------------------------------")

print("Updated data:")

# force all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000) # wider terminal output

# load data
df = pd.read_parquet('../data/finalParquet/train.parquet')

# print the column names to make sure everything is there
print(f"-----All {len(df.columns)} Columns-----")
print(df.columns.tolist())

# first 5 rows with all data
print("\n-----First 5 Rows (Full Data)-----")
print(df.head())