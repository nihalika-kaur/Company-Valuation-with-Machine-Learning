import pandas as pd
import os

# Gets the parquets to add to instead of doing the same thing we did in new_preprocessing
train = pd.read_parquet("./data/new_processed_w_sentiment/train.parquet")
test = pd.read_parquet("./data/new_processed_w_sentiment/test.parquet")

# Gets the new dataset: AnalysisInfo which was from WRDS
consensusData = pd.read_csv("./data/raw/AnalysisInfo.csv")

#makes a new column called fyear with just the year values. Because train + test parquet are based off years only and consensus dataset contains months + year
consensusData['fyear'] = consensusData['STATPERS'].str[:4].astype(int)

# aggrevate the mean rec based on the year and tic. This gets rid of information we dont need in the data set too
# sets MeanRec to the mean over all months in the year. SO basically gets the mean of 12 different values since that was the structure of the consensus dataset
# makes the data mergeable to the train and test parquets
consensus_yearly = consensusData.groupby(['oftic', 'fyear']).agg({
    'MEANREC': 'mean',
    }).reset_index()
# merges to train + test parquets based of the year and tic 
train_merge = pd.merge(train, consensus_yearly, left_on=['tic', 'fyear'], right_on=['oftic', 'fyear'], how='left')

test_merge = pd.merge(test, consensus_yearly, left_on=['tic', 'fyear'], right_on=['oftic', 'fyear'], how='left')

# gets rid of the oftic column since we already have tic from the parquets
train_merge = train_merge.drop(columns = ['oftic'])
test_merge = test_merge.drop(columns = ['oftic'])

# gets rid of the rows where MEANREC is invalid / NA
median_meanrec = train_merge['MEANREC'].median()

train_merge['MEANREC'] = train_merge['MEANREC'].fillna(median_meanrec)
test_merge['MEANREC'] = test_merge['MEANREC'].fillna(median_meanrec)

#scales similarly 
mean_meanrec = train_merge['MEANREC'].mean()
stdMeanRec = train_merge['MEANREC'].std()
if stdMeanRec == 0:
    stdMeanRec = 1.0

train_merge['MEANREC'] = (train_merge['MEANREC'] - mean_meanrec) / stdMeanRec
test_merge['MEANREC'] = (test_merge['MEANREC'] - mean_meanrec) / stdMeanRec

#Makes the new parquets and directory for the models
os.makedirs("./data/finalParquet", exist_ok = True)

train_merge.to_parquet("./data/finalParquet/train.parquet", index = False)
test_merge.to_parquet("./data/finalParquet/test.parquet", index = False)

