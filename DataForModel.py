import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split



min_max_scaler = preprocessing.MinMaxScaler()

Train_scaled = min_max_scaler.fit_transform(pd.read_csv("Data/TrainDataSet_Trades_Network.txt",sep=" "))
TrainDataFrame =pd.DataFrame(Train_scaled)
TrainDataFrame.to_csv("DataFTrainTest/dtrain_Trades_Network.csv", sep=" ",index=False, header=False)


Test_scaled = min_max_scaler.fit_transform(pd.read_csv("Data/TestingDataSet_Trades_Network.txt",sep=" "))
TestDataFrame =pd.DataFrame(Test_scaled)
DataDFS.to_csv("DataFTrainTest/dtest_Trades_Network.csv", sep=" ",index=False, header=False)

