from igraph import Graph
import NetworkFeatures
from sklearn import preprocessing
from numpy import *
import pandas as pd



def FeatureMatrix(graph, date):
    """
    Creating Features from a grpah :
    adamic_adar, jaccard_coefficient, common_neighbors, preferential_attachment, shortest_path, betweenness_centrality and TimeSeriesRate (Weight Moving Average)
    """

    AAMatrix = NetworkFeatures.adamic_adar_score(graph)
    JCMatrix = NetworkFeatures.jaccard_coefficient_score(graph)
    CNMatrix = NetworkFeatures.common_neighbors_score(graph)
    PAMatrix = NetworkFeatures.preferential_attachment_score(graph)
    SPMatrix = NetworkFeatures.shortest_path(graph)
    betweenness = NetworkFeatures.betweenness_centrality_score(graph)
    dailyRate = NetworkFeatures.WeightMovingAverageTimeSeriesRate(graph, date)

    return AAMatrix, JCMatrix, CNMatrix, PAMatrix, SPMatrix, betweenness, dailyRate

def FeatureVector(nodei, nodej, AAMatrix, JCMatrix, CNMatrix, PAMatrix, SPMatrix, betweenness, dailyRate):
    """
    Creating a Feature Vector for a pair of nodes : nodei and nodej
    """

    feature_vector = ""

    feature_vector += str(AAMatrix[nodei][nodej]) + " "
    feature_vector += str(JCMatrix[nodei][nodej]) + " "
    feature_vector += str(CNMatrix[nodei][nodej]) + " "
    feature_vector += str(PAMatrix[nodei][nodej]) + " "
    feature_vector += str(SPMatrix[nodei][nodej]) + " "
    feature_vector += str(betweenness[nodei][nodej]) + " "
    feature_vector += str(dailyRate[nodei]) + " "

    return feature_vector



if __name__ == '__main__':

    #Tranforming Graph to features
    for date in range(1, 31):
    
        DataGraph = Graph.Read_GraphML("Data/trades-timestamped-2009-12-"+str(date)+".graphml")
        print("trades-timestamped-2009-12-"+str(date)+".graphml")
    
        feature_vector = ""
        AAMatrix, JCMatrix, CNMatrix, PAMatrix, SPMatrix, betweenness, dailyRate = FeatureMatrix(DataGraph, date)
        AA = DataGraph.get_adjacency()
        if(date <= 24):
            DataSetFile = open("./TrainTest/TrainDataSet_Trades_Network.txt", "a")
            print("TrainDataSet_Trades_Network.txt for day", date)
        else:
            DataSetFile = open("./TrainTest/TestingDataSet_Trades_Network.txt", "a")
            print("TestingDataSet_Trades_Network.txt for day", date)
    
    
        for i in range(DataGraph.vcount()) :
            for j in range(DataGraph.vcount()):
                if (i!=j):
                    feature_vector = FeatureVector(i, j, AAMatrix, JCMatrix, CNMatrix, PAMatrix, SPMatrix, betweenness, dailyRate)
                    if AA[i][j] == 0:
                        DataSetFile.write(feature_vector+"0\n")
                    else:
                        DataSetFile.write(feature_vector+"1\n")
        DataSetFile.close()

    #Normalizing the data
    min_max_scaler = preprocessing.MinMaxScaler()

    Train_scaled = min_max_scaler.fit_transform(pd.read_csv("./TrainTest/TrainDataSet_Trades_Network.txt", sep=" "))
    TrainDataFrame = pd.DataFrame(Train_scaled)
    TrainDataFrame.to_csv("./TrainTest/dtrain_Trades_Network.csv", sep=" ", index=False, header=False)

    Test_scaled = min_max_scaler.fit_transform(pd.read_csv("./TrainTest/TestingDataSet_Trades_Network.txt", sep=" "))
    TestDataFrame = pd.DataFrame(Test_scaled)
    TestDataFrame.to_csv("./TrainTest/dtest_Trades_Network.csv", sep=" ", index=False, header=False)
