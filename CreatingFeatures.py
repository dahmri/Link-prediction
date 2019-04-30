import igraph
from igraph import Graph
import networkx as nx
import NetworkFeatures
from sklearn import preprocessing
from numpy import *
import pandas as pd
import random
from random import randint
from sklearn.model_selection import train_test_split


def FeatureMatrix(graph, date):

    AAMatrix = []
    AAMatrix = NetworkFeatures.adamic_adar_score(graph)

    JCMatrix = []
    JCMatrix = NetworkFeatures.jaccard_coefficient_score(graph)

    CNMatrix = []
    CNMatrix = NetworkFeatures.common_neighbors_score(graph)

    PAMatrix = []
    PAMatrix = NetworkFeatures.preferential_attachment_score(graph)

    SPMatrix = []
    SPMatrix = NetworkFeatures.shortest_path(graph)

    betweenness = []
    betweenness = NetworkFeatures.betweenness_centrality_score(graph)

    dailyRate = []
    dailyRate = WeightMovingAverageTimeSeriesRate(graph, date)


    return AAMatrix, JCMatrix, CNMatrix, PAMatrix, SPMatrix, betweenness, dailyRate

def FeatureVector(nodei, nodej, AAMatrix, JCMatrix, CNMatrix, PAMatrix, SPMatrix, betweenness, dailyRate):

    feature_vector = ""

    feature_vector += str(AAMatrix[nodei][nodej]) + " "
    feature_vector += str(JCMatrix[nodei][nodej]) + " "
    feature_vector += str(CNMatrix[nodei][nodej]) + " "
    feature_vector += str(PAMatrix[nodei][nodej]) + " "
    feature_vector += str(SPMatrix[nodei][nodej]) + " "
    feature_vector += str(betweenness[(nodei, nodej)]) + " "
    feature_vector += str(dailyRate[nodei]) + " "

    return feature_vector

def WeightMovingAverageTimeSeriesRate(graph,date):

    move = 3
    alpha = 0.2
    beta = 0.3
    gamma = 0.5

    dailyRate = zeros(graph.vcount())
    dailyForecast = zeros(graph.vcount())

    baseDate = "1"
    diff = int(date) - int(baseDate)
    if diff > move:
        baseDate = int(baseDate) + (diff-move-1)
        baseDate = str(baseDate)

    for v in range(graph.vcount()):
        n = 0
        new_date = int(baseDate) + n
        while new_date <= int(date):
            new_graph = Graph.Read_GraphML("Data/trades-timestamped-2009-12-"+str(new_date)+".graphml")
            if (v < new_graph.vcount()):

                index = (new_graph.vs[v]).index
                if n == 0:
                    dailyRate[v] = dailyRate[v] + alpha * new_graph.degree(index, mode='OUT')
                if n == 1:
                    dailyRate[v] = dailyRate[v] + beta * new_graph.degree(index, mode='OUT')
                if n == 2:
                    dailyRate[v] = dailyRate[v] + gamma * new_graph.degree(index, mode='OUT')

            else:
                dailyRate[v] = dailyRate[v] + 0

            n += 1
            new_date = int(baseDate) + n
        dailyForecast[v] = dailyRate[v] / n
    return dailyForecast


for date in range(5, 30):

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



min_max_scaler = preprocessing.MinMaxScaler()

Train_scaled = min_max_scaler.fit_transform(pd.read_csv("./Data/TrainDataSet_Trades_Network.txt", sep=" "))
TrainDataFrame = pd.DataFrame(Train_scaled)
TrainDataFrame.to_csv("./Data/dtrain_Trades_Network.csv", sep=" ", index=False, header=False)


Test_scaled = min_max_scaler.fit_transform(pd.read_csv("./Data/TestingDataSet_Trades_Network.txt", sep=" "))
TestDataFrame = pd.DataFrame(Test_scaled)
TestDataFrame.to_csv("./Data/dtest_Trades_Network.csv", sep=" ", index=False, header=False)
