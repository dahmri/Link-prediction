import igraph 
import networkx as nx
from Graph_Operations import *
import Adamic_Adar
import Jaccard_Coefficient
import Common_Neighbors
import  Preferential_Attachment
import Shortest_Path
import Betweenness_Centrality
from sklearn import preprocessing
from numpy import *
import random
from random import randint


def CreateFeatureMatrix(graph,date):

    # Calculate Adamic-Adar matrix
    AAMatrix = []
    AAMatrix = Adamic_Adar.adamic_adar_score(graph)

    # Calculate Jaccard Coefficient matrix
    JCMatrix = []
    CMatrix = Jaccard_Coefficient.jaccard_coefficient_score(graph)

    # Calculate Common Neighbors matrix
    CNMatrix = []
    CNMatrix = Common_Neighbors.common_neighbors_score(graph)

    # Calculate Preferential Attachment matrix
    PAMatrix = []
    PAMatrix = Preferential_Attachment.preferential_attachment_score(graph)

    # Calculate Shortest Path matrix
    SPMatrix = []
    SPMatrix = Shortest_Path.shortest_path(graph)

    # Calculate Betweenness Centrality
    betweenness = []
    betweenness = Betweenness_Centrality.betweenness_centrality_score(graph)

    # Time Series Daily Rate Forecasting
    dailyRate = []
    dailyRate= WeightMovingAverageTimeSeriesRate(graph,date)

    return AAMatrix,JCMatrix,CNMatrix,PAMatrix,SPMatrix,betweenness,dailyRate

def CreateFeatureVector(node1,node2,AAMatrix,JCMatrix,CNMatrix,PAMatrix,SPMatrix,betweenness,dailyRate):

    feature_vector = ""

    feature_vector += str(AAMatrix[node1][node2]) + " " 
    feature_vector += str(JCMatrix[node1][node2])+ " "
    feature_vector += str(CNMatrix[node1][node2])+ " "
    feature_vector += str(PAMatrix[node1][node2]) + " "
    feature_vector += str(SPMatrix[node1][node2]) + " "
    feature_vector += str(betweenness[(node1,node2)]) + " "
    feature_vector += str(dailyRate[node1]) + " "

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
            new_graph = Graph.Read_GraphML("Trades-Network/trades-timestamped-2009-12-"+str(new_date)+".graphml")
            try:
                index = (new_graph.vs[v]).index
                if n == 0:
                    dailyRate[v] = dailyRate[v] + alpha * new_graph.degree(index,mode='OUT')
                if n == 1:
                    dailyRate[v] = dailyRate[v] + beta* new_graph.degree(index,mode='OUT')
                if n == 2:
                    dailyRate[v] = dailyRate[v] + gamma* new_graph.degree(index,mode='OUT')
            except ValueError:
                dailyRate[v] = dailyRate[v] + 0
            n += 1
            new_date = int(baseDate) + n
        dailyForecast[v] = dailyRate[v] / n
    return dailyForecast


for date in range (30):

    DataGraph=Graph.Read_GraphML("Trades-Network/trades-timestamped-2009-12-"+str(date)+".graphml")
    print "trades-timestamped-2009-12-"+str(date)+".graphml"

    feature_vector = ""
    AAMatrix,JCMatrix,CNMatrix,PAMatrix,SPMatrix,betweenness,dailyRate=CreateFeatureMatrix(DataGraph,date)
    AA = DataGraph.get_adjacency()

    if(date<=24):
        DataSetFile = open("Data/TrainDataSet_Trades_Network.txt","a")
        print "TrainDataSet_Trades_Network.txt for day",date
    else:
        DataSetFile = open("Data/TestingDataSet_Trades_Network.txt","a")
        print "TestingDataSet_Trades_Network.txt for day",date


    for i in range(DataGraph.vcount()):
        for j in range(DataGraph.vcount()):
            if (i!=j):
                feature_vector = CreateFeatureVector(i,j,AAMatrix,JCMatrix,CNMatrix,PAMatrix,SPMatrix,betweenness,dailyRate)
                if AA[i][j] == 0:
                    DataSetFile.write(feature_vector+"0\n")
                else:
                    DataSetFile.write(feature_vector+"1\n")
    DataSetFile.close()



