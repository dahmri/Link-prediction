import networkx as netx
from itertools import combinations
import numpy as np
from igraph import Graph


def adamic_adar_score(graph):
    """
    Computing Adar-Adamic similarity matrix from a graph
    """

    N = graph.vcount()
    AA = np.zeros((N,N))
    AA = graph.similarity_inverse_log_weighted(vertices=None, mode="ALL")

    return np.array(AA)

def betweenness_centrality_score(graph):
    """
    Computing betweenness_centrality matrix from a graph
    """

    BC = np.zeros((graph.ecount(),graph.ecount()))

    A = [edge.tuple for edge in graph.es]
    G = netx.DiGraph(A)
    betweeness = netx.algorithms.edge_betweenness_centrality(G)
    for key in betweeness.keys():
        BC[key[0]][key[1]] = betweeness[key]

    return BC


def common_neighbors_score(graph):
    """
    Computing common_neighbors matrix from a graph
    """
    n = graph.vcount()
    common_neis = np.zeros((n, n))
    for v in range(graph.vcount()):
        neis = graph.neighbors(v)
        for u, w in combinations(neis, 2):
            # v is a common neighbor of u and w
            common_neis[u, w] += 1
            common_neis[w, u] += 1
    for v in range(graph.vcount()):
        for w in range(graph.vcount()):
            if v == w :
                common_neis[v][w] = 0
    return common_neis


def jaccard_coefficient_score(graph):
    """
    Computing jaccard_coefficient matrix from a graph
    """
    N = graph.vcount()
    JC = np.zeros((N,N))

    JC = graph.similarity_jaccard(vertices=None, pairs=None, mode="All")

    return np.array(JC)


def preferential_attachment_score(graph):
    """
    Computing preferential_attachment matrix from a graph
    """
    A = graph.get_adjacency()
    i_degree = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        i_degree[i] = len(graph.neighbors(i))

    PA = np.zeros(A.shape)
    for i in range(PA.shape[0]):
        for j in range(PA.shape[0]):
            PA[i,j] = i_degree[i]*i_degree[j]
    return PA

def shortest_path(graph):
    """
    Computing shortest_path matrix with dijkstra algorithm from a graph
    """
    N = graph.vcount()
    SP = np.zeros((N,N))

    SP = graph.shortest_paths_dijkstra(source=None, target=None, weights=None, mode='ALL')
    return SP

def WeightMovingAverageTimeSeriesRate(graph, date):

    """
    Computing a daily Forecast from a graph using Weight Moving Average rate
    """

    move = 3
    alpha = 0.2
    beta = 0.3
    gamma = 0.5

    dailyRate = np.zeros(graph.vcount())
    dailyForecast = np.zeros(graph.vcount())

    baseDate = 1
    diff = date - baseDate
    if diff > move:
        baseDate = baseDate + (diff-move)
        
    for v in range(graph.vcount()):
        n = 0
        new_date = baseDate + n
        while new_date <= date:
            new_graph = Graph.Read_GraphML("Data/trades-timestamped-2009-12-"+str(new_date)+".graphml")
            if v < new_graph.vcount():

                index = (new_graph.vs[v]).index
                if n == 0:
                    dailyRate[v] = dailyRate[v] + alpha * new_graph.degree(index, mode='OUT')
                if n == 1:
                    dailyRate[v] = dailyRate[v] + beta * new_graph.degree(index, mode='OUT')
                if n == 2:
                    dailyRate[v] = dailyRate[v] + gamma * new_graph.degree(index, mode='OUT')

            n += 1
            new_date = baseDate + n
        dailyForecast[v] = dailyRate[v]
    return dailyForecast
