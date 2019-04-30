import networkx as netx
from itertools import combinations
import numpy as np


def adamic_adar_score(graph):
    """
    Computing Adar-Adamic similarity matrix for an adjacency matrix
    """

    N = graph.vcount()
    AA = np.zeros((N,N))
    AA = graph.similarity_inverse_log_weighted(vertices=None, mode="ALL")

    return np.array(AA)

def betweenness_centrality_score(graph):
    """
    Computes betweenness_centrality matrix
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
    Computes common_neighbors matrix
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
    Computes jaccard_coefficient matrix
    """
    N = graph.vcount()
    JC = np.zeros((N,N))

    JC = graph.similarity_jaccard(vertices=None, pairs=None, mode="All")

    return np.array(JC)


def preferential_attachment_score(graph):
    """
    Computes preferential_attachment matrix
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
    Computes shortest_path matrix with dijkstra algorithm
    """
    N = graph.vcount()
    SP = np.zeros((N,N))

    SP = graph.shortest_paths_dijkstra(source=None, target=None, weights=None, mode='ALL')
    return SP
