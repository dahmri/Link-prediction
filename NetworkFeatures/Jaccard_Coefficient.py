import numpy as np

def jaccard_coefficient_score(graph):
    """
    Computes Adar-Adamic similarity matrix for an adjacency matrix
    """

    N = graph.vcount()
    JC = np.zeros((N,N))

    JC = graph.similarity_jaccard(vertices=None, pairs=None, mode="All")

    return np.array(JC)
