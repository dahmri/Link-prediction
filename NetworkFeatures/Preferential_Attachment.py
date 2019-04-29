import numpy as np

def preferential_attachment_score(graph):

    A = graph.get_adjacency();
    i_degree = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        i_degree[i] = len(graph.neighbors(i))
    PA = np.zeros(A.shape)
    for i in range(PA.shape[0]):
        for j in range(PA.shape[0]):
            PA[i,j] = i_degree[i]*i_degree[j]
    return PA
