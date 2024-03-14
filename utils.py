import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score


def create_filters(n_layers, n_timestamps):
    filter = []
    filter.append(n_timestamps)
    for i in range(n_layers):
        filter.append(2 ** (n_layers + 1 - i))
    return filter

def correlations(data):
    corr = np.array(data.T.corr())
    np.fill_diagonal(corr, 0)  # Set the diagonal entries to 0 (exclude self-correlations)

    corr = torch.tensor(corr, dtype=torch.float)
    return corr

def adj_to_edge(adj, thres2):   # Convert the adjacency matrix to an edge list
    edges = []
    weights = []
    thres1 = 0.99
    weight1 = 1
    weight2 = 0.8

    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i][j] > thres1:
                weights.append(weight1)
                edges.append((i, j))
            elif adj[i][j] > thres2:
                weights.append(adj[i][j] - weight2)
                edges.append((i, j))

    edges = torch.tensor(edges, dtype=torch.long).t()
    weights = torch.tensor(weights, dtype=torch.float)
    return edges, weights

def nearest_neighbors(data, r):
    nn_algo = NearestNeighbors(radius=r).fit(data)
    dist, neighbors = nn_algo.radius_neighbors(data)
    edges = []

    for i, row in enumerate(neighbors):
        source_node = i
        for target_node in row[0:]:
            edges.append((source_node, target_node))

    edges = torch.tensor(edges, dtype=torch.long).t()
    return edges

def clustering(data):
    ap = AffinityPropagation().fit(data)
    clusters = ap.predict(data)
    return clusters

def evaluation(data, pred):
    return silhouette_score(data, pred)

def clusters(pred):
    # Count the number of samples in each cluster
    unique_clusters, cluster_counts = np.unique(pred, return_counts=True)
    res = []

    # Iterate through each cluster
    for cluster in np.unique(pred):
        # Find indices of samples in the current cluster
        genes_per_cluster = np.where(pred == cluster)[0]
        res.extend([(cluster, gene) for gene in genes_per_cluster])
    return res