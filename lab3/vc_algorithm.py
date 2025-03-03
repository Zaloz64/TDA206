import numpy as np

graph = np.loadtxt("graph.txt", dtype=int)

node_vec = np.zeros((graph.shape[0], 1))
nr_of_edges = np.sum(graph) / 2
edge_vec = np.zeros((int(nr_of_edges), 1))
