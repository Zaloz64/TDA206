import numpy as np

graph = np.loadtxt("lab3/graph.txt", dtype=int)

node_vec = np.zeros((graph.shape[0], 1))
nr_of_edges = np.sum(graph) / 2
node_to_max_edge_value = [(i, 1/np.sum(graph[i])) for i in range(graph.shape[0])]
node_to_max_edge_value.sort(key=lambda x: x[1], reverse=True)

# Convert to node-edge incidence matrix
node_edge_incidence = np.zeros((int(nr_of_edges), graph.shape[0]))
edge_index = 0
for i in range(graph.shape[0]):
    for j in range(i+1, graph.shape[0]):
        if graph[i, j] == 1:
            node_edge_incidence[edge_index, i] = 1
            node_edge_incidence[edge_index, j] = 1
            edge_index += 1


frozen = []
for i, val in node_to_max_edge_value:
    