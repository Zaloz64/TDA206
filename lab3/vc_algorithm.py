import numpy as np

graph = np.loadtxt("lab3/graph.txt", dtype=int)

node_vec = np.zeros((graph.shape[0], 1))
nr_of_edges = int(np.sum(graph) / 2) # For numpy conversion, exact number is 490 since graph is undirected
node_to_max_edge_value = {}
for i in range(graph.shape[0]):
    node_to_max_edge_value[i] = 1/np.sum(graph[i])

edge_vec = np.zeros((nr_of_edges, 1))

def get_lowest_node():
    for vertex, val in node_to_max_edge_value.items():
        if val == min(node_to_max_edge_value.values()):
            return vertex, val

# Convert to node-edge incidence matrix
node_edge_incidence = np.zeros((nr_of_edges, graph.shape[0]))
edge_index = 0
for i in range(graph.shape[0]):
    for j in range(i+1, graph.shape[0]):
        if graph[i, j] == 1:
            node_edge_incidence[edge_index, i] = 1
            node_edge_incidence[edge_index, j] = 1
            edge_index += 1


def get_incident_edges(vertex):
    return np.where(node_edge_incidence[:, vertex] == 1)[0]

def get_incident_node(node, edge):
    incident = np.where(node_edge_incidence[edge, :] == 1)[0]
    if incident[0] == node:
        return incident[1]
    else:
        return incident[0]

frozen = np.array([])
for i in range(node_edge_incidence.shape[0]):
    vertex, val = get_lowest_node()
    incident_edges = get_incident_edges(vertex)
    for edge in incident_edges:
        if edge not in frozen:
            edge_vec[edge] = val
        current_sum = np.matmul(node_edge_incidence.T, edge_vec)[vertex][0]
        if  round(current_sum, 8) == 1:
            node_vec[vertex] = 1
            frozen = np.append(frozen, incident_edges)
            node_to_max_edge_value.pop(vertex)
            for edge in incident_edges:
                node = get_incident_node(vertex, edge)
                node_to_max_edge_value[node] = (1 - val )/len(np.setdiff1d(get_incident_edges(node), frozen))
            break

print(len(frozen))
print(np.sum(node_vec))