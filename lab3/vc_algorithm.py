import numpy as np
graph = np.loadtxt("lab3/graph.txt", dtype=int)
nr_of_edges = int(np.sum(graph) / 2) # The assignment says approx 1000 edges butshould be half since graph is undirected

# Create solution vectors for vertex and edge variables
vertex_vec = np.zeros((graph.shape[0], 1))
edge_vec = np.zeros((nr_of_edges, 1))



# Creates dictionary containing the maximum value allowed for each edge incident on a vertex
node_to_max_edge_value = {}
for i in range(graph.shape[0]):
    node_to_max_edge_value[i] = 1/np.sum(graph[i])

# Returns the node with the smallest value for their edges.
# This is done since we always want to start with nodes with the largest amount of incident edges (since we have uniform weights)
# This is a greedy way to ensure that the "best" vertices are prioritised
def get_lowest_node():
    smallest = min(node_to_max_edge_value.values())
    for vertex, val in node_to_max_edge_value.items():
        if val == smallest:
            return vertex, val

# Convert vertex-vertex matrix to vertex-edge incidence matrix
node_edge_incidence = np.zeros((nr_of_edges, graph.shape[0]))
edge_index = 0
for i in range(graph.shape[0]):
    for j in range(i+1, graph.shape[0]):
        if graph[i, j] == 1:
            node_edge_incidence[edge_index, i] = 1
            node_edge_incidence[edge_index, j] = 1
            edge_index += 1

# Returns incident edges to a vertex
def get_incident_edges(vertex):
    return np.where(node_edge_incidence[:, vertex] == 1)[0]


# Returns neighbour vertex given source and edge
def get_incident_node(node, edge):
    incident = np.where(node_edge_incidence[edge, :] == 1)[0]
    if incident[0] == node:
        return incident[1]
    else:
        return incident[0]

frozen = np.array([])
# Runs until all edges have been frozen
while len(frozen) != nr_of_edges:
    vertex, val = get_lowest_node()
    incident_edges = get_incident_edges(vertex)

    # Set-Difference operation to remove frozen edges from pool
    free_edges = np.setdiff1d(get_incident_edges(vertex), frozen) 

    # If there are no free edges, all available edges from this vertex are already included in the solution and so 
    # it is reduntand to check it
    if len(free_edges) == 0:
        node_to_max_edge_value.pop(vertex)   
        continue

    for edge in free_edges:
        # Set corresponding edge in solution vector to maximum val
        edge_vec[edge] = val

        # Computes total of incident edges to the edge to check if we have reached the total maximum of 1
        # If total is 1, freeze all incident edges, set the corresponding vertex solution variable to 1 and remove it from the dictionary.
        current_sum = np.matmul(node_edge_incidence.T, edge_vec)[vertex][0]
        if round(current_sum, 8) == 1: 
            vertex_vec[vertex] = 1
            frozen = np.append(frozen, free_edges)
            node_to_max_edge_value.pop(vertex)

            # For the current vertex, update the values of its neighbouring vertices to reallocate the maximum for their 
            # remaining unfrozen edges.
            for edge in free_edges:
                node = get_incident_node(vertex, edge)
                edges_left = get_incident_edges(node)
                diff = len(np.setdiff1d(edges_left, frozen))
                node_to_max_edge_value[node] = (1 - sum([edge_vec[edge] for edge in edges_left if edge in frozen]))/diff

            break

print(len(frozen) == nr_of_edges)
print(np.sum(vertex_vec))
print(np.sum(edge_vec))

# Confirm that solution is indeed a VC
vc = True
for edge in node_edge_incidence:
    nodes = np.where(edge == 1)
    if not (vertex_vec[nodes[0][0]] == 1 or vertex_vec[nodes[0][1]] == 1):
        vc = False
        break
print(f"The solution produces valid VC: {vc}")
    

