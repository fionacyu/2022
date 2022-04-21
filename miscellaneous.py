import sys
import torch

def flatten(t):
    return [item for sublist in t for item in sublist]

def find_conjugated_system(conjugated_edges, edge): # returns a list of lists of the edges of a conjugated system that the input edge is part of
    edges_lol = [x for x in conjugated_edges if edge in x or edge[::-1] in x]
    if len(edges_lol) == 1:
        return edges_lol[0]
    else:
        print('conjugated_edges list of lists is not mutually exclusive')
        sys.exit()

def node_of_element(graph, edge, element): #returns the node of the edge which == element
    blist = [element == graph.nodes[x]['element'] for x in edge]
    return edge[blist.index(True)]

def donor_acceptor_of_edge(dictionary, edge):
    daList = [name for name, _ in dictionary.items() if edge in dictionary[name].edges]

    return daList

def donor_acceptor_nodes(da_object, nodeList): # tells which nodes belong to the donor or acceptor
    da_nodes = [x for x in nodeList if x in da_object.nodes]
    return da_nodes

def index_of_cycle_list(cycle_list, edge):
    # print('cycle_list', cycle_list)
    blist = [edge in x for x in cycle_list]
    ind_list = [i for i, x in enumerate(blist) if x == True]
    return ind_list

# def gen_all_binary_vectors(length: int) -> torch.Tensor:
#     return ((torch.arange(2**length).unsqueeze(1) >> torch.arange(length-1, -1, -1)) & 1).float()
def shortest_path_length(graph, node1, node2):
    # BFS method, performs in linear time
    path_list = [[node1]]
    path_index = 0
    # To keep track of previously visited nodes
    previous_nodes = {node1}
    if node1 == node2:
        return (node1, 0)#path_list[0]

    dist = 1

    while path_index < len(path_list):
        current_path = path_list[path_index]
        last_node = current_path[-1]
        next_nodes = graph[last_node]
        # Search goal node
        if node2 in next_nodes:
            current_path.append(node2)
            # return current_path # 1?
            return (node1, dist)
        # Add new paths
        for next_node in next_nodes:
            if not next_node in previous_nodes:
                new_path = current_path[:]
                new_path.append(next_node)
                path_list.append(new_path)
                # To avoid backtracking
                previous_nodes.add(next_node)
                dist += 1
        # Continue to next path in list
        path_index += 1
    # No path is found
    return (node1, 0)