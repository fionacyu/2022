import sys
import load_data
import graph_characterisation

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

def get_pi_elec(conjNodeList, conjEdgeList, graph):
    tupleList = []
    for i, n in enumerate([x for x in conjNodeList]):
            # print(n, graph.nodes[n]['element'])
            valence = load_data.get_valence(graph.nodes[n]['element'])
            # print('valence', valence)
            # sigmaBonds = len([x for x in graph.neighbors(n)]) # number of sigma bonds
            sigmaBonds = graph.degree[n]
            # print('sigmaBonds', sigmaBonds)
            elecDom = graph.nodes[n]['ed']
            # print('elecDom', elecDom)

            piELec = valence - sigmaBonds - 2 * (elecDom - sigmaBonds)- graph.nodes[n]['charge'] # gives the number of pi electrons in the conjugated system, formula is essentially FC = V - N - B/2           
            if i == 0 or i == len([x for x in conjNodeList]) - 1:
                # need to check for the fact that the atom is connected to other pi systems not part of the conjugated system or separate conjugated systems
                edgeList = graph_characterisation.get_edges_of_node(n, [x for x in graph.edges if graph[x[0]][x[1]]['bo'] >= 2]) #edges the node is part of which is double bond or triple
                # edgeList = edgeList + [x[::-1] for x in edgeList]
                # print('edgeList', edgeList)
                # reject_edges = [x for x in edgeList if x in conjugated_edges[j] or x[::-1] in conjugated_edges[j]] 
                # reject_edges = list(set(edgeList).intersection(conjugated_edges[j])) + list(set([x[::-1] for x in edgeList]).intersection(conjugated_edges[j]))
                reject_edges = list( (set(edgeList).intersection(conjEdgeList)).union((set([x[::-1] for x in edgeList]).intersection(conjEdgeList)) ))
                # print('reject_edges', reject_edges)
                # edgeList = [x for x in edgeList if x not in reject_edges] # non conjugated edges the node is bonded to
                # edgeList = list(set(edgeList) - set(reject_edges))
                # edgeList = [x for x in edgeList if x not in reject_edges and x[::-1] not in reject_edges]
                edgeList = [x for x in edgeList if len(set([x]).intersection(reject_edges)) == 0 and  len(set([x[::-1]]).intersection(reject_edges)) == 0]
                # print('edgeList after removal', edgeList)

                bo_diff_list = [graph[x[0]][x[1]]['bo'] - 1 for x in edgeList] # minus 1 because one bond will be sigma bond (we want the pi bond)
                # print('bo_diff_list', bo_diff_list)
                non_conj_pi_elec = sum(bo_diff_list)
                # print('non_conj_pi_elec', non_conj_pi_elec)

                # graph.nodes[n]['pi'] = piELec - non_conj_pi_elec
                tupleList.append((n, piELec - non_conj_pi_elec))

            else:
                # graph.nodes[n]['pi'] = piELec # tells us the number of pi electrons per atom, but doesn't distinguish between conjugated vs. non-conjugated systems
                tupleList.append((n, piELec))
    
    return tupleList
    