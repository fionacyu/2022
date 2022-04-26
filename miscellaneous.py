import sys
import load_data
import graph_characterisation
from itertools import chain
import networkx as nx


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
    
def hyperconj_penalty_connection(graph, connection, connectionDict, donorDict, acceptorDict, edges_to_cut_list):
    donor = connection[0]
    acceptor = connection[1]

    boxLabelList = donorDict[donor].boxLabelList + acceptorDict[acceptor].boxLabelList
    # edges in the same box or neighbouring boxes as the donors/acceptors
    edgesBoxes = [e for e in edges_to_cut_list if len(set([graph.nodes[e[0]]['box']]).intersection(boxLabelList)) > 0 or len(set([graph.nodes[e[1]]['box']]).intersection(boxLabelList)) > 0 ]
    influential_edges = [e for e in edgesBoxes if (e in connectionDict[connection].simple_paths or e in donorDict[donor].edges or e in acceptorDict[acceptor].edges) or (e[::-1] in connectionDict[connection].simple_paths or e[::-1] in donorDict[donor].edges or e[::-1] in acceptorDict[acceptor].edges)]
    #these are the edges that will impact the donor-acceptor pair
    if influential_edges:
        # print(connection)

        da_graph = nx.Graph()
        nodeList = donorDict[donor].nodes + acceptorDict[acceptor].nodes + [x for x in set(chain(*connectionDict[connection].simple_paths))]
        nodeList = list(dict.fromkeys(nodeList)) # remove duplicates that arise from connection edges which comprise terminal nodes of donors and acceptors
        edgeList = donorDict[donor].edges + acceptorDict[acceptor].edges + connectionDict[connection].simple_paths
        # print('edgeList: ', edgeList)
        rejected_edges = [e for e in edgeList if e in influential_edges or e[::-1] in influential_edges]
        # edgeList = [e for e in edgeList if e not in rejected_edges] # remove influential edges/ edges to cut
        edgeList = list(set(edgeList) - set(rejected_edges))
        da_graph.add_nodes_from(nodeList)
        da_graph.add_edges_from(edgeList)

        connected_comp_list = [x for x in nx.connected_components(da_graph)] # gives list of nodes which are connected to each other
        # print('connected_comp_list', connected_comp_list)

        dsList, asList = [], []
        for cc in connected_comp_list:
            cc_nodes = [x for x in cc]
            dnodes = donor_acceptor_nodes(donorDict[donor], cc_nodes)
            anodes = donor_acceptor_nodes(acceptorDict[acceptor], cc_nodes)

            # print('dnodes', dnodes)
            # print('anodes', anodes)

            if dnodes:
                donor_electrons = sum([donorDict[donor].node_electrons[x] for x in dnodes])
                da_node_number = len(dnodes) + len(anodes)
                # print('ds', donor_electrons/da_node_number)
                dsList.append(donor_electrons/da_node_number)
            
            if anodes:
                donor_electrons = sum([donorDict[donor].node_electrons[x] for x in dnodes])
                da_node_number = len(dnodes) + len(anodes)
                # print('as', -1 * donor_electrons/da_node_number)
                asList.append(-1 * donor_electrons/da_node_number)
        

        connection_penalty = 1/connectionDict[connection].bond_separation * ((sum(dsList)/len(dsList)) + sum(asList)/len(asList))
        # print('connection_penalty', connection_penalty)
        return connection_penalty
    else:
        return 0 

def aromaticity_penalty_para(graph, asys, aromaticDict, edges_to_cut_list):
    boxLabelList = aromaticDict[asys].boxLabelList
    edgesBoxes = [e for e in edges_to_cut_list if len(set([graph.nodes[e[0]]['box']]).intersection(boxLabelList)) > 0 or len(set([graph.nodes[e[1]]['box']]).intersection(boxLabelList)) > 0 ]

    influential_edges = [e for e in edgesBoxes if e in flatten(aromaticDict[asys].cycle_list) and e not in aromaticDict[asys].bridging_edges] # doesn't include bridging edges
    bedgeList = [e for e in edgesBoxes if e in flatten(aromaticDict[asys].cycle_list) and e in aromaticDict[asys].bridging_edges] # bridging edges only

    nonbe_cycle_ind_list = flatten([index_of_cycle_list(aromaticDict[asys].cycle_list, edge) for edge in influential_edges])
    nonbe_cycle_ind_list = list(dict.fromkeys(nonbe_cycle_ind_list)) # get the unique values
    be_cycle_ind_list = flatten([index_of_cycle_list(aromaticDict[asys].cycle_list, edge) for edge in bedgeList])
    be_cycle_ind_list = list(dict.fromkeys(be_cycle_ind_list))
    # nonbe_cycle_ind_list = [x for x in nonbe_cycle_ind_list if x not in be_cycle_ind_list]
    nonbe_cycle_ind_list = list(set(nonbe_cycle_ind_list) - set(be_cycle_ind_list))

    # inc_penalty = aromaticDict[asys].size - (sum([len(aromaticDict[asys].nonbridging_edges[x]) for x in nonbe_cycle_ind_list]) + sum([len(aromaticDict[asys].nonbridging_edges[x]) for x in be_cycle_ind_list]) + len(bedgeList))
    inc_penalty = sum([len(aromaticDict[asys].nonbridging_edges[x]) for x in nonbe_cycle_ind_list]) + sum([len(aromaticDict[asys].nonbridging_edges[x]) for x in be_cycle_ind_list]) + len(bedgeList)
    return inc_penalty

# def conj_penalty_para(graph, system, edges_of_interest):
#     nodeList = set(chain(*system))
#     cs_list = [graph.nodes[x]['pi']/len(nodeList) for x in nodeList]
#     average_cs = sum(cs_list) / len(cs_list)
#     # system_cs_list.append(average_cs)

#     # getting the updated cs after breaking edges
#     edges_to_remove = [x for x in system if x in edges_of_interest or x[::-1] in edges_of_interest]
#     # subsystem_edge_list = [x for x in system if x not in edges_to_remove]
#     subsystem_edge_list = list(set(system) - set(edges_to_remove))
#     # print('subsystem_edge_list', subsystem_edge_list)
#     subsystem_node_list = set(chain(*system))
#     # print(subsystem_node_list)

#     # constructing subgraph
#     sg = nx.Graph()
#     sg.add_nodes_from([x for x in subsystem_node_list])
#     sg.add_edges_from(subsystem_edge_list)

#     connected_comp_list = [x for x in nx.connected_components(sg)] # gives list of nodes which are connected to each other
#     # print('connected_comp_list', connected_comp_list)
#     sg_cs_list = []
#     for comp in connected_comp_list:
#         comp_cs_list = [graph.nodes[x]['pi']/len(comp) for x in list(comp)]
#         sg_cs_list.extend(comp_cs_list)
#         # average_comp_cs = sum(comp_cs_list) / len(comp_cs_list)
#         # subsystem_cs_list.append(average_comp_cs)
#     # print('sg_cs_list', sg_cs_list)
#     average_sg_cs = sum(sg_cs_list) / len(sg_cs_list)
#     # subsystem_cs_list.append(average_sg_cs)
#     return average_sg_cs - average_cs