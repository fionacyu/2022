import optimize
import load_data
from cmath import cos, exp
import miscellaneous
import networkx as nx
from itertools import chain
import numpy as np 
from collections import Counter
import math
import multiprocessing as mp

def bond_order_penalty(graph, edges_to_cut_list):
    penalty = 0
    for edges in edges_to_cut_list:
        penalty += graph[edges[0]][edges[1]]['bo']
    return penalty


def branching_penalty(graph, edges_to_cut_list):
    penalty = 0
    for edges in [e for e in edges_to_cut_list if graph[e[0]][e[1]]['bo'] == 1]:
        # if graph[edges[0]][edges[1]]['bo'] == 1: # only accounting for alkane branching
        nodeList = [x for x in graph.neighbors(edges[0]) if graph.nodes[x]['ed'] == 4 and graph.nodes[x]['element'] == 'C'] + [x for x in graph.neighbors(edges[1]) if graph.nodes[x]['ed'] == 4 and graph.nodes[x]['element'] == 'C']
        penalty += len(nodeList) # corresponds to the number of branching interactions lost due to edges being cut
    return penalty

def hybridisation_penalty(graph, edges_to_cut_list): # fix this
    penalty = 0
    influential_edges = [e for e in edges_to_cut_list if graph[e[0]][e[1]]['bo'] >= 2]
    nodeList = list(set(chain(*influential_edges)))
    for node in nodeList:
        edges = [e for e in influential_edges if node in e]
        # edges = miscellaneous.node_in_edgelist(influential_edges, node)
        boList = [graph[e[0]][e[1]]['bo'] for e in edges]
        deltaboList = [x - 1 for x in boList]
        penalty += sum(deltaboList)
    return penalty


def conjugation_penalty(graph, edges_to_cut_list, conjugated_edges):
    # try:
    edges_of_interest = [e for e in edges_to_cut_list if graph[e[0]][e[1]]['conjugated'] == 'yes']
    if len(edges_of_interest) == 0:
        return 0

    # list of lists containing the conjugated systems that will be disturbed or broken from the breaking of edges (will have repeats)
    conjugated_systems = [miscellaneous.find_conjugated_system(conjugated_edges, x) for x in edges_of_interest]

    unique_conjugated_systems = set([tuple(x) for x in conjugated_systems])
    unique_conjugated_systems = [list(x) for x in unique_conjugated_systems]
    # unique_conjugated_systems = [] # remove repeats
    # for system in conjugated_systems:
    #     if system not in unique_conjugated_systems:
    #         unique_conjugated_systems.append(system)
    # # print(unique_conjugated_systems)

    system_cs_list = []
    subsystem_cs_list = []
    for system in unique_conjugated_systems:
        # print('system', system)
        # getting the original cs's
        nodeList = set(chain(*system))
        cs_list = [graph.nodes[x]['pi']/len(nodeList) for x in nodeList]
        average_cs = sum(cs_list) / len(cs_list)
        system_cs_list.append(average_cs)
    
        # getting the updated cs after breaking edges
        edges_to_remove = [x for x in system if x in edges_of_interest or x[::-1] in edges_of_interest]
        # subsystem_edge_list = [x for x in system if x not in edges_to_remove]
        subsystem_edge_list = list(set(system) - set(edges_to_remove))
        # print('subsystem_edge_list', subsystem_edge_list)
        subsystem_node_list = set(chain(*system))
        # print(subsystem_node_list)

        # constructing subgraph
        sg = nx.Graph()
        sg.add_nodes_from([x for x in subsystem_node_list])
        sg.add_edges_from(subsystem_edge_list)

        connected_comp_list = [x for x in nx.connected_components(sg)] # gives list of nodes which are connected to each other
        # print('connected_comp_list', connected_comp_list)
        sg_cs_list = []
        for comp in connected_comp_list:
            comp_cs_list = [graph.nodes[x]['pi']/len(comp) for x in list(comp)]
            sg_cs_list.extend(comp_cs_list)
            # average_comp_cs = sum(comp_cs_list) / len(comp_cs_list)
            # subsystem_cs_list.append(average_comp_cs)
        # print('sg_cs_list', sg_cs_list)
        average_sg_cs = sum(sg_cs_list) / len(sg_cs_list)
        subsystem_cs_list.append(average_sg_cs)
    
    # print(system_cs_list)
    # print(subsystem_cs_list)
    penalty = np.sum(np.array(subsystem_cs_list) - np.array(system_cs_list))
    # pool = mp.Pool(mp.cpu_count())
    # penalty = sum(pool.starmap_async(miscellaneous.conj_penalty_para, [(graph, system, edges_of_interest) for system in unique_conjugated_systems]).get() )
    # pool.close()
    return round(penalty,4)


def hyperconjugation_penalty(donorDict, acceptorDict, connectionDict, edges_to_cut_list):
    if len(connectionDict) == 0:
        return 0
    
    # penalty = 0
    # for connection in connectionDict:
    #     donor = connection[0]
    #     acceptor = connection[1]

    #     boxLabelList = donorDict[donor].boxLabelList + acceptorDict[acceptor].boxLabelList
    #     # edges in the same box or neighbouring boxes as the donors/acceptors
    #     edgesBoxes = [e for e in edges_to_cut_list if len(set([graph.nodes[e[0]]['box']]).intersection(boxLabelList)) > 0 or len(set([graph.nodes[e[1]]['box']]).intersection(boxLabelList)) > 0 ]
    #     influential_edges = [e for e in edgesBoxes if (e in connectionDict[connection].simple_paths or e in donorDict[donor].edges or e in acceptorDict[acceptor].edges) or (e[::-1] in connectionDict[connection].simple_paths or e[::-1] in donorDict[donor].edges or e[::-1] in acceptorDict[acceptor].edges)]
    #     #these are the edges that will impact the donor-acceptor pair
    #     if influential_edges:
    #         # print(connection)

    #         da_graph = nx.Graph()
    #         nodeList = donorDict[donor].nodes + acceptorDict[acceptor].nodes + [x for x in set(chain(*connectionDict[connection].simple_paths))]
    #         nodeList = list(dict.fromkeys(nodeList)) # remove duplicates that arise from connection edges which comprise terminal nodes of donors and acceptors
    #         edgeList = donorDict[donor].edges + acceptorDict[acceptor].edges + connectionDict[connection].simple_paths
    #         # print('edgeList: ', edgeList)
    #         rejected_edges = [e for e in edgeList if e in influential_edges or e[::-1] in influential_edges]
    #         # edgeList = [e for e in edgeList if e not in rejected_edges] # remove influential edges/ edges to cut
    #         edgeList = list(set(edgeList) - set(rejected_edges))
    #         da_graph.add_nodes_from(nodeList)
    #         da_graph.add_edges_from(edgeList)

    #         connected_comp_list = [x for x in nx.connected_components(da_graph)] # gives list of nodes which are connected to each other
    #         # print('connected_comp_list', connected_comp_list)

    #         dsList, asList = [], []
    #         for cc in connected_comp_list:
    #             cc_nodes = [x for x in cc]
    #             dnodes = miscellaneous.donor_acceptor_nodes(donorDict[donor], cc_nodes)
    #             anodes = miscellaneous.donor_acceptor_nodes(acceptorDict[acceptor], cc_nodes)

    #             # print('dnodes', dnodes)
    #             # print('anodes', anodes)

    #             if dnodes:
    #                 donor_electrons = sum([donorDict[donor].node_electrons[x] for x in dnodes])
    #                 da_node_number = len(dnodes) + len(anodes)
    #                 # print('ds', donor_electrons/da_node_number)
    #                 dsList.append(donor_electrons/da_node_number)
                
    #             if anodes:
    #                 donor_electrons = sum([donorDict[donor].node_electrons[x] for x in dnodes])
    #                 da_node_number = len(dnodes) + len(anodes)
    #                 # print('as', -1 * donor_electrons/da_node_number)
    #                 asList.append(-1 * donor_electrons/da_node_number)
            

    #         connection_penalty = 1/connectionDict[connection].bond_separation * ((sum(dsList)/len(dsList)) + sum(asList)/len(asList))
    #         # print('connection_penalty', connection_penalty)
    #         penalty += connection_penalty
    # print('len of connectionDict', len(connectionDict))
    pool = mp.Pool(mp.cpu_count())
    penalty = sum(pool.starmap_async(miscellaneous.hyperconj_penalty_connection, [( connection, connectionDict, donorDict, acceptorDict, edges_to_cut_list) for connection in connectionDict]).get())
    pool.close()


    return round(penalty,4)


def aromaticity_penalty(graph, aromaticDict, edges_to_cut_list):
    if len(aromaticDict) == 0:
        return 0

    # penalty = 0 
    # for asys in aromaticDict:
    #     #box edges go here
    #     boxLabelList = aromaticDict[asys].boxLabelList
    #     edgesBoxes = [e for e in edges_to_cut_list if len(set([graph.nodes[e[0]]['box']]).intersection(boxLabelList)) > 0 or len(set([graph.nodes[e[1]]['box']]).intersection(boxLabelList)) > 0 ]

    #     influential_edges = [e for e in edgesBoxes if e in miscellaneous.flatten(aromaticDict[asys].cycle_list) and e not in aromaticDict[asys].bridging_edges] # doesn't include bridging edges
    #     bedgeList = [e for e in edgesBoxes if e in miscellaneous.flatten(aromaticDict[asys].cycle_list) and e in aromaticDict[asys].bridging_edges] # bridging edges only

    #     nonbe_cycle_ind_list = miscellaneous.flatten([miscellaneous.index_of_cycle_list(aromaticDict[asys].cycle_list, edge) for edge in influential_edges])
    #     nonbe_cycle_ind_list = list(dict.fromkeys(nonbe_cycle_ind_list)) # get the unique values
    #     be_cycle_ind_list = miscellaneous.flatten([miscellaneous.index_of_cycle_list(aromaticDict[asys].cycle_list, edge) for edge in bedgeList])
    #     be_cycle_ind_list = list(dict.fromkeys(be_cycle_ind_list))
    #     # nonbe_cycle_ind_list = [x for x in nonbe_cycle_ind_list if x not in be_cycle_ind_list]
    #     nonbe_cycle_ind_list = list(set(nonbe_cycle_ind_list) - set(be_cycle_ind_list))

    #     # inc_penalty = aromaticDict[asys].size - (sum([len(aromaticDict[asys].nonbridging_edges[x]) for x in nonbe_cycle_ind_list]) + sum([len(aromaticDict[asys].nonbridging_edges[x]) for x in be_cycle_ind_list]) + len(bedgeList))
    #     inc_penalty = sum([len(aromaticDict[asys].nonbridging_edges[x]) for x in nonbe_cycle_ind_list]) + sum([len(aromaticDict[asys].nonbridging_edges[x]) for x in be_cycle_ind_list]) + len(bedgeList)
    #     penalty += inc_penalty
    pool = mp.Pool(mp.cpu_count())
    penalty = sum(pool.starmap_async(miscellaneous.aromaticity_penalty_para, [(graph, asys, aromaticDict, edges_to_cut_list) for asys in aromaticDict] ).get() )
    pool.close()

    return penalty

def ring_strain(size):
    m, d, a, b, l = 1.30822, -125.911, 0.312439, 1.19633, -188.39
    y = size
    strain = pow(y, -1*m) * (d - 1 - l) + ((y-1+l)/y) * np.exp(-1 * a * (y-1)) * np.cos(b * (y-1)) 
    return strain.real

def ring_penalty(graph, cycleDict, edges_to_cut_list):
    if len(cycleDict) == 0:
        return 0
    # box edges go here, get the edges that would only be near cycles
    boxLabelList = list(dict.fromkeys(miscellaneous.flatten([cycleDict[x].boxLabelList for x in cycleDict])))
    edgesBoxes = [e for e in edges_to_cut_list if len(set([graph.nodes[e[0]]['box']]).intersection(boxLabelList)) > 0 or  len(set([graph.nodes[e[1]]['box']]).intersection(boxLabelList)) > 0  ]

    impactedCycles = [c for c in cycleDict if len(set(edgesBoxes).intersection(cycleDict[c].edgeList)) > 0]
    impactedCycleSize = [len(cycleDict[c].edgeList) for c in impactedCycles]
    totalstrain = round(sum([ring_strain(x) for x in impactedCycleSize]),4)
    return abs(totalstrain)

def reference_vol(atoms, minAtomNo):
    atomCount = Counter(atoms)
    refRad = sum([v/len(atoms) * load_data.get_radii(k) for k,v in atomCount.items() ])
    refVol = 4/3 * math.pi * refRad**3 * minAtomNo
    return refVol


def volume_penalty(atoms, graph, edges_to_cut_list, proxMatrix, minAtomNo):
    refVol = reference_vol(atoms, minAtomNo)
    tgraph = graph.copy() 
    tgraph.remove_edges_from(edges_to_cut_list)
    connectedComp = (tgraph.subgraph(x) for x in nx.connected_components(tgraph))
    penalty = 0 
    for sg in connectedComp:
        penalty += (refVol - load_data.get_volume(sg, proxMatrix))**2
    # pool = mp.Pool(mp.cpu_count())
    # penalty = sum(pool.starmap_async(load_data.get_volume, [(sg, proxMatrix, refVol) for sg in connectedComp]).get())
    # pool.close()
    return penalty


def full_penalty(atoms, graph, edges_to_cut_list, conjugated_edges, donorDict, acceptorDict, connectionDict, aromaticDict, cycleDict, betalist, proxMatrix, minAtomNo):
    penalty_list = [bond_order_penalty(graph, edges_to_cut_list), aromaticity_penalty(graph, aromaticDict, edges_to_cut_list), ring_penalty(graph, cycleDict, edges_to_cut_list), branching_penalty(graph, edges_to_cut_list), hybridisation_penalty(graph, edges_to_cut_list), conjugation_penalty(graph, edges_to_cut_list, conjugated_edges), hyperconjugation_penalty(donorDict, acceptorDict, connectionDict, edges_to_cut_list), volume_penalty(atoms, graph, edges_to_cut_list, proxMatrix, minAtomNo)]
    penalty_list = np.array(penalty_list)
    beta_values = np.array(betalist)

    total_penalty = np.dot(penalty_list, beta_values)
    return total_penalty

def full_penalty_opt(x, feasible_edges, atoms, graph, conjugated_edges, donorDict, acceptorDict, connectionDict, aromaticDict, cycleDict, betalist, proxMatrix, minAtomNo):
    edges_to_cut_list = optimize.convert_bvector_edges(x, feasible_edges)
    penalty_list = [bond_order_penalty(graph, edges_to_cut_list), aromaticity_penalty(graph, aromaticDict, edges_to_cut_list), ring_penalty(graph, cycleDict, edges_to_cut_list), branching_penalty(graph, edges_to_cut_list), hybridisation_penalty(graph, edges_to_cut_list), conjugation_penalty(graph, edges_to_cut_list, conjugated_edges), hyperconjugation_penalty(donorDict, acceptorDict, connectionDict, edges_to_cut_list), volume_penalty(atoms, graph, edges_to_cut_list, proxMatrix, minAtomNo)]
    penalty_list = np.array(penalty_list)
    print('penalty_list:', penalty_list)
    beta_values = np.array(betalist)

    total_penalty = np.dot(penalty_list, beta_values)
    return total_penalty
    