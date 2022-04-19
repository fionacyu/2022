import boxing
import load_data
from cmath import cos, exp
import miscellaneous
import networkx as nx
from itertools import chain
import numpy as np 
import math

def bond_order_penalty(graph, edges_to_cut_list):
    penalty = 0
    for edges in edges_to_cut_list:
        penalty += graph[edges[0]][edges[1]]['bo']
    return penalty


def branching_penalty(graph, edges_to_cut_list):
    penalty = 0
    for edges in edges_to_cut_list:
        if graph[edges[0]][edges[1]]['bo'] == 1: # only accounting for alkane branching
            nodeList = [x for x in graph.neighbors(edges[0]) if graph.nodes[x]['ed'] == 4 and graph.nodes[x]['element'] == 'C'] + [x for x in graph.neighbors(edges[1]) if graph.nodes[x]['ed'] == 4 and graph.nodes[x]['element'] == 'C']
            penalty += len(nodeList) # corresponds to the number of branching interactions lost due to edges being cut
    return penalty

def hybridisation_penalty(graph, edges_to_cut_list): # fix this
    penalty = 0
    influential_edges = [e for e in edges_to_cut_list if graph[e[0]][e[1]]['bo'] >= 2]
    nodeList = list(set(chain(*influential_edges)))
    for node in nodeList:
        edges = [e for e in influential_edges if node in e]
        boList = [graph[e[0]][e[1]]['bo'] for e in edges]
        deltaboList = [x - 1 for x in boList]
        penalty += sum(deltaboList)
    return penalty


def conjugation_penalty(graph, edges_to_cut_list, conjugated_edges):
    try:
        edges_of_interest = [e for e in edges_to_cut_list if graph[e[0]][e[1]]['conjugated'] == 'yes']
    # print('edges_of_interest', edges_of_interest)
    except KeyError:
        return 0

    # list of lists containing the conjugated systems that will be disturbed or broken from the breaking of edges (will have repeats)
    conjugated_systems = [miscellaneous.find_conjugated_system(conjugated_edges, x) for x in edges_of_interest]

    unique_conjugated_systems = [] # remove repeats
    for system in conjugated_systems:
        if system not in unique_conjugated_systems:
            unique_conjugated_systems.append(system)
    # print(unique_conjugated_systems)

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
        subsystem_edge_list = [x for x in system if x not in edges_to_remove]
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
    return round(penalty,4)


def hyperconjugation_penalty(graph, donorDict, acceptorDict, connectionDict, edges_to_cut_list, boxDict):
    penalty = 0
    for connection in connectionDict:
        donor = connection[0]
        acceptor = connection[1]

        boxLabelList = donorDict[donor].boxLabelList + acceptorDict[acceptor].boxLabelList
        neighBoxes = boxing.neighbouring_boxes(boxLabelList, boxDict)
        # edges in the same box or neighbouring boxes as the donors/acceptors
        edgesNeighBoxes = [e for e in edges_to_cut_list if graph.nodes[e[0]]['box'] in neighBoxes or graph.nodes[e[1]]['box'] in neighBoxes]
        influential_edges = [e for e in edgesNeighBoxes if (e in connectionDict[connection].simple_paths or e in donorDict[donor].edges or e in acceptorDict[acceptor].edges) or (e[::-1] in connectionDict[connection].simple_paths or e[::-1] in donorDict[donor].edges or e[::-1] in acceptorDict[acceptor].edges)]
        #these are the edges that will impact the donor-acceptor pair
        if influential_edges:
            # print(connection)

            da_graph = nx.Graph()
            nodeList = donorDict[donor].nodes + acceptorDict[acceptor].nodes + [x for x in set(chain(*connectionDict[connection].simple_paths))]
            nodeList = list(dict.fromkeys(nodeList)) # remove duplicates that arise from connection edges which comprise terminal nodes of donors and acceptors
            edgeList = donorDict[donor].edges + acceptorDict[acceptor].edges + connectionDict[connection].simple_paths
            # print('edgeList: ', edgeList)
            rejected_edges = [e for e in edgeList if e in influential_edges or e[::-1] in influential_edges]
            edgeList = [e for e in edgeList if e not in rejected_edges] # remove influential edges/ edges to cut
            da_graph.add_nodes_from(nodeList)
            da_graph.add_edges_from(edgeList)

            connected_comp_list = [x for x in nx.connected_components(da_graph)] # gives list of nodes which are connected to each other
            # print('connected_comp_list', connected_comp_list)

            dsList, asList = [], []
            for cc in connected_comp_list:
                cc_nodes = [x for x in cc]
                dnodes = miscellaneous.donor_acceptor_nodes(donorDict[donor], cc_nodes)
                anodes = miscellaneous.donor_acceptor_nodes(acceptorDict[acceptor], cc_nodes)

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
            penalty += connection_penalty
    
    return penalty


def aromaticity_penalty(graph, aromaticDict, edges_to_cut_list, boxDict):
    penalty = 0 
    for asys in aromaticDict:
        #box edges go here
        boxLabelList = aromaticDict[asys].boxLabelList
        neighBoxes = boxing.neighbouring_boxes(boxLabelList, boxDict)
        edgesNeighBoxes = [e for e in edges_to_cut_list if graph.nodes[e[0]]['box'] in neighBoxes or graph.nodes[e[1]]['box'] in neighBoxes]

        influential_edges = [e for e in edgesNeighBoxes if e in miscellaneous.flatten(aromaticDict[asys].cycle_list) and e not in aromaticDict[asys].bridging_edges] # doesn't include bridging edges
        bedgeList = [e for e in edgesNeighBoxes if e in miscellaneous.flatten(aromaticDict[asys].cycle_list) and e in aromaticDict[asys].bridging_edges] # bridging edges only

        nonbe_cycle_ind_list = miscellaneous.flatten([miscellaneous.index_of_cycle_list(aromaticDict[asys].cycle_list, edge) for edge in influential_edges])
        nonbe_cycle_ind_list = list(dict.fromkeys(nonbe_cycle_ind_list)) # get the unique values
        be_cycle_ind_list = miscellaneous.flatten([miscellaneous.index_of_cycle_list(aromaticDict[asys].cycle_list, edge) for edge in bedgeList])
        be_cycle_ind_list = list(dict.fromkeys(be_cycle_ind_list))
        nonbe_cycle_ind_list = [x for x in nonbe_cycle_ind_list if x not in be_cycle_ind_list]

        # inc_penalty = aromaticDict[asys].size - (sum([len(aromaticDict[asys].nonbridging_edges[x]) for x in nonbe_cycle_ind_list]) + sum([len(aromaticDict[asys].nonbridging_edges[x]) for x in be_cycle_ind_list]) + len(bedgeList))
        inc_penalty = sum([len(aromaticDict[asys].nonbridging_edges[x]) for x in nonbe_cycle_ind_list]) + sum([len(aromaticDict[asys].nonbridging_edges[x]) for x in be_cycle_ind_list]) + len(bedgeList)
        penalty += inc_penalty
    return penalty

def ring_strain(size):
    m, d, a, b, l = 1.30822, -125.911, 0.312439, 1.19633, -188.39
    y = size
    strain = pow(y, -1*m) * (d - 1 - l) + ((y-1+l)/y) * np.exp(-1 * a * (y-1)) * np.cos(b * (y-1)) 
    return strain.real

def ring_penalty(graph, cycleDict, edges_to_cut_list, boxDict):
    # box edges go here, get the edges that would only be near cycles
    boxLabelList = list(dict.fromkeys(miscellaneous.flatten([cycleDict[x].boxLabelList for x in cycleDict])))
    neighBoxes = boxing.neighbouring_boxes(boxLabelList, boxDict)
    edgesNeighBoxes = [e for e in edges_to_cut_list if graph.nodes[e[0]]['box'] in neighBoxes or graph.nodes[e[1]]['box'] in neighBoxes]

    impactedCycles = [c for c in cycleDict if len(set(edgesNeighBoxes).intersection(cycleDict[c].edgeList)) > 0]
    impactedCycleSize = [len(cycleDict[c].edgeList) for c in impactedCycles]
    totalstrain = round(sum([ring_strain(x) for x in impactedCycleSize]),4)
    return abs(totalstrain)

def volume_penalty(graph, edges_to_cut_list, proxMatrix, refRad):
    refVol = 4/3 * math.pi * refRad**3
    graph.remove_edges_from(edges_to_cut_list)
    connectedComp = (graph.subgraph(x) for x in nx.connected_components(graph))
    penalty = 0 
    for sg in connectedComp:
        penalty += (refVol - load_data.get_volume(sg, proxMatrix))**2
    return penalty


def full_penalty(graph, edges_to_cut_list, conjugated_edges, donorDict, acceptorDict, connectionDict, aromaticDict, cycleDict, betalist, boxDict, proxMatrix, refRad):
    penalty_list = [bond_order_penalty(graph, edges_to_cut_list), aromaticity_penalty(graph, aromaticDict, edges_to_cut_list, boxDict), ring_penalty(graph, cycleDict, edges_to_cut_list, boxDict), branching_penalty(graph, edges_to_cut_list), hybridisation_penalty(graph, edges_to_cut_list), conjugation_penalty(graph, edges_to_cut_list, conjugated_edges), hyperconjugation_penalty(graph, donorDict, acceptorDict, connectionDict, edges_to_cut_list, boxDict), volume_penalty(graph, edges_to_cut_list, proxMatrix, refRad)]
    penalty_list = np.array(penalty_list)
    beta_values = np.array(betalist)

    total_penalty = np.dot(penalty_list, beta_values)
    return total_penalty
    



