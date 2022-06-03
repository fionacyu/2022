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
    boList = [int(graph[edges[0]][edges[1]]['bo']) for edges in edges_to_cut_list]
    penaltyList = [(bo**2 - bo * 1)/bo**2 for bo in boList]
    
    if penaltyList:
        return round(sum(penaltyList)/len(penaltyList),4) #normalised
    else:
        return 0


def conjugation_penalty(graph, edges_to_cut_list, conjugated_edges):
    # try:
    edges_of_interest = [e for e in edges_to_cut_list if graph[e[0]][e[1]]['conjugated'] == 'yes']
    if len(edges_of_interest) == 0:
        return 0

    # list of lists containing the conjugated systems that will be disturbed or broken from the breaking of edges (will have repeats)
    conjugated_systems = [miscellaneous.find_conjugated_system(conjugated_edges, x) for x in edges_of_interest]

    unique_conjugated_systems = set([tuple(x) for x in conjugated_systems])
    unique_conjugated_systems = [list(x) for x in unique_conjugated_systems]

    system_cs_list = []
    subsystem_cs_list = []
    subsystem_cs_worst = []
    for system in unique_conjugated_systems:
        # print('system', system)
        # getting the original cs's
        nodeList = set(chain(*system))
        cs_list = [graph.nodes[x]['pi']/len(nodeList) for x in nodeList]
        average_cs = sum(cs_list) / len(cs_list)
        system_cs_list.append(average_cs)
    
        # getting the updated cs after breaking edges
        edges_to_remove = [x for x in system if x in edges_of_interest or x[::-1] in edges_of_interest]
        subsystem_edge_list = list(set(system) - set(edges_to_remove))
        subsystem_node_list = set(chain(*system))
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
        average_sg_cs = sum(sg_cs_list) / len(sg_cs_list)
        subsystem_cs_list.append(average_sg_cs)
        subsystem_cs_worst.append(len(nodeList) - 1)
    # penalty = np.sum(np.array(subsystem_cs_list) - np.array(system_cs_list))
    penaltyList = list(np.divide(np.array(subsystem_cs_list) - np.array(system_cs_list), np.array(system_cs_list)))  # relative error list
    print(("%-20s " * 2) % tuple([round(sum(penaltyList)/len(penaltyList),4), round(sum(subsystem_cs_worst)/len(subsystem_cs_worst), 4)]), file=open('conjugation.dat', "a"))
    penaltyList = [miscellaneous.sigmoid_conj_hyper(penaltyList[i], subsystem_cs_worst[i]) for i in range(len(penaltyList))] # transformation ensures penalty is between 0 and 1
    return round(sum(penaltyList)/len(penaltyList),4) #normalised


def hyperconjugation_penalty(donorDict, acceptorDict, connectionDict, edges_to_cut_list):
    if len(connectionDict) == 0:
        return 0
    
    penaltyList = []
    bondsepweights = []
    worstscore = []
    for connection in connectionDict:
        donor = connection[0]
        acceptor = connection[1]

        # boxLabelList = donorDict[donor].boxLabelList + acceptorDict[acceptor].boxLabelList
        # # edges in the same box or neighbouring boxes as the donors/acceptors
        # edgesBoxes = [e for e in edges_to_cut_list if len(set([graph.nodes[e[0]]['box']]).intersection(boxLabelList)) > 0 or len(set([graph.nodes[e[1]]['box']]).intersection(boxLabelList)) > 0 ]
        # influential_edges = [e for e in edgesBoxes if (e in connectionDict[connection].simple_paths or e in donorDict[donor].edges or e in acceptorDict[acceptor].edges) or (e[::-1] in connectionDict[connection].simple_paths or e[::-1] in donorDict[donor].edges or e[::-1] in acceptorDict[acceptor].edges)]
        influential_edges = list( (set(edges_to_cut_list).intersection(connectionDict[connection].simple_paths)).union((set(edges_to_cut_list).intersection(donorDict[donor].edges))).union((set(edges_to_cut_list).intersection(acceptorDict[acceptor].edges))).union((set([e[::-1] for e in edges_to_cut_list]).intersection(connectionDict[connection].simple_paths))).union(set([e[::-1] for e in edges_to_cut_list]).intersection(donorDict[donor].edges)).union(set([e[::-1] for e in edges_to_cut_list]).intersection(acceptorDict[acceptor].edges)) )
        #these are the edges that will impact the donor-acceptor pair
        if influential_edges:
            # print('connection', connection)

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

            dsList, asList = np.array([]), np.array([])
            for cc in connected_comp_list:
                cc_nodes = [x for x in cc]
                dnodes = miscellaneous.donor_acceptor_nodes(donorDict[donor], cc_nodes)
                anodes = miscellaneous.donor_acceptor_nodes(acceptorDict[acceptor], cc_nodes)
                if dnodes:
                    donor_electrons = sum([donorDict[donor].node_electrons[x] for x in dnodes])
                    da_node_number = len(dnodes) + len(anodes)
                    # print('ds', donor_electrons/da_node_number)
                    dsList = np.append(dsList, donor_electrons/da_node_number)
                
                if anodes:
                    donor_electrons = sum([donorDict[donor].node_electrons[x] for x in dnodes])
                    da_node_number = len(dnodes) + len(anodes)
                    # print('as', -1 * donor_electrons/da_node_number)
                    asList = np.append(asList, -1 * donor_electrons/da_node_number)

            bondsepweights.append(1/connectionDict[connection].bond_separation)
            connection_penalty = np.average(dsList) + np.average(asList)
            # print('connection_penalty', connection_penalty)
            penaltyList.append(connection_penalty)
            worstscore.append(sum([donorDict[donor].node_electrons[x] for x in donorDict[donor].nodes]))

    # print('hyper penaltyList: ', penaltyList)
    bondsepweights = np.array(bondsepweights)
    penaltyList = np.array([miscellaneous.sigmoid_conj_hyper(penaltyList[i], worstscore[i]) for i in range(len(penaltyList))]) # transformation ensures penalty is between 0 and 1
    if penaltyList.size == 0:
        return 0
    else:
        fpenalty = np.dot(bondsepweights, penaltyList)/penaltyList.size # bond separation weights are considered
        return round(fpenalty,4) #normnalised


def aromaticity_penalty(graph, aromaticDict, edges_to_cut_list):
    if len(aromaticDict) == 0:
        return 0

    penaltyList = []
    for asys in aromaticDict:
        #box edges go here
        boxLabelList = aromaticDict[asys].boxLabelList
        edgesBoxes = [e for e in edges_to_cut_list if len(set([graph.nodes[e[0]]['box']]).intersection(boxLabelList)) > 0 or len(set([graph.nodes[e[1]]['box']]).intersection(boxLabelList)) > 0 ]

        influential_edges = [e for e in edgesBoxes if e in miscellaneous.flatten(aromaticDict[asys].cycle_list) and e not in aromaticDict[asys].bridging_edges] # doesn't include bridging edges
        bedgeList = [e for e in edgesBoxes if e in miscellaneous.flatten(aromaticDict[asys].cycle_list) and e in aromaticDict[asys].bridging_edges] # bridging edges only

        nonbe_cycle_ind_list = miscellaneous.flatten([miscellaneous.index_of_cycle_list(aromaticDict[asys].cycle_list, edge) for edge in influential_edges])
        nonbe_cycle_ind_list = list(dict.fromkeys(nonbe_cycle_ind_list)) # get the unique values
        be_cycle_ind_list = miscellaneous.flatten([miscellaneous.index_of_cycle_list(aromaticDict[asys].cycle_list, edge) for edge in bedgeList])
        be_cycle_ind_list = list(dict.fromkeys(be_cycle_ind_list))
        # nonbe_cycle_ind_list = [x for x in nonbe_cycle_ind_list if x not in be_cycle_ind_list]
        nonbe_cycle_ind_list = list(set(nonbe_cycle_ind_list) - set(be_cycle_ind_list))

        inc_penalty = sum([len(aromaticDict[asys].nonbridging_edges[x]) for x in nonbe_cycle_ind_list]) + sum([len(aromaticDict[asys].nonbridging_edges[x]) for x in be_cycle_ind_list]) + len(bedgeList)
        # penalty += inc_penalty
        # print('aromaticDict[asys].cycle_list', aromaticDict[asys].cycle_list)
        penaltyList.append(inc_penalty/len(miscellaneous.flatten(aromaticDict[asys].cycle_list))) # relative error?

    return round(sum(penaltyList)/len(penaltyList), 4)

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
    # return abs(totalstrain)
    return round(abs(totalstrain)/658.7,4) # relative error

def reference_vol(atoms, minAtomNo):
    atomCount = Counter(atoms)
    refRad = sum([v/len(atoms) * load_data.get_radii(k) for k,v in atomCount.items() ])
    refVol = 4/3 * math.pi * refRad**3 * minAtomNo
    return refVol


def volume_penalty(atoms, graph, edges_to_cut_list, proxMatrix, minAtomNo):
    refVol = reference_vol(atoms, minAtomNo)
    tgraph = graph.copy() 
    tgraph.remove_edges_from(edges_to_cut_list)
    # print([x for x in nx.connected_components(tgraph)], file=open('connectedCompVol.dat', 'a'))
    connectedComp = (tgraph.subgraph(x) for x in nx.connected_components(tgraph))
    penaltyList = [(load_data.get_volume(sg, proxMatrix)/refVol - 1)**2 for sg in connectedComp]
    return round(sum(penaltyList)/len(penaltyList),4)

def peff_penalty(graph, coordinates, edges_to_cut_list):
    # read in prm file
    prmDict = load_data.read_prm()
    # first need potential energy of entire system 
    

    monFrags, monHcaps, jdimerFrags, jdimerHcaps = miscellaneous.peff_hfrags(graph, coordinates, edges_to_cut_list)
    # ^^ these are all dictionaries
    # monFrags and jdimerFrags already have hydrogen caps appended to them 

    # get mbe2 energy for each individual energy type: bond, angle, torsional, inversion, vdw
    # then sum altogether?




def full_penalty(atoms, graph, edges_to_cut_list, conjugated_edges, donorDict, acceptorDict, connectionDict, aromaticDict, cycleDict, betalist, proxMatrix, minAtomNo):
    penalty_list = [bond_order_penalty(graph, edges_to_cut_list), aromaticity_penalty(graph, aromaticDict, edges_to_cut_list), ring_penalty(graph, cycleDict, edges_to_cut_list), conjugation_penalty(graph, edges_to_cut_list, conjugated_edges), hyperconjugation_penalty(donorDict, acceptorDict, connectionDict, edges_to_cut_list), volume_penalty(atoms, graph, edges_to_cut_list, proxMatrix, minAtomNo)]
    penalty_list = np.array(penalty_list)
    # print('penalty_list:', penalty_list)
    beta_values = np.array(betalist)

    total_penalty = np.dot(penalty_list, beta_values)
    return total_penalty

def full_penalty_opt(x, feasible_edges, atoms, graph, conjugated_edges, donorDict, acceptorDict, connectionDict, aromaticDict, cycleDict, betalist, proxMatrix, minAtomNo):
    edges_to_cut_list = optimize.convert_bvector_edges1(x, feasible_edges)
    pool = mp.Pool(mp.cpu_count())
    penalty_list = pool.starmap_async(miscellaneous.full_penalty, [(atoms, graph, x[i], edges_to_cut_list[i], conjugated_edges, donorDict, acceptorDict, connectionDict, aromaticDict, cycleDict, betalist, proxMatrix, minAtomNo) for i in range(len(edges_to_cut_list))]).get()
    pool.close()
    return penalty_list
    