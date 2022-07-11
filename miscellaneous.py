import rings
import hyperconj
import sys
from collections import Counter
import load_data
import graph_characterisation
import calculate_penalty
import networkx as nx
import numpy as np
import os
from itertools import product
from itertools import combinations
import math
from numpy import linalg
import time

def flatten(t):
    return [item for sublist in t for item in sublist]

def pairs(*lists):
    for t in combinations(lists, 2):
        for pair in product(*t):
            #Don't output pairs containing duplicated elements 
            if pair[0] != pair[1]:
                yield pair

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
    # da_nodes = [x for x in nodeList if x in da_object.nodes]
    da_nodes = list(set(nodeList).intersection(da_object.nodes))
    return da_nodes

def index_of_cycle_list(cycle_list, edge):
    # print('cycle_list', cycle_list)
    blist = [edge in x for x in cycle_list]
    ind_list = [i for i, x in enumerate(blist) if x == True]
    return ind_list

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
            # return (node1, dist)
            return (node1, len(current_path) - 1)
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
                reject_edges = list( (set(edgeList).intersection(conjEdgeList)).union((set([x[::-1] for x in edgeList]).intersection(conjEdgeList)) ))
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

def sigmoid_conj_hyper(x, max, tol=0.05):
    # the min value is 0
    # a - exponent, x -variable
    a = -1 / max * math.log(tol/(2 - tol))
    return (1 - math.exp(-1 * a * x))/(1 + math.exp(-1 * a * x))

def nodes_connected(u, v, G):
    return u in G.neighbors(v)


def full_penalty(atoms, graph, pos, edges_to_cut_list, conjugated_edges, donorDict, acceptorDict, connectionDict, aromaticDict, betalist, proxMatrix, minAtomNo, E, prmDict):
    penalty_list = [calculate_penalty.bond_order_penalty(graph, edges_to_cut_list), calculate_penalty.aromaticity_penalty(graph, aromaticDict, edges_to_cut_list), calculate_penalty.peff_penalty5(graph, edges_to_cut_list, E, prmDict), calculate_penalty.conjugation_penalty(graph, edges_to_cut_list, conjugated_edges), calculate_penalty.hyperconjugation_penalty(donorDict, acceptorDict, connectionDict, edges_to_cut_list), calculate_penalty.volume_penalty(atoms, graph, edges_to_cut_list, proxMatrix, minAtomNo)]
    penalty_list = np.array(penalty_list)
    print(("%-20s " * len(penalty_list)) % tuple([str(i) for i in penalty_list]), file=open('penalties.dat', "a"))
    print(' '.join(str(j) for j in pos),file=open('positions.dat', "a"))
    beta_values = np.array(betalist)

    total_penalty = np.dot(penalty_list, beta_values)
    print(total_penalty, file=open('cost.dat', "a"))
    return total_penalty

def get_fragments(graph, optimal_edges_to_cut, coordinates):
    fgraph = graph.copy()
    fgraph.remove_edges_from(optimal_edges_to_cut)

    symbolList, coordList, weightList, idList = [], [], [], []
    hfragDict = {}
    fragNodes = {}
    for i, cc in enumerate(nx.connected_components(fgraph)):
        symbols = [graph.nodes[x]['element'] for x in cc]
        coords = flatten([coordinates[x-1] for x in cc])
        weights = [-1] * len(cc)
        fragids = [i+1] * len(cc)

        symbolList.extend(symbols)
        coordList.extend(coords)
        weightList.extend(weights)
        idList.extend(fragids)

        fragNodes[i+1] = [x for x in cc]

        nodes_affected = set(flatten(optimal_edges_to_cut)).intersection(cc)
        edges_cut = [e for e in optimal_edges_to_cut if any([e[0] in nodes_affected, e[1] in nodes_affected])]

        for edge in edges_cut:
            if edge not in hfragDict:
                hfragDict[edge] = [i+1]
            else:
                hfragDict[edge].append(i+1)
    
    count = len([x for x in nx.connected_components(fgraph)]) + 1
    monids = range(1,count)
    monpairs = [x for x in combinations(monids, 2)]
    # print('monpairs', monpairs)
    existing_pairs = {}
    for edge in optimal_edges_to_cut:
        nodes = nx.node_connected_component(fgraph, edge[0]) | nx.node_connected_component(fgraph, edge[1])
        symbols = [graph.nodes[x]['element'] for x in nodes]
        coords = flatten([coordinates[x-1] for x in nodes])
        
        for pair in monpairs:
            # print('mon nodes', set(fragNodes[pair[0]]) | set(fragNodes[pair[1]]))
            # print('dimer nodes', set(nodes))
            if (set(fragNodes[pair[0]]) | set(fragNodes[pair[1]])).issubset(set(nodes)):

                if pair not in existing_pairs:
                    weights = [1] * len(nodes)
                    fragids = [count] * len(nodes)

                    symbolList.extend(symbols)
                    coordList.extend(coords)
                    weightList.extend(weights)
                    idList.extend(fragids)

                    print(count, pair[0], pair[1], file=open('dimercomp.dat', 'a'))
                    existing_pairs[pair] = count # this is ok bc a pair of monomers will correspond to one dimer only

                    fragNodes[count] = [x for x in nodes]

                    nodes_affected = set(flatten(optimal_edges_to_cut)).intersection(nodes) 
                    edges_cut = [e for e in optimal_edges_to_cut if any([e[0] in nodes_affected, e[1] in nodes_affected])]
                    edges_cut.remove(edge)
                    for e in edges_cut:
                        if e not in hfragDict:
                            hfragDict[e] = [count]
                        else:
                            hfragDict[e].append(count)
                    count += 1

                break
            else:
                continue

    return symbolList, coordList, weightList, idList, hfragDict, fragNodes

def get_fragments_sg(graph, optimal_edges_to_cut, idcount):
    fgraph = graph.copy()
    fgraph.remove_edges_from(optimal_edges_to_cut)

    symbolList, coordList, idList = [], [], []
    hfragDict = {}
    fragNodes = {}

    count = idcount
    for i, cc in enumerate(nx.connected_components(fgraph)):
        symbols = [graph.nodes[x]['element'] for x in cc]
        coords = flatten([graph.nodes[x]['coord'] for x in cc])
        fragids = [count] * len(cc)

        symbolList.extend(symbols)
        coordList.extend(coords)
        idList.extend(fragids)

        fragNodes[count] = [x for x in cc]

        nodes_affected = set(flatten(optimal_edges_to_cut)).intersection(cc)
        edges_cut = [e for e in optimal_edges_to_cut if any([e[0] in nodes_affected, e[1] in nodes_affected])]

        for edge in edges_cut:
            if edge not in hfragDict:
                hfragDict[edge] = [count]
            else:
                hfragDict[edge].append(count)
        
        count += 1
    
    return symbolList, coordList, idList, hfragDict, fragNodes, count-1

def peff_hfrags(graph, edges_to_cut_list):
    # print(edges_to_cut_list)
    fgraph = graph.copy()
    fgraph.remove_edges_from(edges_to_cut_list)

    monFrags, jdimerFrags = {}, {}
    monHcaps, jdimerHcaps = {}, {}
    connectedComp = (fgraph.subgraph(x) for x in nx.connected_components(fgraph))
    # ncount = len(list(graph.nodes))
    ncount = max(graph.nodes)
    cutedge_monid = {}
    monid_cutedge = {}
    jd_cutedge = {}
    # print('ncount', ncount)
    for i, sg in enumerate(connectedComp):
        sg1 = sg.copy()
        # print('mon :', i+1)
        cc = list(sg.nodes)
        # print('cc', cc)
        # fragNodes[i+1] = [x for x in cc]

        nodes_affected = set(flatten(edges_to_cut_list)).intersection(cc) 
        cutedge_monid.update(dict.fromkeys(list(nodes_affected), i+1))
        monid_cutedge[i+1] = list(nodes_affected)
        edges_cut = [e for e in edges_to_cut_list if any([e[0] in nodes_affected, e[1] in nodes_affected])] # gets the edges that were cut to make the fragment

        # print('edges_cut', edges_cut)
        for edge in edges_cut:
            # print('edge', edge)
            # print(set(edge), set(cc))
            try:
                nodeh = min(set(edge) - set(cc)) # node replaced with hydrogen 
                # print('nodeh', nodeh)
                othernode = min(set(edge).intersection([x for x in cc]))

                nodehel = graph.nodes[nodeh]['element']
                othernodeel = graph.nodes[othernode]['element']

                if nodehel == 'C':
                        degree = graph.degree[nodeh]
                        nodehel = 'C' + str(degree)
                    
                if othernodeel == 'C':
                    degree = graph.degree[othernode]
                    othernodeel = 'C' + str(degree)

                scalar = (load_data.get_covradii('H') + load_data.get_covradii(othernodeel)) / (load_data.get_covradii(nodehel) + load_data.get_covradii(othernodeel))
                # hcoords = list(np.array(coordinates[othernode -1]) + scalar * (np.array(coordinates[nodeh -1]) - np.array(coordinates[othernode-1])))
                hcoords = list(np.array(graph.nodes[othernode]['coord']) + scalar * (np.array(graph.nodes[nodeh]['coord']) - np.array(graph.nodes[othernode]['coord'])))

                ncount += 1
                # print(ncount, {"element": 'H', "coord": hcoords})
                sg1.add_node(ncount, **{"element": 'H', "charge": 0, "coord": hcoords,"ed":1,  "at": 'H_'})
                sg1.add_edge(ncount, othernode, **{'bo': 1, 'r': linalg.norm(np.array(hcoords) - np.array(graph.nodes[othernode]['coord']))})
            
            except ValueError: # the nodes are still connected (occurs in ring systems)
                continue
                # sys.exit()
            
        monFrags['%d' % (i+1)] = sg1
        monHcaps['%d' % (i+1)] = len(list(sg1.nodes)) - len(cc)
    existing_pairs = []
    # print('edges_to_cut_list', edges_to_cut_list)
    for e in edges_to_cut_list: # dimers
        ##############
        node1, node2 = e[0], e[1]
        mon1, mon2 = cutedge_monid[node1], cutedge_monid[node2]
        # nodes = nx.node_connected_component(fgraph, e[0]) | nx.node_connected_component(fgraph, e[1])
        hcaps = 0 
        if mon1 != mon2:
            monpair = tuple(sorted([mon1, mon2]))
            if monpair not in existing_pairs:
                existing_pairs.append(monpair) 
                mon1nodes, mon2nodes = [x for x in monFrags[str(monpair[0])].nodes if x <= max(graph.nodes)], [x for x in monFrags[str(monpair[1])].nodes if x <= max(graph.nodes)]
                mon1graph, mon2graph = monFrags[str(monpair[0])].copy(), monFrags[str(monpair[1])].copy()
                mon1graph.remove_nodes_from(list(set(mon1graph.nodes) - set(mon1nodes)))
                mon2graph.remove_nodes_from(list(set(mon2graph.nodes) - set(mon2nodes)))
                jdimer = nx.compose(mon1graph, mon2graph)
                jdimer.add_edge(e[0], e[1], **{'bo': graph[e[0]][e[1]]['bo'], 'r': linalg.norm(np.array(graph.nodes[e[0]]['coord']) - np.array(graph.nodes[e[1]]['coord']))})

                nodes_affected = monid_cutedge[mon1] + monid_cutedge[mon2]
                for node_cut in nodes_affected:
                    nc_el = graph.nodes[node_cut]['element']
                    if nc_el == 'C':
                        nc_el = 'C' + str(graph.degree[node_cut])
                    neighbourList = list(graph.neighbors(node_cut))
                    neighbourList = list(set(neighbourList) - set(jdimer.nodes))# [x for x in neighbourList if x not in set(jdimer.nodes)]
                    
                    # the nodes in neighbourList will be replaced with hydrogen caps 
                    neighElements = [graph.nodes[x]['element'] if graph.nodes[x]['element'] != 'C' else 'C' + str(graph.degree[x]) for x in neighbourList]
                    # print(neighElements)
                    neighScalar = [(load_data.get_covradii('H') + load_data.get_covradii(nc_el)) / (load_data.get_covradii(x) + load_data.get_covradii(nc_el)) for x in neighElements]
                    neighHcoords = [list(np.array(graph.nodes[node_cut]['coord']) + neighScalar[j] * (np.array(graph.nodes[neighbourList[j]]['coord']) - np.array(graph.nodes[node_cut]['coord']))) for j in range(len(neighbourList))]
                    hcaps += len(neighbourList)

                    for k in range(len(neighbourList)):
                        ncount +=1
                        jdimer.add_node(ncount, **{"element": 'H', "charge": 0, "coord": neighHcoords[k], "ed":1, "at": 'H_'})
                        jdimer.add_edge(ncount, node_cut, **{'bo': 1, 'r': linalg.norm(np.array(neighHcoords[k]) - np.array(graph.nodes[node_cut]['coord']))})
                        
                jdimerFrags['%d_%d' % (monpair[0], monpair[1])] = jdimer
                jdimerHcaps['%d_%d' % (monpair[0], monpair[1])] = hcaps
                jd_cutedge['%d_%d' % (monpair[0], monpair[1])] = [e]
            else:
                jdimer1 = jdimerFrags['%d_%d' % (monpair[0], monpair[1])]
                jdimer1.add_edge(e[0], e[1], **{'bo': graph[e[0]][e[1]]['bo'], 'r': linalg.norm(np.array(graph.nodes[e[0]]['coord']) - np.array(graph.nodes[e[1]]['coord']))})
                jdimerFrags['%d_%d' % (monpair[0], monpair[1])] = jdimer1
                jd_cutedge['%d_%d' % (monpair[0], monpair[1])].append(e)

    return monFrags, monHcaps, jdimerFrags, jdimerHcaps, jd_cutedge

def centroid(graph):
    mon_centroid = np.zeros(3)

    for i in graph.nodes:
        mon_centroid[0] += graph.nodes[i]['coord'][0]
        mon_centroid[1] += graph.nodes[i]['coord'][1]
        mon_centroid[2] += graph.nodes[i]['coord'][2]
    
    weight = 1.0/len(graph.nodes)
    mon_centroid[0] *= weight
    mon_centroid[1] *= weight
    mon_centroid[2] *= weight

    return mon_centroid

def dimer_distance(mon1g, mon2g):
    mon1_centroid = centroid(mon1g)
    mon2_centroid = centroid(mon2g)

    ix, iy, iz = mon1_centroid[0], mon1_centroid[1], mon1_centroid[2]
    jx, jy, jz = mon2_centroid[0], mon2_centroid[1], mon2_centroid[2]

    distance_sqr = (jx - ix) * (jx - ix)
    distance_sqr += (jy - iy) * (jy - iy)
    distance_sqr += (jz - iz) * (jz - iz)

    return np.sqrt(distance_sqr)

def disjoint_dimers(monFrags, jdimerFrags, cutoff=30): #distance is in bohr
    distance_cutoff = cutoff * 0.529177 #converting distance to angstrom
    monKeys = list(monFrags)
    # monPairs = combinations(monKeys, 2)

    ddimerFrags = []
    # for pair in monPairs:
    #     if "%s_%s" % (pair[0], pair[1]) in jdimerFrags or "%s_%s" % (pair[1], pair[0]) in jdimerFrags:
    #         continue
    #     else:
    #         ddgraph = nx.compose(monFrags[pair[0]], monFrags[pair[1]])
    #         ddimerFrags["%s_%s" % (pair[0], pair[1])] = ddgraph
    
    # os.system('mkdir ddfrags')
    # for dd in ddimerFrags:
    #     atomno = len(ddimerFrags[dd].nodes)
    #     print('%d\n' % atomno, file=open('ddfrags/%s.xyz' % dd, 'a'))

    #     for node in list(ddimerFrags[dd].nodes):
    #         print('%s' % ddimerFrags[dd].nodes[node]['element'], '\t'.join(str(hc) for hc in ddimerFrags[dd].nodes[node]['coord']), file=open('ddfrags/%s.xyz' % dd, 'a'))
    for i in monKeys:
        jList = [x for x in monKeys if x < i]
        for j in jList:
            monpair = tuple(sorted([i, j]))
            if i == j:
                continue
            if '%s_%s' % (monpair[0], monpair[1]) in jdimerFrags:
                continue
            if dimer_distance(monFrags[i], monFrags[j]) <= distance_cutoff:
                ddimerFrags.append('%s_%s' % (monpair[0], monpair[1]))


    return ddimerFrags


def fragment_xyz(symbolList, coordList, idList, graph, coordinates, hfragDict, fragNodes):
    os.system('mkdir fragxyz')
    # print('hfragDict')
    # print('edge, fragid')
    # for item in hfragDict:
    #     print(item, hfragDict[item])
    # for k, v in Counter(idList).items():
    #     print('%s\n' % v, file=open('fragxyz/%s.xyz' % k, 'a'))

    for i in range(len(symbolList)):
        print(symbolList[i], '\t'.join(str(j) for j in coordList[3*i: 3*i + 3]), file=open('fragxyz/%s.xyz' % idList[i], 'a'))
    
    for edge, idlist in hfragDict.items():
        # print('edge', edge)
        for id in idlist:
            # print('id', id)
            # print('fragNodes[id]', fragNodes[id])
            try:
                nodeh = min(set(edge) - set(fragNodes[id]))  # this node will be replaced with hydrogen
                othernode = min(set(edge).intersection(fragNodes[id]))

                nodehel = graph.nodes[nodeh]['element']
                othernodeel = graph.nodes[othernode]['element']

                if nodehel == 'C':
                    degree = graph.degree[nodeh]
                    nodehel = 'C' + str(degree)
                
                if othernodeel == 'C':
                    degree = graph.degree[othernode]
                    othernodeel = 'C' + str(degree)

                scalar = (load_data.get_covradii('H') + load_data.get_covradii(othernodeel)) / (load_data.get_covradii(nodehel) + load_data.get_covradii(othernodeel))
                hcoords = list(np.array(coordinates[othernode -1]) + scalar * (np.array(coordinates[nodeh -1]) - np.array(coordinates[othernode-1])))
                # print('hcoords', hcoords)
                print('H', '\t'.join(str(hc) for hc in hcoords), file=open('fragxyz/%s.xyz' % id, 'a'))

            except ValueError:
                continue
            
    nonHcap = Counter(idList)
    for fragID in set(idList):
        with open('fragxyz/%s.xyz' % fragID, 'r') as f:
            data = f.read()
            lines = data.split('\n')
            atomno = len(list(filter(None, lines)))
            # print('atomno', atomno)
        with open('fragxyz/%s.xyz' % fragID, 'w') as fw:
            fw.write('%d\n' % atomno + '\n' + data)
        
        ogatomno = nonHcap[fragID]
        hcap = atomno - ogatomno
        print(fragID, hcap, file=open('hcapno.dat', 'a'))


def hyperconj_connections_para(graphDict, comb, donorDict, acceptorDict):
    donorLabel = comb[0]
    acceptorLabel = comb[1]

    donor_terminal_nodes = donorDict[donorLabel].terminal_nodes
    acceptor_terminal_nodes = acceptorDict[acceptorLabel].terminal_nodes

    terminal_comb_list = list(product(donor_terminal_nodes, acceptor_terminal_nodes))
    successfulIters = []
    for i, tcomb in enumerate(terminal_comb_list):
        path = rings.shortest_path(graphDict, tcomb[0], tcomb[1], 3)
        if len(path) > 0:
            successfulIters.append(i)
            if len(path) == 2: # bond separation is 1 bc the number of bonds == len(path) - 1
                daConnection = hyperconj.DonorAcceptorConnection()
                daConnection.add_simple_paths([tuple(sorted((path[i], path[i+1]))) for i in range(len(path)-1)])
                daConnection.add_bond_separation(len(path)-1)
                # connectionDict[comb] = daConnection
                break
            
            if i == successfulIters[0]:
                daConnection = hyperconj.DonorAcceptorConnection()
                daConnection.add_simple_paths([tuple(sorted((path[i], path[i+1]))) for i in range(len(path)-1)])
                daConnection.add_bond_separation(len(path)-1)
            
            elif i > successfulIters[0]:
                if len(path) - 1 < daConnection.bond_separation:
                    daConnection.add_simple_paths([tuple(sorted((path[i], path[i+1]))) for i in range(len(path)-1)])
                    daConnection.add_bond_separation(len(path)-1)
    if len(successfulIters) > 0:
        return (comb, daConnection)
    else:
        return (comb, None)


def check_brokenbonds(graph, optimal_edges_to_cut):
    fgraph = graph.copy()
    fgraph.remove_edges_from(optimal_edges_to_cut)
    edgeList = optimal_edges_to_cut

    for edges in optimal_edges_to_cut:
        node1, node2 = edges[0], edges[1]
        node1connected = set(nx.node_connected_component(fgraph, node1))
        if node2 in node1connected:
            edgeList.remove(edges)
    
    return edgeList


# def hybridisation2_para(graph, node, proxMatrix, tol):
    
#     bondED, bondElec = 0, 0
#     bondVector = proxMatrix[:,node-1] + proxMatrix[node-1,:] 
#     otherAtoms = [x for x in range(len(bondVector)) if bondVector[x] < 3 and bondVector[x] > 0]
#     for j in otherAtoms:
#         atom1, atom2 = graph.nodes[node]['element'], graph.nodes[j+1]['element']
#         bo = load_data.get_bond_order(atom1, atom2, bondVector[j], tol)
#         if bo != 0:
#             if len(set([(node, j+1), (j+1, node)]).intersection(node)) == 0:
#                 # edgeAttr[(nodeNumber, j+1)] = {'bo': bo}
#                 eAttrTuple = (tuple(sorted((node, j+1))), {'bo': bo} )
#             bondElec = bondElec + 2 * bo
#             bondED = bondED + 1
#     valElec = load_data.get_valence(graph.nodes[node]['element'])
#     elecDom = math.ceil(bondED + 0.5 * (valElec - 0.5 * bondElec - graph.nodes[node]['charge'])) 
#     nAttrTuple = (node, {'ed': elecDom})

#     return [nAttrTuple, eAttrTuple]