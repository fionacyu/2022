import load_data
import miscellaneous
import math
import numpy as np 
import networkx as nx
from itertools import combinations
from itertools import chain
from itertools import product
from collections import Counter
import multiprocessing as mp
import time

def hybridisation2(graph, proxMatrix, tol=0.003):
    # returns the edge attributes
    # print(proxMatrix)
    # print('ha')
    # heavy atoms only, not concerned with hydrogens because they only form one single bond anyways
    ha = [x for x in list(graph.nodes) if graph.nodes[x]['element'] != 'H'] # contains integers which are node labels
    hlist = [x for x in list(graph.nodes) if graph.nodes[x]['element'] == 'H']

    for hatom in hlist:
        graph.nodes[hatom]['ed'] = 1 # electron domain of hydrogen is defaulted to 1
        graph.nodes[hatom]['at'] = 'H_'
    # print(ha)

    edgeAttr = {} #this will be used to construct the molecular graph of the molecule, attributes will be distance and bond type (single, double etc.)

    for nodeNumber in ha:
        # print('atom of interest: ', graph.nodes[nodeNumber]['element'])
        bondED = 0 # to begin with, the no. of electron domains from bonding is 0
        bondElec = 0 # to begin with, the no. of bonding electrons is 0

        bondVector = proxMatrix[:,nodeNumber-1] + proxMatrix[nodeNumber-1,:] # contains the distances
        # print(bondVector)
        otherAtoms = [x for x in range(len(bondVector)) if bondVector[x] < 3 and bondVector[x] > 0]
        # otherAtoms = [list(bondVector).index(x) for x in bondVector if x < 3 and x > 0]
        # print('otherAtoms: ', otherAtoms)

        # weightDict = {'single':1, 'aromatic': 1.5, 'double':2, 'triple': 3}

        for j in otherAtoms: 
            
            atom1, atom2 = graph.nodes[nodeNumber]['element'], graph.nodes[j+1]['element']
            bo = load_data.get_bond_order(atom1, atom2, bondVector[j], tol)

            if bo != 0:
                if len(set([(nodeNumber, j+1), (j+1, nodeNumber)]).intersection(edgeAttr)) == 0:
                    edgeAttr[(nodeNumber, j+1)] = {'bo': bo, 'r': bondVector[j]}
                bondElec = bondElec + 2 * bo
                bondED = bondED + 1
        valElec = load_data.get_valence(graph.nodes[nodeNumber]['element'])
        elecDom = math.ceil(bondED + 0.5 * (valElec - 0.5 * bondElec - graph.nodes[nodeNumber]['charge'])) 
        # elecDom = round(bondED + 0.5 * (valElec - 0.5 * bondElec - graph.nodes[nodeNumber]['charge'])) 
        # print('elecDom', elecDom)

        graph.nodes[nodeNumber]['ed'] = elecDom

        # setting the UFF atom types
        if graph.nodes[nodeNumber]['element'] in ['F', 'Cl', 'Br', 'I']:
            _status = False
            if len(graph.nodes[nodeNumber]['element']) == 1:
                _status = True
            
            if _status:
                at = graph.nodes[nodeNumber]['element'] + '_' 
            else:
                at = graph.nodes[nodeNumber]['element']
            
            graph.nodes[nodeNumber]['at'] = at
        elif graph.nodes[nodeNumber]['element'] in ['O', 'C', 'N']:
            at = graph.nodes[nodeNumber]['element'] + '_' + str(elecDom - 1)
            # need to fix if aromatic, do this later with aromaticity
            graph.nodes[nodeNumber]['at'] = at
        elif graph.nodes[nodeNumber]['element'] == 'S':
            if elecDom == 4:
                neighDict = Counter([graph.nodes[x]['element'] for x in graph.neighbors(nodeNumber)])
                if neighDict['O'] == 2:
                    at = 'S_3+4'
                elif neighDict['O'] == 3:
                    at = 'S_3+6'
                else:
                    at = 'S_3+2'
            else:
                at = 'S_' + str(elecDom - 1)
            graph.nodes[nodeNumber]['at'] = at
        elif graph.nodes[nodeNumber]['element'] == 'P':
            graph.nodes[nodeNumber]['at'] = 'P_3+3'
            
    graph.add_edges_from([k for k, _ in edgeAttr.items()])
    nx.set_edge_attributes(graph, edgeAttr)

    return graph

def check_hybrid(graph): #check for hybridisation of oxygen, sulfur and nitrogen 

    oddCases = ['N', 'O', 'S'] # odd cases associated with nitrogen, oxygen and sulfur for conjugated systems
    oddElemList = [x for x,y in graph.nodes(data=True) if y['element'] in oddCases and y['ed'] == 4] 
    if len(oddElemList) == 0:
        return graph
    else:
    # we need to look over the hybridisation of oxygens, nitrogens, and sulfurs as they can exhibit sp2 hybridisation despite electron domain  = 4, need to corrrect electron domain
    # e.g. pyrrole, thiophene and furan

        for element in oddElemList:
            # non hydrogen neighbours
            haNeighbours = [x for x in graph.neighbors(element) if graph.nodes[x]['element'] != 'H']

            # checking every 2 combinations of neighbours
            if len(haNeighbours) >= 2:
                for comb in combinations(haNeighbours, 2):
                    if graph.nodes[comb[0]]['ed'] < 4 and graph.nodes[comb[1]]['ed'] < 4:
                        graph.nodes[element]['ed'] = 3
                    
        # nitrogen (oxygen) has a special case like in formamide (anisole) etc. where it needs to look at a branch of sp/sp2 atoms
        # also considering the halogens where pi back donation is present
        noList = [x for x in graph.nodes() if graph.nodes[x]['element'] == 'N' and graph.nodes[x]['ed'] == 4] + [x for x in graph.nodes() if graph.nodes[x]['element'] == 'O' and graph.nodes[x]['ed'] == 4] + [x for x in graph.nodes() if graph.nodes[x]['element'] == 'F' and graph.nodes[x]['ed'] == 4] + [x for x in graph.nodes() if graph.nodes[x]['element'] == 'Cl' and graph.nodes[x]['ed'] == 4] + [x for x in graph.nodes() if graph.nodes[x]['element'] == 'Br' and graph.nodes[x]['ed'] == 4] + [x for x in graph.nodes() if graph.nodes[x]['element'] == 'I' and graph.nodes[x]['ed'] == 4]
        for node in noList:
            secondNeigh = list(nx.dfs_edges(graph, source=node, depth_limit=2))
            del secondNeigh[0]
            
            for pairs in secondNeigh:
                hybridList = []
                hybridList.append(graph.nodes[pairs[0]]['ed'])
                hybridList.append(graph.nodes[pairs[1]]['ed'])

                if all(np.array(hybridList) < 4):
                    graph.nodes[node]['ed'] = 3
                    break

        return graph

def get_edges_of_node(node_label, edges_list):
    edgeList = [x for x in edges_list if node_label in x]
    return edgeList

def conjugate_region(graph): # returns set of nodes(atoms) whose electrons are conjugated together
    edgeList = [e for e in graph.edges]
    unsaturated_edges_list = [x for x in edgeList if 1 < graph.nodes[x[0]]['ed'] < 4 and 1 < graph.nodes[x[1]]['ed'] < 4] # getting edges with vertices that exhibit sp2 or sp hybridization 
    # print('unsaturated_edges_list', unsaturated_edges_list)
    # create a (sub)graph using these edges
    subNodeList = set(chain(*unsaturated_edges_list))

    # subNodeList = set(chain(*unsaturated_edges_list))       
    # print('unsaturated_edges_list redefined', unsaturated_edges_list)

    sg = nx.Graph()
    sg.add_nodes_from([x for x in subNodeList])
    # print('sg nodeList', [x for x in subNodeList])
    sg.add_edges_from(unsaturated_edges_list)
    # print('sg edgeList', unsaturated_edges_list)

    conjugated_nodes = []
    conjugated_edges = []
    connected_nodes = [x for x in nx.connected_components(sg)]
    # print('connected_nodes', connected_nodes)
    connected_nodes = [x for x in connected_nodes if len(x) > 2]
    for components in connected_nodes:
        nodeList = list(components)
        cnodesList = [x for x in nodeList if graph.nodes[x]['element'] == 'C']
        # print('connected nodeList', nodeList)
        # problematic nodes only accounts for allenes
        problematic_nodes = [] # list of nodes which contain double bonds on either side of it, scenario 1. one conjugated, the other isn't or 2. both are invovled in conjugation but are not in the same conjugation system
        # pnDict = {}
        for node in cnodesList:
            # print('node', node)
            neighbourList = [x for x in sg.neighbors(node)]
            # print('neighbourList', neighbourList)

            boList = np.array([graph[node][x]['bo'] for x in neighbourList])
            elemList = [graph.nodes[x]['element'] for x in neighbourList]
            indList = np.where(boList == 2)[0]
            if indList.size >= 2 and len(np.where(np.array([elemList[i] for i in indList]) == 'C')[0]) >= 2:
                problematic_nodes.append(node)

        # print('problematic_nodes', problematic_nodes)

        if len(problematic_nodes) == 0:
            conjugated_edges.append([e for e in graph.edges if e[0] in components and e[1] in components])
            conjugated_nodes.append(components)
        
        else: # applicable to allenes (chain and cyclic)
            terminal_nodes = [x for x in nodeList if sg.degree[x] == 1 and graph.nodes[x]['element'] == 'C']
            # print(terminal_nodes)
            pnode_tnode_comb_list = list(product(problematic_nodes, terminal_nodes))
            # print('pnode_tnode_comb_list', pnode_tnode_comb_list)
            for comb in pnode_tnode_comb_list:
                # print('comb', comb)
                pnode, tnode = comb[0], comb[1]
                other_pnode_list = [x for x in problematic_nodes if x != pnode]
                path = list(nx.all_simple_paths(sg, source=pnode, target=tnode)) 
                # print('path', path)
                pnode_in_path = [x for x in other_pnode_list if x in path[0]] # there should be only one list, hence the [0]
                if not pnode_in_path and len(path[0]) > 2: # there is no other problematic node in the path 
                    conjugated_edges.append([list(x) for x in map(nx.utils.pairwise, path)][0])
                    conjugated_nodes.append(path[0])
            
            if len(problematic_nodes) >= 2:
                paircombinations = combinations(problematic_nodes, 2)
                for pair in paircombinations:
                    # print('pair', pair)
                    nonpair_pnodes = [x for x in problematic_nodes if x != pair[0] and x != pair[1]]
                    pair_path = list(nx.all_simple_paths(sg, source=pair[0], target=pair[1]))
                    pair_path = sorted(pair_path, key=len)
                    # print('pair_path', pair_path)
                    pnode_in_pairpath = [x for x in nonpair_pnodes if x in pair_path[0]]
                    if not pnode_in_pairpath and len(pair_path[0]) > 2:
                        conjugated_edges.append([list(x) for x in map(nx.utils.pairwise, pair_path)][0])
                        conjugated_nodes.append(pair_path[0])

    for e in [x for x in graph.edges]:
        if e in miscellaneous.flatten(conjugated_edges) or e[::-1] in miscellaneous.flatten(conjugated_edges):
            graph[e[0]][e[1]]['conjugated'] = 'yes'
        else:
            graph[e[0]][e[1]]['conjugated'] = 'no'
    return conjugated_nodes, conjugated_edges, graph # conjugated edges returned will be mutually exclusive, that is, no edge in one list in conjugated_edges will appear in another list in conjugated_edges

def update_graph_pi(graph): # adds the no. of pi electrons (participating in conjugation) to each node
    t1 = time.process_time()
    conjugated_nodes, conjugated_edges, graph = conjugate_region(graph)
    print('     obtaining conjugate region time: ', time.process_time() - t1)
    t2 = time.process_time()
    pool = mp.Pool(mp.cpu_count())
    node_pi_list = pool.starmap_async(miscellaneous.get_pi_elec, [(conjugated_nodes[x], conjugated_edges[x], graph) for x in range(len(conjugated_edges)) ]).get()
    pool.close()
    # print(node_pi_list)
    node_pi_list = list(set(miscellaneous.flatten(node_pi_list)))

    for node_pi in node_pi_list:
        graph.nodes[node_pi[0]]['pi'] = node_pi[1]

    print('     evaluating pi electrons: ', time.process_time() - t2)
    return graph, conjugated_nodes, conjugated_edges


def main(graph, coordinates):
    # graph.add_nodes_from(nodeList)

    # distance matrix
    t1 = time.process_time()
    # matrix = load_data.proxMat(atoms, coordinates)
    matrix = load_data.EDM(np.array(coordinates), np.array(coordinates))
    # print(matrix)
    print('     proximity matrix time: ', time.process_time() - t1)

    # updating graph 
    t2 = time.process_time()
    graph = hybridisation2(graph, matrix) # adds electron domain as node attribute, adds edges (bond order as attribute)
    print('     hybridisation2 time: ', time.process_time() - t2)

    t3 = time.process_time()
    graph = check_hybrid(graph) # double checks the electron domains of oxygens and nitrogens in tricky cases (e.g. furan, pyrrole and formamide)
    print('     check hybridisation time: ', time.process_time() - t3)

    # graph = branching(graph)

    t4 = time.process_time()
    graph, _, conjugated_edges = update_graph_pi(graph) # adds pi electrons (participating in conjugation) as node attribute, retrieves the edges involved in a conjugated system
    print('     update graph pi time: ', time.process_time() - t4)

    return graph, conjugated_edges, matrix#, small_aromatic_cycles#, cycle_edge_list