import rings
import load_data
import math
import numpy as np 
import networkx as nx
from itertools import combinations
from itertools import chain
from itertools import product


def hybridisation2(graph, proxMatrix, tol=0.003):
    # returns the edge attributes
    # print(proxMatrix)
    # print('ha')
    # heavy atoms only, not concerned with hydrogens because they only form one single bond anyways
    ha = [x for x in list(graph.nodes) if graph.nodes[x]['element'] != 'H'] # contains integers which are node labels
    hlist = [x for x in list(graph.nodes) if graph.nodes[x]['element'] == 'H']

    for hatom in hlist:
        graph.nodes[hatom]['ed'] = 1 # electron domain of hydrogen is defaulted to 1
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

        weightDict = {'single':1, 'aromatic': 1.5, 'double':2, 'triple': 3}

        for j in otherAtoms: 
            
            atom1, atom2 = graph.nodes[nodeNumber]['element'], graph.nodes[j+1]['element']
            # print(atom1, atom2)
            # print(nodeNumber, j+1)
            rdf = load_data.retrieveBI(atom1, atom2)
            bondtype= rdf.bond_type.to_list()
            # print(bondtype)
            weightBT = [weightDict['%s' % x] for x in bondtype]
            # print(weightBT)

            sortedBT = [i for _, i in sorted(zip(weightBT, bondtype), reverse=True)]
            # print(sortedBT)

            for bt in sortedBT:
                min, max = load_data.get_range(rdf, bt)
                min, max = min - tol, max + tol
                # print(min, max)
                if min <= bondVector[j] <= max:
                    # print('loop is running', min, max)
                    bondtype = bt
                    # print(bondtype, atoms[j])
                    bondElec = bondElec + 2 * weightDict[bt]
                    bondED = bondED + 1
                    
                    if (nodeNumber, j+1) not in edgeAttr and (j+1, nodeNumber) not in edgeAttr:
                        edgeAttr[(nodeNumber, j+1)] = {'bo': weightDict[bt]}
                    break
                elif (bt == 'single' and bondVector[j] < min) or (bt == 'single' and bondVector[j] > max): #no covalent bond exists between atom1 and atom2
                    break
                else:
                    continue
        # print('bondED', bondED)
        # print('bondElec', bondElec)
        valElec = load_data.get_valence(graph.nodes[nodeNumber]['element'])
        elecDom = math.ceil(bondED + 0.5 * (valElec - 0.5 * bondElec - graph.nodes[nodeNumber]['charge'])) 
        # elecDom = round(bondED + 0.5 * (valElec - 0.5 * bondElec - graph.nodes[nodeNumber]['charge'])) 
        # print('elecDom', elecDom)
        graph.nodes[nodeNumber]['ed'] = elecDom
    # print(edgeAttr)
    
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
    for components in connected_nodes:

        nodeList = list(components)
        # print('connected nodeList', nodeList)
        # problematic nodes only accounts for allenes
        problematic_nodes = [] # list of nodes which contain double bonds on either side of it, scenario 1. one conjugated, the other isn't or 2. both are invovled in conjugation but are not in the same conjugation system
        # pnDict = {}
        for node in nodeList:
            # print('node', node)
            neighbourList = [x for x in sg.neighbors(node)]
            # print('neighbourList', neighbourList)
            combList = combinations(neighbourList, 2) # comparing two adjacent bonds

            for comb in list(combList):
                # print('comb', comb)
                boList = [graph[node][comb[0]]['bo'], graph[node][comb[1]]['bo']]
                # print('boList', boList)

                if boList[0] >=2 and boList[1] >= 2: # problem here
                    # print('problemmm')
                    problematic_nodes.append(node)

        # print('problematic_nodes', problematic_nodes)

        if len(problematic_nodes) == 0:
            conjugated_edges.append([e for e in graph.edges if e[0] in components and e[1] in components])
            conjugated_nodes.append(components)
        
        else: # applicable to allenes (chain and cyclic)
            terminal_nodes = [x for x in nodeList if sg.degree[x] == 1]
            # print(terminal_nodes)
            pnode_tnode_comb_list = list(product(problematic_nodes, terminal_nodes))
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
                    nonpair_pnodes = [x for x in problematic_nodes if x != pair[0] and x != pair[1]]
                    pair_path = list(nx.all_simple_paths(sg, source=pair[0], target=pair[1]))
                    pnode_in_pairpath = [x for x in nonpair_pnodes if x in pair_path[0]]
                    if not pnode_in_pairpath and len(pair_path[0]) > 2:
                        conjugated_edges.append([list(x) for x in map(nx.utils.pairwise, pair_path)][0])
                        conjugated_nodes.append(pair_path[0])

    for conj_path in conjugated_edges:
        for edge in conj_path:
            # print(edge)
            # print(graph[edge[0]][edge[1]])
            graph[edge[0]][edge[1]]['conjugated'] = 'yes'
    return conjugated_nodes, conjugated_edges # conjugated edges returned will be mutually exclusive, that is, no edge in one list in conjugated_edges will appear in another list in conjugated_edges

def update_graph_pi(graph): # adds the no. of pi electrons (participating in conjugation) to each node
    conjugated_nodes, conjugated_edges = conjugate_region(graph)
    # print('conjugated_nodes', conjugated_nodes)
    # print('conjugated_edges', conjugated_edges)
    for j, connections in enumerate(conjugated_nodes): 
        # print([x for x in connections])
        for i, n in enumerate([x for x in connections]):
            # print(n, graph.nodes[n]['element'])
            valence = load_data.get_valence(graph.nodes[n]['element'])
            # print('valence', valence)
            sigmaBonds = len([x for x in graph.neighbors(n)]) # number of sigma bonds
            # print('sigmaBonds', sigmaBonds)
            elecDom = graph.nodes[n]['ed']
            # print('elecDom', elecDom)
            piELec = valence - sigmaBonds - 2 * (elecDom - sigmaBonds)- graph.nodes[n]['charge'] # gives the number of pi electrons in the conjugated system, formula is essentially FC = V - N - B/2           
            if i == 0 or i == len([x for x in connections]) - 1:
                # need to check for the fact that the atom is connected to other pi systems not part of the conjugated system or separate conjugated systems
                edgeList = get_edges_of_node(n, [x for x in graph.edges if graph[x[0]][x[1]]['bo'] >= 2]) #edges the node is part of which is double bond or triple
                reject_edges = [x for x in edgeList if x in conjugated_edges[j] or x[::-1] in conjugated_edges[j]] 
                edgeList = [x for x in edgeList if x not in reject_edges] # non conjugated edges the node is bonded to
                # print('edgeList', edgeList)

                bo_diff_list = [graph[x[0]][x[1]]['bo'] - 1 for x in edgeList] # minus 1 because one bond will be sigma bond (we want the pi bond)
                # print('bo_diff_list', bo_diff_list)
                non_conj_pi_elec = sum(bo_diff_list)
                # print('non_conj_pi_elec', non_conj_pi_elec)

                graph.nodes[n]['pi'] = piELec - non_conj_pi_elec

            else:
                graph.nodes[n]['pi'] = piELec # tells us the number of pi electrons per atom, but doesn't distinguish between conjugated vs. non-conjugated systems

    
    return graph, conjugated_nodes, conjugated_edges


# def find_all_cycles(G, source=None, cycle_length_limit=None):
#     # returns list of lists where each list contains the nodes in a cycle, each adjacent node has an edge between them 
#     # taken from https://gist.github.com/joe-jordan/6548029
#     """forked from networkx dfs_edges function. Assumes nodes are integers, or at least
#     types which work with min() and > ."""
#     if source is None:
#         # produce edges for all components
#         nodes=[list(i)[0] for i in nx.connected_components(G)]
#     else:
#         # produce edges for components with source
#         nodes=[source]
#     # extra variables for cycle detection:
#     cycle_stack = []
#     output_cycles = set()
    
#     def get_hashable_cycle(cycle):
#         """cycle as a tuple in a deterministic order."""
#         m = min(cycle)
#         mi = cycle.index(m)
#         mi_plus_1 = mi + 1 if mi < len(cycle) - 1 else 0
#         if cycle[mi-1] > cycle[mi_plus_1]:
#             result = cycle[mi:] + cycle[:mi]
#         else:
#             result = list(reversed(cycle[:mi_plus_1])) + list(reversed(cycle[mi_plus_1:]))
#         return tuple(result)
    
#     for start in nodes:
#         if start in cycle_stack:
#             continue
#         cycle_stack.append(start)
        
#         stack = [(start,iter(G[start]))]
#         while stack:
#             _,children = stack[-1]
#             try:
#                 child = next(children)
                
#                 if child not in cycle_stack:
#                     cycle_stack.append(child)
#                     stack.append((child,iter(G[child])))
#                 else:
#                     i = cycle_stack.index(child)
#                     if i < len(cycle_stack) - 2: 
#                       output_cycles.add(get_hashable_cycle(cycle_stack[i:]))
                
#             except StopIteration:
#                 stack.pop()
#                 cycle_stack.pop()
    
#     return [list(i) for i in output_cycles]


# def update_edge_rings(graph, cycle_edge_list):
#     for edge_list in cycle_edge_list:
#         # print(edge_list)
#         ring_size = len(edge_list)
#         for i, edge in enumerate(edge_list):
#             # print(edge)
#             # if i == len(edge_list) - 1:
#             #     if 'ring' not in graph[edge_list[0]][edge_list[i]]:
#             #         graph[edge_list[0]][edge_list[i]]['ring'] = [ring_size]
#             #     elif 'ring' in graph[edge_list[0]][edge_list[i]]:
#             #         graph[edge_list[0]][edge_list[i]]['ring'].append(ring_size)
#             if 'ring' not in graph[edge[0]][edge[1]]:
#                 graph[edge[0]][edge[1]]['ring'] = [ring_size]
#             elif 'ring' in graph[edge[0]][edge[1]]:
#                 graph[edge[0]][edge[1]]['ring'].append(ring_size)
    
#     return graph


# def get_branching(graph): # considers alkane branching/protobranching, updates the node attribute of the graph with no. of protobranches
#     nodeList = [x for x in list(graph.nodes) if graph.nodes[x]['element'] == 'C' and graph.nodes[x]['ed'] == 4]
#     # print('***')
#     # print(nodeList)

#     for node in nodeList:
#         dfsList = list(nx.dfs_edges(graph, source=node, depth_limit=2))
#         # print(dfsList)
#         dfsList = [x[1] for x in dfsList if x[0] != node and x[1] != node and graph.nodes[x[1]]['element'] == 'C' and graph.nodes[x[1]]['ed'] == 4] # contains the nodes of other carbonds which are 2 bonds away 
#         # print(dfsList)
#         graph.nodes[node]['protobranches'] = len(dfsList)
    
#     return graph     



# def update_edge_aromaticity(graph, aromatic_edges):
#     for edge_list in aromatic_edges:
#         for edge in edge_list:
#             if 'aromatic' not in graph[edge[0]][edge[1]]:
#                 graph[edge[0]][edge[1]]['aromatic'] = 1
#             elif 'aromatic' in graph[edge[0]][edge[1]]:
#                 graph[edge[0]][edge[1]]['aromatic'] += 1
#     # the aromatic score assigned to each edge tells us how many aromatic systems the edge is part of
#     # therefore if you break an edge which is part of two aromatic systems, the penalty will be higher 
#     return graph

def main(graph, atoms, coordinates, nodeList):
    graph.add_nodes_from(nodeList)

    # distance matrix
    matrix = load_data.proxMat(atoms, coordinates)

    # updating graph 
    graph = hybridisation2(graph, matrix) # adds electron domain as node attribute, adds edges (bond order as attribute)
    graph = check_hybrid(graph) # double checks the electron domains of oxygens and nitrogens in tricky cases (e.g. furan, pyrrole and formamide)
    graph, _, conjugated_edges = update_graph_pi(graph) # adds pi electrons (participating in conjugation) as node attribute, retrieves the edges involved in a conjugated system
    # cycle_edge_list = rings.minimum_cycle_basis(graph) # list of lists
    # print('cycle_edge_list', cycle_edge_list)
    # print('cycle_edge_list', cycle_edge_list)
    # graph = update_edge_rings(graph, cycle_edge_list)
    # graph = get_branching(graph)
    # small_aromatic_cycles = check_aromaticity(graph, conjugated_edges, coordinates)
    # print('aromatic_edges', len(aromatic_edges))

    # graph = update_edge_aromaticity(graph, aromatic_edges)

    return graph, conjugated_edges#, small_aromatic_cycles#, cycle_edge_list