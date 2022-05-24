import miscellaneous
import boxing
from itertools import chain
import networkx as nx
from itertools import product
import numpy as np

class aromatic_system:
    def __init__(self):
        self.cycle_list = []
        self.bridging_edges = []
        self.nonbridging_edges = []
        self.boxLabelList = []
    
    def add_cycle(self, cycle_list):
        self.cycle_list.extend(cycle_list)
    
    def add_bridging_edges(self, edgeList):
        self.bridging_edges.extend(edgeList)
    
    def add_nonbridging_edges(self, edgeList):
        self.nonbridging_edges.extend(edgeList)

    def add_size(self,size):
        self.size = size

    def add_boxLabels(self, labelList):
        self.boxLabelList.extend(labelList)

def get_as_from_edge(small_aromatic_cycles, edge): # returns list of list
    blist = zip([edge in x for x in small_aromatic_cycles], [edge[::-1] in x for x in small_aromatic_cycles])
    trueind = [i for i, x in enumerate(blist) if any(x) == True]
    acycles = [small_aromatic_cycles[i] for i in trueind]
    return acycles

def check_aromaticity(graph, conjugated_edges, coordinates, cycleDict): # returns list of lists of edges in an aromatic system, len(return[output]) = no. of aromatic systems there are
    if len(cycleDict) == 0:
        return []
    else:
        small_aromatic_cycles = []
        cycle_keys = [k for k in cycleDict]
        conjugated_indices = list(range(len(conjugated_edges)))
        prodList = list(product(cycle_keys,conjugated_indices)) # box edges needs to go here
        prodList = [x for x in prodList if boxing.adjacent_systems([graph.nodes[y]['box'] for y in list(set(chain(*conjugated_edges[x[1]])))], cycleDict[x[0]].boxLabelList) == 1]
        # print('prodList', prodList)
        for prod in prodList:
            # cycles = sorted(cycle_edge_list[prod[0]])
            cycles = list(set(chain(*cycleDict[prod[0]].edgeList)))
            # print('cycles', cycles)
            conjugated = sorted([x for x in set(chain(*conjugated_edges[prod[1]]))])
            # print('conjugated', conjugated)

            # check if the rings are conjugated
            if set(cycles).issubset(conjugated):
                electrons = sum([graph.nodes[x]['pi'] for x in cycles])
                # print('electrons', electrons)
                # check if huckel's rule is obeyed
                if electrons % 4 == 2: # huckel's rule: 4n + 2
                    # now need to check if it is planar

                    distanceList = [np.linalg.norm(np.array(coordinates[x -1])) for x in cycles]
                    # chooses the 3 points closest to the origin to form the basis of vectors for the plane 
                    # print(distanceList)
                    if len(distanceList) > 3:
                        idx = np.argpartition(distanceList, 3)
                    elif len(distanceList) == 3:
                        idx = [0,1,2]

                    u = np.array(coordinates[cycles[idx[0]] - 1])
                    v = np.array(coordinates[cycles[idx[1]] - 1])
                    w = np.array(coordinates[cycles[idx[2]] - 1])

                    uv = v - u 
                    uw = w - u 
                    
                    norm_vector = np.cross(uv, uw)
                    norm_vector = 1/np.linalg.norm(norm_vector) * norm_vector # normalising the normal vector 

                    # other_nodes = [x for x in cycles if x not in [cycles[idx[0]], cycles[idx[1]], cycles[idx[2]]]]
                    other_nodes = list(set(cycles) - set([cycles[idx[0]], cycles[idx[1]], cycles[idx[2]]]))
                    # print('other_nodes', other_nodes)

                    deviationList = []
                    for other_atoms in other_nodes:
                        deviation = np.linalg.norm(np.dot(norm_vector, np.array(coordinates[other_atoms - 1]) - u))
                        deviationList.append(deviation)
                    # print('deviationList', deviationList)
                    # check if planar, threshold here is 0.5 A
                    if all(x < 0.5 for x in deviationList):
                        # print('hello')
                        small_aromatic_cycles.append([x for x in conjugated_edges[prod[1]] if x[0] in cycles and x[1] in cycles])
                        # aromatic_edges.append([x for x in cycles])
        
        # print('small_aromatic_cycles', small_aromatic_cycles)
        return small_aromatic_cycles

def classify_aromatic_systems(graph, conjugated_edges, coordinates, cycleDict):
    small_aromatic_cycles = check_aromaticity(graph, conjugated_edges, coordinates, cycleDict)
    print('small_aromatic_cycles', small_aromatic_cycles)
    edgeList = miscellaneous.flatten(small_aromatic_cycles)
    nodeList = [x for x in set(chain(*miscellaneous.flatten(small_aromatic_cycles)))]

    agraph = nx.Graph()
    agraph.add_nodes_from(nodeList)
    agraph.add_edges_from(edgeList)

    connected_sgs = [agraph.subgraph(x) for x in nx.connected_components(agraph)]
    count = 0 
    aromaticDict = {}
    for sg in connected_sgs:
        
        aromaticsys = aromatic_system()
        aromaticsys.add_size(len(sg.edges))
        bedgeList = [e for e in sg.edges() if sum([e in x for x in small_aromatic_cycles]) >= 2]
        if len(bedgeList) > 0:
            cycleList = [asys for asys in small_aromatic_cycles if len(set(bedgeList).intersection(asys)) > 0]
            intersectionList = [list(set(bedgeList).intersection(asys)) for asys in small_aromatic_cycles]
            nonbe_cycleList = [list(set(cycle) - set(intersectionList[i])) for i, cycle in enumerate(cycleList)]
            aromaticsys.add_nonbridging_edges(nonbe_cycleList)
        else:
            cycleList = [[e for e in sg.edges]]
            aromaticsys.add_nonbridging_edges(cycleList)
        aromaticsys.add_cycle(cycleList)
        aromaticsys.add_bridging_edges(bedgeList)
        aromaticDict['aroma%d' % count] = aromaticsys
    
    return aromaticDict 

