import rings
import miscellaneous
import boxing
import load_data
import networkx as nx
from itertools import chain
from itertools import product
import multiprocessing as mp
import time

# defining the donor and acceptor objects
# each donor and acceptor will be labelled with dn (donor) or an (acceptor) where n is an integer
# e.g. d1 represents a donor, a90 represents an acceptor

class Donor:
    def __init__(self, name):
        self.name = name
        self.nodes = []
        self.edges = []
        self.terminal_nodes = []
        self.node_electrons = {}
        self.boxLabelList = []

    def add_nodes(self, nodeList):
        self.nodes.extend(nodeList)
    
    def add_edges(self, edgeList):
        self.edges.extend(edgeList)
    
    def add_terminal_nodes(self, tnodeList):
        self.terminal_nodes.extend(tnodeList)
    
    def add_classification(self, classification): # pi or sigma
        self.classification = classification
    # def add_electrons(self, electron_number): # number of electrons involved in the donation
    #     self.electrons = electron_number
    
    def add_electrons_dictionary(self, nodeList, electronList):
        for i, node in enumerate(nodeList):
            self.node_electrons[node] = electronList[i]

    def add_boxLabels(self, labelList):
        self.boxLabelList.extend(labelList)

    # to access the nodes of a donor, e.g. d1 = name of donor object, d1.nodes returns the list of nodes in donor d1

class Acceptor:
    def __init__(self, name):
        self.name = name
        self.nodes = []
        self.edges = []
        self.terminal_nodes = []
        self.boxLabelList = []

    def add_nodes(self, nodeList):
        self.nodes.extend(nodeList)
    
    def add_edges(self, edgeList):
        self.edges.extend(edgeList)
    
    def add_terminal_nodes(self, tnodeList):
        self.terminal_nodes.extend(tnodeList)

    def add_classification(self, classification):
        self.classification = classification

    def add_boxLabels(self, labelList):
        self.boxLabelList.extend(labelList)

class DonorAcceptorConnection:
    
    def add_simple_paths(self, edgeList):
        self.simple_paths = edgeList
    
    def add_bond_separation(self, bondseparation):
        self.bond_separation = bondseparation


def donor_acceptor_status_conj_nodes(graph, conjugated_edges):
    # acceptorList, donorList = [], []
    acceptorDict, donorDict = {}, {}
    for i, conj_system in enumerate(conjugated_edges):
        sg = nx.Graph()
        sg.add_nodes_from([x for x in set(chain(*conj_system))])
        sg.add_edges_from(conj_system)
        terminal_nodes = [x for x in sg.nodes() if sg.degree[x] == 1]
        # print('sg nodes', sg.nodes())
        
        donor = Donor('d%d' % i)
        donor.add_nodes([x for x in set(chain(*conj_system))]) 
        donor.add_edges(conj_system)
        donor.add_terminal_nodes(terminal_nodes)
        donor.add_classification('pi')
        donor.add_electrons_dictionary([x for x in sg.nodes()], [graph.nodes[x]['pi'] for x in sg.nodes()])
        # donor.add_electrons(sum([graph.nodes[x]['pi'] for x in sg.nodes()]))
        # donorList.append(donor)
        donor.add_boxLabels(list(dict.fromkeys([graph.nodes[y]['box'] for y in [x for x in set(chain(*conj_system))]])))
        donorDict['d%d' % i] = donor

        acceptor = Acceptor('a%d' % i)
        acceptor.add_nodes([x for x in set(chain(*conj_system))])
        acceptor.add_edges(conj_system)
        acceptor.add_terminal_nodes(terminal_nodes)
        acceptor.add_classification('pi')
        acceptor.add_boxLabels(list(dict.fromkeys([graph.nodes[y]['box'] for y in [x for x in set(chain(*conj_system))]])))
        # acceptorList.append(acceptor)                               
        acceptorDict['a%d' % i] = acceptor

    return donorDict, acceptorDict


def donor_acceptor_status_nonconj_nodes(graph, conjugated_edges):
    acceptorDict, donorDict = {}, {}
    conjugated_nodes = set(chain(*conjugated_edges))
    ncnodeList = [x for x in graph.nodes() if x not in conjugated_nodes]
    CNOnodesList = [x for x in ncnodeList if graph.nodes[x]['element'] == 'N' or graph.nodes[x]['element'] == 'C' or graph.nodes[x]['element'] == 'O'] 

    dcount = len(conjugated_edges)
    acount = len(conjugated_edges)
    for node in CNOnodesList:
        if graph.nodes[node]['element'] == 'C':
            if graph.nodes[node]['charge'] == -1 and graph.nodes[node]['ed'] == 3: # carbon anion
                donor = Donor('d%d' % dcount)
                donor.add_nodes([node])
                donor.add_terminal_nodes([node])
                donor.add_electrons_dictionary([node], [2])
                donor.add_classification('pi')
                donor.add_boxLabels([graph.nodes[node]['box']])
                donorDict['d%d' % dcount] = donor
                dcount += 1
                
            elif graph.nodes[node]['charge'] == 1 and graph.nodes[node]['ed'] == 3: # carbon cation
                acceptor = Acceptor('a%d' % acount)
                acceptor.add_nodes([node])
                acceptor.add_terminal_nodes([node])
                acceptor.add_classification('pi')
                acceptor.add_boxLabels([graph.nodes[node]['box']])
                acceptorDict['a%d' % acount] = acceptor
                acount += 1
                
        
        elif graph.nodes[node]['element'] == 'N':
            if graph.nodes[node]['ed'] == 4:
                donor = Donor('d%d' % dcount)
                donor.add_nodes([node])
                donor.add_terminal_nodes([node])
                donor.add_electrons_dictionary([node], [2])
                donor.add_classification('pi')
                donor.add_boxLabels([graph.nodes[node]['box']])
                donorDict['d%d' % dcount] = donor
                dcount += 1
                
        
        elif graph.nodes[node]['element'] == 'O':
            if graph.nodes[node]['ed'] == 4:
                donor = Donor('d%d' % dcount)
                donor.add_nodes([node])
                donor.add_terminal_nodes([node])
                donor.add_electrons_dictionary([node], [2])
                donor.add_classification('pi')
                donor.add_boxLabels([graph.nodes[node]['box']])
                donorDict['d%d' % dcount] = donor
                dcount += 1
                
    
    return donorDict, acceptorDict, dcount, acount

def donor_acceptor_status_nonconj_edges(graph, conjugated_edges, dcount_start, acount_start):
    acceptorDict, donorDict = {}, {}
    conj_edgeList = list(dict.fromkeys(miscellaneous.flatten(conjugated_edges))) + [x[::-1] for x in list(dict.fromkeys(miscellaneous.flatten(conjugated_edges)))] # add on the reversed because python thinks (1,2) and (2,1) are different
    # print('conj_edgeList: ', conj_edgeList)
    edgeList = [e for e in graph.edges if e not in conj_edgeList] # non conjugated edges
    # print('edgeList: ', edgeList)

    acount, dcount = acount_start, dcount_start
    # halogens = ['F', 'Cl', 'Br', 'I']
    halogenh = ['I', 'Br', 'Cl', 'F', 'H']
    donorCarbon, acceptorCarbon = [], []
    for edge in edgeList: 
        # print('edge', edge)
        # C - H sigma bond
        if (graph.nodes[edge[0]]['element'] == 'C' and graph.nodes[edge[1]]['element'] in halogenh and graph.nodes[edge[0]]['ed'] == 4 and len(set([edge[0]]).intersection(donorCarbon + acceptorCarbon)) == 0) or (graph.nodes[edge[0]]['element'] in halogenh and graph.nodes[edge[1]]['element'] == 'C' and graph.nodes[edge[1]]['ed'] == 4 and len(set([edge[1]]).intersection(donorCarbon + acceptorCarbon)) == 0):
            carbon_node = miscellaneous.node_of_element(graph, edge, 'C')
            # print('carbon_node', carbon_node)
        #     # if graph.nodes[carbon_node]['ed'] == 4:
            nodeNeighbours = [x for x in graph.neighbors(carbon_node) if graph.nodes[x]['element'] in halogenh]
            # print('nodeNeighbours', nodeNeighbours)
            elemNeighbours = [graph.nodes[x]['element'] for x in nodeNeighbours]
            # print('elemNeighbours', elemNeighbours)
            if 'H' in elemNeighbours:
                donor = Donor('d%d' % dcount)
                donor.add_nodes([carbon_node, nodeNeighbours[elemNeighbours.index('H')]])
                donor.add_edges([tuple(sorted((carbon_node, nodeNeighbours[elemNeighbours.index('H')])))])
                donor.add_terminal_nodes([carbon_node, nodeNeighbours[elemNeighbours.index('H')]])
                donor.add_electrons_dictionary([carbon_node, nodeNeighbours[elemNeighbours.index('H')]], [1,1])
                donor.add_classification('sigma')
                donor.add_boxLabels(list(dict.fromkeys([graph.nodes[carbon_node]['box'], graph.nodes[nodeNeighbours[elemNeighbours.index('H')]]['box']])))
                donorDict['d%d' % dcount] = donor
                dcount += 1
                donorCarbon.append(carbon_node)
            
            vdwradii = [load_data.get_radii(x) for x in elemNeighbours]
            sortedNeighbours = [x for x, _ in sorted(zip(nodeNeighbours, vdwradii), key= lambda pair: pair[1], reverse=True)]
            if sortedNeighbours:
                acceptor = Acceptor('a%d' % acount)
                acceptor.add_nodes([carbon_node, sortedNeighbours[0]])
                acceptor.add_edges([tuple(sorted((carbon_node, sortedNeighbours[0])))])
                acceptor.add_terminal_nodes([carbon_node, sortedNeighbours[0]])
                acceptor.add_classification('sigma')
                acceptor.add_boxLabels(list(dict.fromkeys([graph.nodes[carbon_node]['box'], graph.nodes[sortedNeighbours[0]]['box']])))
                acceptorDict['a%d' % acount] = acceptor
                acount +=1 
                acceptorCarbon.append(carbon_node)

        # C=C and C≡C bonds
        elif graph.nodes[edge[0]]['element'] == 'C' and graph.nodes[edge[1]]['element'] == 'C':
            if  graph[edge[0]][edge[1]]['bo'] == 2 or  graph[edge[0]][edge[1]]['bo'] == 3:
                donor = Donor('d%d' % dcount)
                donor.add_nodes([x for x in edge])
                donor.add_edges([edge])
                donor.add_terminal_nodes([x for x in edge])
                donor.add_electrons_dictionary([x for x in edge], [1,1])
                donor.add_classification('pi')
                donor.add_boxLabels(list(dict.fromkeys([graph.nodes[edge[0]]['box'], graph.nodes[edge[1]]['box']])))
                donorDict['d%d' % dcount] = donor
                dcount += 1

                acceptor = Acceptor('a%d' % acount)
                acceptor.add_nodes([x for x in edge])
                acceptor.add_edges([edge])
                acceptor.add_terminal_nodes([x for x in edge])
                acceptor.add_classification('pi')
                acceptor.add_boxLabels(list(dict.fromkeys([graph.nodes[edge[0]]['box'], graph.nodes[edge[1]]['box']])))
                acceptorDict['a%d' % acount] = acceptor
                acount += 1

        # C=O bonds
        elif (graph.nodes[edge[0]]['element'] == 'C' and graph.nodes[edge[1]]['element'] == 'O') or (graph.nodes[edge[0]]['element'] == 'O' and graph.nodes[edge[1]]['element'] == 'C'):
            if graph[edge[0]][edge[1]]['bo'] == 2:
                acceptor = Acceptor('a%d' % acount)
                acceptor.add_nodes([x for x in edge])
                acceptor.add_edges([edge])
                acceptor.add_terminal_nodes([x for x in edge])
                acceptor.add_classification('pi')
                acceptor.add_boxLabels(list(dict.fromkeys([graph.nodes[edge[0]]['box'], graph.nodes[edge[1]]['box']])))
                acceptorDict['a%d' % acount] = acceptor
                acount += 1
    
    return donorDict, acceptorDict

def donor_acceptor_connections(graph, donorDict, acceptorDict):
    connectionDict = {}
    da_comb_list = list(product([donorDict[k].name for k, _ in donorDict.items()], [acceptorDict[k].name for k, _ in acceptorDict.items()])) # need to remove ones which are the same
    rejected_combinations = [x for x in da_comb_list if donorDict[x[0]].nodes == acceptorDict[x[1]].nodes and donorDict[x[0]].edges == acceptorDict[x[1]].edges and donorDict[x[0]].terminal_nodes == acceptorDict[x[1]].terminal_nodes] + [x for x in da_comb_list if donorDict[x[0]].classification == acceptorDict[x[1]].classification]
    da_comb_list = list(set(da_comb_list) - set(rejected_combinations))
    print('     da comb before boxing', len(da_comb_list))
    t1 = time.process_time()
    pool = mp.Pool(mp.cpu_count())
    da_comb_list = pool.starmap_async(boxing.adjacent_da, [(da, donorDict, acceptorDict) for da in da_comb_list]).get()
    pool.close()
    print('     da comb time', time.process_time() - t1)
    print('     da comb after boxing', len([x for x in filter(None, da_comb_list)]))

    g = {n: set(graph.neighbors(n)) for n in graph.nodes()}

    pool = mp.Pool(mp.cpu_count())
    results = pool.starmap_async(miscellaneous.hyperconj_connections_para, [(g, comb, donorDict, acceptorDict) for comb in filter(None, da_comb_list)]).get()
    pool.close()

    connectionDict = dict([x for x in filter(lambda x: x[1] != None, results)])

    # for comb in filter(None, da_comb_list): #serial implementation
    #     donorLabel = comb[0]
    #     acceptorLabel = comb[1]

    #     donor_terminal_nodes = donorDict[donorLabel].terminal_nodes
    #     acceptor_terminal_nodes = acceptorDict[acceptorLabel].terminal_nodes

    #     terminal_comb_list = list(product(donor_terminal_nodes, acceptor_terminal_nodes))
    #     t2 = time.process_time()
    #     for tcomb in terminal_comb_list:
    #         # paths = list(nx.all_simple_paths(graph, source=tcomb[0], target=tcomb[1], cutoff=3)) #algo scales O(V+E)
    #         path = rings.shortest_path(g, tcomb[0], tcomb[1], 3)

    #         if len(path) > 0:
    #             if len(path) == 2: # bond separation is 1 bc the number of bonds == len(path) - 1
    #                 daConnection = DonorAcceptorConnection()
    #                 daConnection.add_simple_paths([tuple(sorted((path[i], path[i+1]))) for i in range(len(path)-1)])
    #                 daConnection.add_bond_separation(len(path)-1)
    #                 connectionDict[comb] = daConnection
    #                 break
    #             if comb in connectionDict:
    #                 if len(path) - 1 < connectionDict[comb].bond_separation:
    #                     connectionDict[comb].add_simple_paths([tuple(sorted((path[i], path[i+1]))) for i in range(len(path)-1)])
    #                     connectionDict[comb].add_bond_separation(len(path)-1)
    #                     connectionDict[comb] = daConnection
    #             elif comb not in connectionDict:
    #                 daConnection = DonorAcceptorConnection()
    #                 daConnection.add_simple_paths([tuple(sorted((path[i], path[i+1]))) for i in range(len(path)-1)])
    #                 daConnection.add_bond_separation(len(path)-1)
    #                 connectionDict[comb] = daConnection
    # path_time += time.process_time() -t2
    return connectionDict
            

def classify_donor_acceptor_connections(graph, conjugated_edges):
    donorDict, acceptorDict = {}, {}

    t1 = time.process_time()
    dlist, alist = donor_acceptor_status_conj_nodes(graph, conjugated_edges)
    print('da status conj systems time', time.process_time() - t1)
    donorDict.update(dlist)
    acceptorDict.update(alist)
    del dlist, alist

    t2 = time.process_time()
    dlist, alist, dcount, acount = donor_acceptor_status_nonconj_nodes(graph, conjugated_edges)
    print('da status nonconj nodes', time.process_time() -t2)
    donorDict.update(dlist)
    acceptorDict.update(alist)
    del dlist, alist

    t3 = time.process_time()
    dlist, alist = donor_acceptor_status_nonconj_edges(graph, conjugated_edges, dcount, acount)
    print('da status nonconj edges', time.process_time() - t3)
    donorDict.update(dlist)
    acceptorDict.update(alist)
    del dlist, alist

    t4 = time.process_time()
    connectionDict = donor_acceptor_connections(graph, donorDict, acceptorDict)
    print('da connections time', time.process_time() - t4)

    return donorDict, acceptorDict, connectionDict
