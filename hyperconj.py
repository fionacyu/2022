import miscellaneous
import boxing
import networkx as nx
from itertools import chain
from itertools import product
import multiprocessing as mp


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
    halogens = ['F', 'Cl', 'Br', 'I']
    for edge in edgeList: 
        # print('edge', edge)
        # C - H sigma bond
        if (graph.nodes[edge[0]]['element'] == 'C' and graph.nodes[edge[1]]['element'] == 'H') or (graph.nodes[edge[0]]['element'] == 'H' and graph.nodes[edge[1]]['element'] == 'C'):
            # hybridization of carbon must be 4
            carbon_node = miscellaneous.node_of_element(graph, edge, 'C')
            if graph.nodes[carbon_node]['ed'] == 4:
                donor = Donor('d%d' % dcount)
                donor.add_nodes([x for x in edge])
                donor.add_edges([edge])
                donor.add_terminal_nodes([x for x in edge])
                donor.add_electrons_dictionary([x for x in edge], [1,1])
                donor.add_classification('sigma')
                donor.add_boxLabels(list(dict.fromkeys([graph.nodes[edge[0]]['box'], graph.nodes[edge[1]]['box']])))
                donorDict['d%d' % dcount] = donor
                dcount += 1
                

                acceptor = Acceptor('a%d' % acount)
                acceptor.add_nodes([x for x in edge])
                acceptor.add_edges([edge])
                acceptor.add_terminal_nodes([x for x in edge])
                acceptor.add_classification('sigma')
                acceptor.add_boxLabels(list(dict.fromkeys([graph.nodes[edge[0]]['box'], graph.nodes[edge[1]]['box']])))
                acceptorDict['a%d' % acount] = acceptor
                acount +=1 

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
        
        # C-F, C-Cl, C-Br, C-I bonds
        elif (graph.nodes[edge[0]]['element'] == 'C' and graph.nodes[edge[1]]['element'] in halogens) or (graph.nodes[edge[0]]['element'] in halogens and graph.nodes[edge[1]]['element'] == 'C'):
            acceptor = Acceptor('a%d' % acount)
            acceptor.add_nodes([x for x in edge])
            acceptor.add_edges([edge])
            acceptor.add_terminal_nodes([x for x in edge])
            acceptor.add_classification('sigma')
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
    # print('donor: ', [k for k, _ in donorDict.items()])
    # print('acceptor: ', [k for k, _ in acceptorDict.items()])

    da_comb_list = list(product([donorDict[k].name for k, _ in donorDict.items()], [acceptorDict[k].name for k, _ in acceptorDict.items()])) # need to remove ones which are the same
    # print('     da_comb_list', len(da_comb_list))#, da_comb_list)
    rejected_combinations = [x for x in da_comb_list if donorDict[x[0]].nodes == acceptorDict[x[1]].nodes and donorDict[x[0]].edges == acceptorDict[x[1]].edges and donorDict[x[0]].terminal_nodes == acceptorDict[x[1]].terminal_nodes] + [x for x in da_comb_list if donorDict[x[0]].classification == acceptorDict[x[1]].classification]
    # print('     rejected comb', len(rejected_combinations))
    # da_comb_list = [x for x in da_comb_list if x not in rejected_combinations] # removes donors and acceptor pairings which correspond to the same group and donor acceptor pairings of the same classification (sigma or pi)
    da_comb_list = list(set(da_comb_list) - set(rejected_combinations))
    # print('     removing rejected combinartions', len(da_comb_list))
    # da_comb_list = [x for x in da_comb_list if boxing.adjacent_status_da(donorDict[x[0]].boxLabelList, acceptorDict[x[1]].boxLabelList, boxDict)] # the boxes that donor and acceptors belong in are neighbours
    pool = mp.Pool(mp.cpu_count())
    da_comb_list = pool.starmap_async(boxing.adjacent_da, [(da, donorDict, acceptorDict) for da in da_comb_list]).get()
    pool.close()
    # print('     da connections boxing', len(da_comb_list))
    # print('da_comb_list', da_comb_list)

    # count = 0 
    for comb in filter(None, da_comb_list):
        donorLabel = comb[0]
        acceptorLabel = comb[1]

        donor_terminal_nodes = donorDict[donorLabel].terminal_nodes
        acceptor_terminal_nodes = acceptorDict[acceptorLabel].terminal_nodes

        terminal_comb_list = list(product(donor_terminal_nodes, acceptor_terminal_nodes))
        potential_path_lists, potential_path_edges_list = [], [] # lol
        potential_tcomb_list = [] # lol
        for tcomb in terminal_comb_list:
            paths = list(nx.all_simple_paths(graph, source=tcomb[0], target=tcomb[1], cutoff=3)) #algo scales O(V+E)

            if len(paths) > 0:
                potential_path_lists.append(paths)
                potential_path_edges_list.append([list(x) for x in map(nx.utils.pairwise, paths)][0])
                potential_tcomb_list.append(tcomb)
        
        # print('potential_path_edges_list', potential_path_edges_list)

        if len(potential_tcomb_list) > 0: # there is a connection between the donor comb[0] and acceptor comb[1]
            sorted_potential_path_lists = [sorted(x, key=len)[0] for x in potential_path_lists] # getting the shortest path from donor to acceptor
            new_sorted_path_lists = sorted(sorted_potential_path_lists, key=len)
            indx = sorted_potential_path_lists.index(new_sorted_path_lists[0])
            daConnection = DonorAcceptorConnection()
            daConnection.add_simple_paths(potential_path_edges_list[indx])
            daConnection.add_bond_separation(len(potential_path_edges_list[indx]))
            connectionDict[comb] = daConnection
    
    return connectionDict
            

def classify_donor_acceptor_connections(graph, conjugated_edges):
    donorDict, acceptorDict = {}, {}

    dlist, alist = donor_acceptor_status_conj_nodes(graph, conjugated_edges)
    print('da status conj systems')
    donorDict.update(dlist)
    acceptorDict.update(alist)
    del dlist, alist

    dlist, alist, dcount, acount = donor_acceptor_status_nonconj_nodes(graph, conjugated_edges)
    print('da status nonconj nodes')
    donorDict.update(dlist)
    acceptorDict.update(alist)
    del dlist, alist

    dlist, alist = donor_acceptor_status_nonconj_edges(graph, conjugated_edges, dcount, acount)
    print('da status nonconj edges')
    donorDict.update(dlist)
    acceptorDict.update(alist)
    del dlist, alist

    connectionDict = donor_acceptor_connections(graph, donorDict, acceptorDict)

    return donorDict, acceptorDict, connectionDict
