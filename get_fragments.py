import graph_characterisation
import load_data
import miscellaneous
import calculate_penalty
import hyperconj
import aromaticity
import rings
import boxing
import optimize
import argparse
import networkx as nx
import numpy as np
import time
import multiprocessing as mp
import os

mp.set_start_method('fork')

os.system('rm *.dat')
os.system('rm -r fragxyz')
penNames = ['bo', 'aromaticity', 'ring', 'branching', 'hybridisation', 'conjugation', 'hyperconjugation', 'volume']
print(("%-20s " * len(penNames)) % tuple([str(i) for i in penNames]), file=open('penalties.dat', "a"))

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True)
# parser.add_argument('--xyz', required=True) # provide xyz file
# parser.add_argument('--charges', required=False) # provide file with charges of each atom, if not provided, assumes the atoms are neutral
args = parser.parse_args()
# xyzFile = args.xyz
inputPath = args.input
xyzFile, chargefile, minAtomNo = load_data.read_input(inputPath)
# if charge file is supplied

t1 = time.process_time()

if not chargefile:
#     chargeFile = args.charges
    atoms, coordinates, nodeList = load_data.read_xyz(xyzFile)
#     atoms, coordinates, nodeList = load_data.read_xyz(xyzFile, chargeFile)
else:
    atoms, coordinates, nodeList = load_data.read_xyz(xyzFile, chargefile)

# construct graph of molecule 
t2 = time.process_time()
G = nx.Graph()
G = boxing.box_classification(coordinates, G, nodeList) # d parameter goes at the end of this function
print('boxing.box_classification time: ', time.process_time() - t2)

t1 = time.process_time()
G, conjugated_edges, proxMatrix = graph_characterisation.main(G, coordinates)
print('graph_characterisation time: ', time.process_time() - t1)

t3 = time.process_time()
cycleDict = rings.edgeList_dictionary(G)
print('defining rings time: ', time.process_time() - t3)

# print('rings ***')
# for cycle in cycleDict:
#     print(cycleDict[cycle].edgeList)
# print('***')

t4 = time.process_time()
cycleDict = boxing.classify_cycles(G, cycleDict)
print('ring classification boxes time: ', time.process_time() - t4)
# print('cycleDict', cycleDict)

t5 = time.process_time()
aromaticDict = aromaticity.classify_aromatic_systems(G, conjugated_edges, coordinates, cycleDict)
print('aromaticity classification time: ', time.process_time() - t5)

t6 = time.process_time()
donorDict, acceptorDict, connectionDict = hyperconj.classify_donor_acceptor_connections(G, conjugated_edges)
print('len of connectionDict', len(connectionDict))
print('hyerpconjugation classification time: ', time.process_time() - t6)

# for donor in donorDict:
#     print(donor, donorDict[donor].nodes)

# for acc in acceptorDict:
#     print(acc, acceptorDict[acc].nodes)

# for connection in connectionDict:
#     print(connection, 'bond separation', connectionDict[connection].simple_paths)

# defining boxes
t7 = time.process_time()
donorDict, acceptorDict, aromaticDict = boxing.all_classification(G, donorDict, acceptorDict, cycleDict, aromaticDict) 
print('boxing classification of donorDict, acceptorDict, aromaticDict time: ', time.process_time() - t7)
print('conjugated_edges', conjugated_edges)
# print('conjugated nodes', set(chain(*conjugated_edges[0])))

# for edge in G.edges(data=True):
#     print(edge)


# for node in list(G.nodes):
#     print(node, G.nodes[node])

# nonHedges = [e for e in G.edges if G.nodes[e[0]]['element'] != 'H' and G.nodes[e[1]]['element'] != 'H']
# np.random.RandomState(100)
# binaryList = np.random.randint(2,size=len(nonHedges))

# edges_to_cut_list = [e for i, e in enumerate(nonHedges) if binaryList[i] == 1]
# print('edges_to_cut_list', edges_to_cut_list)


# t = time.process_time()
# # penalty
# conj_penalty = calculate_penalty.conjugation_penalty(G, [x for x in edges_to_cut_list], conjugated_edges)
# print('conj_penalty', conj_penalty)
# print('conj_penalty time', time.process_time() - t)
# # aromaticity_penalty = calculate_penalty.aromaticity_penalty(G, [x for x in edges_to_cut_list])
# # print('aromaticity_penalty', aromaticity_penalty)
# tbo = time.process_time()
# bo_penalty = calculate_penalty.bond_order_penalty(G, [x for x in edges_to_cut_list])
# print('bond order penalty', bo_penalty)
# print('bo_penalty time', time.process_time() - tbo)

# tbranch = time.process_time()
# branch_penalty = calculate_penalty.branching_penalty(G, [x for x in edges_to_cut_list])
# print('branch_penalty', branch_penalty)
# print('branch_penalty time', time.process_time() - tbranch)

# thybrid = time.process_time()
# hybrid_penalty = calculate_penalty.hybridisation_penalty(G, [x for x in edges_to_cut_list])
# print('hybrid_penalty', hybrid_penalty)
# print('hybrid_penalty time', time.process_time() - thybrid)

# taroma = time.process_time()
# aromatic_penalty = calculate_penalty.aromaticity_penalty(G, aromaticDict, [x for x in edges_to_cut_list])
# print('aromatic_penalty', aromatic_penalty)
# print('aromatic_penalty time', time.process_time() - taroma)
# # for k,v in connectionDict.items():
# #     print(k)
# #     print('simple path edges: ', connectionDict[k].simple_paths)
# #     print('bond separation: ', connectionDict[k].bond_separation)
# tring = time.process_time()
# ring_penalty = calculate_penalty.ring_penalty(G, cycleDict, edges_to_cut_list)
# print('ring_penalty', ring_penalty)
# print('ring penalty time', time.process_time() - tring)

# vol_penalty = calculate_penalty.volume_penalty(atoms, G, edges_to_cut_list, proxMatrix, minAtomNo)
# print('vol_penalty', vol_penalty)

# thyper = time.process_time()
# hyper_penalty = calculate_penalty.hyperconjugation_penalty(donorDict, acceptorDict, connectionDict, edges_to_cut_list)
# print('hyper penalty', hyper_penalty)
# print('hyper penalty time', time.process_time() - thyper)
# elapsed_time = time.process_time() - t
# print('penalty time: ', elapsed_time)
# final_time = time.process_time() - t1
# print('total time: ', final_time)

# t8 = time.process_time()
# minAtomNo = np.random.randint(low=5, high=15, size=1)[0]
# betalist = [1,1,1.3,1,1,1.6,1.6,0.8]
betalist = [1,1,1,1,1,1,1,1]
# total_penalty = calculate_penalty.full_penalty(atoms, G, edges_to_cut_list, conjugated_edges, donorDict, acceptorDict, connectionDict, aromaticDict, cycleDict, betalist, proxMatrix, minAtomNo)
# print('total_penalty', total_penalty)
# print('total penalty time', time.process_time() - t8)

feasible_edges = optimize.get_feasible_edges(G)
print('\n'.join(str(i) for i in feasible_edges), file=open('feasibleEdges.dat', "a"))
dim = len(feasible_edges)
pos = optimize.run_optimizer(atoms, G, feasible_edges, conjugated_edges, donorDict, acceptorDict, connectionDict, aromaticDict, cycleDict, betalist, proxMatrix, minAtomNo, dim)
# print('optimal edges to cut: ', optimize.convert_bvector_edges(pos, feasible_edges))

symbolList, coordList, weightList, idList = miscellaneous.get_fragments(G,  optimize.convert_bvector_edges(pos, feasible_edges), coordinates)
miscellaneous.fragment_xyz(symbolList, coordList, idList)