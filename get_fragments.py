import graph_characterisation
import load_data
import calculate_penalty
import hyperconj
import aromaticity
import rings
import boxing
import argparse
import networkx as nx
from itertools import chain

parser = argparse.ArgumentParser()
parser.add_argument('--xyz', required=True) # provide xyz file
parser.add_argument('--charges', required=False) # provide file with charges of each atom, if not provided, assumes the atoms are neutral
args = parser.parse_args()
xyzFile = args.xyz

# if charge file is supplied
if args.charges:
    chargeFile = args.charges
    atoms, coordinates, nodeList = load_data.read_xyz(xyzFile, chargeFile)
else:
    atoms, coordinates, nodeList = load_data.read_xyz(xyzFile)

edges_to_cut = input("edges to cut (separated by space, e.g. 2,1 15,16 ...): ")
edges_to_cut_list = edges_to_cut.split()
# print(edges_to_cut_list)
edges_to_cut_list = [(int(x.split(',')[0]), int(x.split(',')[1])) for x in edges_to_cut_list]
# print(edges_to_cut_list)

# construct graph of molecule 
G = nx.Graph()
G, conjugated_edges = graph_characterisation.main(G, atoms, coordinates, nodeList)
G, boxDict = boxing.box_classification(coordinates, G) # d parameter goes at the end of this function

cycleDict = rings.edgeList_dictionary(G)
cycleDict = boxing.classify_cycles(G, cycleDict)
aromaticDict = aromaticity.classify_aromatic_systems(G, conjugated_edges, coordinates, cycleDict, boxDict)

donorDict, acceptorDict, connectionDict = hyperconj.classify_donor_acceptor_connections(G, conjugated_edges)



# defining boxes
donorDict, acceptorDict, aromaticDict = boxing.all_classification(G, donorDict, acceptorDict, cycleDict, aromaticDict) 




print('conjugated_edges', conjugated_edges)
# print('conjugated nodes', set(chain(*conjugated_edges[0])))

for edge in G.edges(data=True):
    print(edge)

# print('blah', [x for x in G.edges])

for node in list(G.nodes):
    print(node, G.nodes[node])



# print('minimum_cycle_basis', [c for c in rings.minimum_cycle_basis(G)])

# penalty
conj_penalty = calculate_penalty.conjugation_penalty(G, [x for x in edges_to_cut_list], conjugated_edges)
print('conj_penalty', conj_penalty)
# aromaticity_penalty = calculate_penalty.aromaticity_penalty(G, [x for x in edges_to_cut_list])
# print('aromaticity_penalty', aromaticity_penalty)
bo_penalty = calculate_penalty.bond_order_penalty(G, [x for x in edges_to_cut_list])
print('bond order penalty', bo_penalty)

branch_penalty = calculate_penalty.branching_penalty(G, [x for x in edges_to_cut_list])
print('branch_penalty', branch_penalty)

hybrid_penalty = calculate_penalty.hybridisation_penalty(G, [x for x in edges_to_cut_list])
print('hybrid_penalty', hybrid_penalty)



# print('donors')
# for k,v in donorDict.items():
#     print('name', donorDict[k].name)
#     print('nodes', donorDict[k].nodes)
#     print('edges', donorDict[k].edges)
#     print('terminal_nodes', donorDict[k].terminal_nodes)
#     print('node_electrons', donorDict[k].node_electrons)

# print('acceptorList')
# for k,v in acceptorDict.items():
#     print('name', acceptorDict[k].name)
#     print('nodes', acceptorDict[k].nodes)
#     print('edges', acceptorDict[k].edges)
#     print('terminal_nodes', acceptorDict[k].terminal_nodes)

hyperconj_penalty = calculate_penalty.hyperconjugation_penalty(G, donorDict, acceptorDict, connectionDict, [x for x in edges_to_cut_list], boxDict)
print('hyperconjugation_penalty', hyperconj_penalty)
# print(donorList)


aromatic_penalty = calculate_penalty.aromaticity_penalty(G, aromaticDict, [x for x in edges_to_cut_list], boxDict)
print('aromatic_penalty', aromatic_penalty)
# for k,v in connectionDict.items():
#     print(k)
#     print('simple path edges: ', connectionDict[k].simple_paths)
#     print('bond separation: ', connectionDict[k].bond_separation)

ring_penalty = calculate_penalty.ring_penalty(G, cycleDict, edges_to_cut_list, boxDict)
print('ring_penalty', ring_penalty)