import graph_characterisation
import load_data
import miscellaneous
import calculate_penalty
import hyperconj
import aromaticity
import rings
import boxing
import optimize
import uff
import argparse
import networkx as nx
import numpy as np
import time
import multiprocessing as mp
from collections import Counter
import os
import json
import sys
import pygad
from multiprocessing import Pool
mp.set_start_method('fork')

os.system('rm *.dat')
os.system('rm -r fragxyz')
penNames = ['bo', 'aromaticity', 'penergy',  'conjugation', 'hyperconjugation', 'volume']
print(("%-20s " * len(penNames)) % tuple([str(i) for i in penNames]), file=open('penalties.dat', "a"))

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True)
# parser.add_argument('--xyz', required=True) # provide xyz file
# parser.add_argument('--charges', required=False) # provide file with charges of each atom, if not provided, assumes the atoms are neutral
args = parser.parse_args()
# xyzFile = args.xyz
inputPath = args.input
xyzFile, chargefile, desAtomNo = load_data.read_input(inputPath)
# if charge file is supplied

t1 = time.process_time()

if not chargefile:
#     chargeFile = args.charges
    atoms, coordinates, nodeList = load_data.read_xyz(xyzFile)
#     atoms, coordinates, nodeList = load_data.read_xyz(xyzFile, chargeFile)
else:
    atoms, coordinates, nodeList = load_data.read_xyz(xyzFile, chargefile)

print('atom no: ', len(atoms))


if len(atoms) <= desAtomNo:
    print('Number of atoms in system is less than or equal to fragSize. Please consider leaving the system as one fragment or reducing fragSize.')
    sys.exit()

# construct graph of molecule 
t2 = time.process_time()
G = nx.Graph()
G = boxing.box_classification(coordinates, G, nodeList) # d parameter goes at the end of this function

# print('boxing.box_classification time: ', time.process_time() - t2)

t1 = time.process_time()
G, conjugated_edges, proxMatrix = graph_characterisation.main(G, coordinates)
# for edge in G.edges(data=True):
#     print(edge)
# print('graph_characterisation time: ', time.process_time() - t1)

t3 = time.process_time()
cycleDict = rings.edgeList_dictionary(G)
# print('defining rings time: ', time.process_time() - t3)

# print('rings ***')
# for cycle in cycleDict:
#     print(cycleDict[cycle].edgeList)
# print('***')

t4 = time.process_time()
cycleDict = boxing.classify_cycles(G, cycleDict)
# print('ring classification boxes time: ', time.process_time() - t4)
# print('cycleDict', cycleDict)

t5 = time.process_time()
aromaticDict, G = aromaticity.classify_aromatic_systems(G, conjugated_edges, coordinates, cycleDict)
# print('aromaticity classification time: ', time.process_time() - t5)

t6 = time.process_time()
donorDict, acceptorDict, connectionDict = hyperconj.classify_donor_acceptor_connections(G, conjugated_edges)
# print('len of connectionDict', len(connectionDict))
# print('hyerpconjugation classification time: ', time.process_time() - t6)

# for donor in donorDict:
#     print(donor, donorDict[donor].nodes)

# for acc in acceptorDict:
#     print(acc, acceptorDict[acc].nodes)

# for connection in connectionDict:
#     print(connection, 'bond separation', connectionDict[connection].simple_paths)

# defining boxes
t7 = time.process_time()
donorDict, acceptorDict, aromaticDict = boxing.all_classification(G, donorDict, acceptorDict, cycleDict, aromaticDict) 
# print('boxing classification of donorDict, acceptorDict, aromaticDict time: ', time.process_time() - t7)
# print('conjugated_edges', conjugated_edges)
# print('conjugated nodes', set(chain(*conjugated_edges[0])))




# for node in list(G.nodes):
#     print(node, G.nodes[node])

nonHedges = [e for e in G.edges if G.nodes[e[0]]['element'] != 'H' and G.nodes[e[1]]['element'] != 'H']
np.random.RandomState(100)
binaryList = np.random.randint(2,size=len(nonHedges))

edges_to_cut_list = [e for i, e in enumerate(nonHedges) if binaryList[i] == 1]
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

prmDict = load_data.read_prm()
t11 = time.process_time()
E = uff.total_energy(G)
print('E time', time.process_time() - t11)
# peff_penalty = calculate_penalty.peff_penalty3(G, edges_to_cut_list, E, prmDict)
# print('peff_penalty3', peff_penalty)
# t8 = time.process_time()
betalist = [1,1,1,1,1,1]
# total_penalty = calculate_penalty.full_penalty(atoms, G, edges_to_cut_list, conjugated_edges, donorDict, acceptorDict, connectionDict, aromaticDict, cycleDict, betalist, proxMatrix, minAtomNo)
# print('total_penalty', total_penalty)
# print('total penalty time', time.process_time() - t8)

feasible_edges = optimize.get_feasible_edges(G)
print('\n'.join(str(i) for i in feasible_edges), file=open('feasibleEdges.dat', "a"))

# mbe2wcs = calculate_penalty.peff_wcs(G, feasible_edges, E)
dim = len(feasible_edges)
# pos = optimize.run_optimizer(atoms, G, feasible_edges, conjugated_edges, donorDict, acceptorDict, connectionDict, aromaticDict, betalist, proxMatrix, desAtomNo, dim, E, prmDict)
# pos = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0])
# pos = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0])

def fitness_function(solution, solution_idx):
    solution1 = np.array(solution)
    # print(' '.join(str(j) for j in solution1),file=open('positions.dat', "a"))
    edges_to_cut_list = optimize.convert_bvector_edges(solution1, feasible_edges)

    # need to multiply by -1 because GA only accepts maximization functions
    penalty = - calculate_penalty.full_penalty_ga(solution1, atoms, G, edges_to_cut_list, conjugated_edges, donorDict, acceptorDict, connectionDict, aromaticDict, betalist, proxMatrix, desAtomNo, E, prmDict)
    # print(penalty)
    return round(penalty, 4)


def fitness_wrapper(solution):
    return fitness_function(solution, 0)


class PooledGA(pygad.GA):
    # def __init__(self):
    best_fitness = -5.0
    #     self.best_pos = np.array([])

    def cal_pop_fitness(self):
        global pool
        
        pop_fitness = pool.map(fitness_wrapper, self.population)
        pop_fitness = np.array(pop_fitness)
        max_value = np.max(pop_fitness)
        max_value_idx = np.argmax(pop_fitness)
        self.best_pos = np.array(self.population[max_value_idx])
        if max_value > self.best_fitness:
            self.best_fitness = max_value
        
        print([round(x,4) for x in pop_fitness])
        
        return pop_fitness

# ga_instance = pygad.GA(num_generations=1000,
#                         num_parents_mating=2,
#                         sol_per_pop=8,
#                         num_genes=desAtomNo,
#                         fitness_func=fitness_function,

#                         init_range_low=0,
#                         init_range_high=2,

#                         random_mutation_min_val=0,
#                         random_mutation_max_val=2,

#                         mutation_by_replacement=True,
#                         parent_selection_type="tournament",
#                         crossover_type="single_point",

#                         gene_type=int,
#                         )

# ga_instance.run()

# solution, solution_fitness, _ = ga_instance.best_solution(ga_instance.last_generation_fitness)
# print("Parameters of the best solution : {solution}".format(solution=solution))
# print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=  solution_fitness))
# pos = np.array(solution)
# print('optimal edges to cut: ', optimize.convert_bvector_edges(pos, feasible_edges))
start_time = time.time()
ga_instance = PooledGA(num_generations=1000,
                        num_parents_mating=2,
                        sol_per_pop=8,
                        num_genes=desAtomNo,
                        fitness_func=fitness_function,

                        init_range_low=0,
                        init_range_high=2,

                        random_mutation_min_val=0,
                        random_mutation_max_val=2,

                        mutation_by_replacement=True,
                        parent_selection_type="tournament",
                        crossover_type="single_point",

                        gene_type=int)

with Pool(processes=10) as pool:
    ga_instance.run()

    # solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=ga_instance.best_pos))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=-1 * ga_instance.best_fitness))
    # pos = np.array(solution)
    print('optimal edges to cut: ', optimize.convert_bvector_edges(ga_instance.best_pos, feasible_edges))
    print("--- %s seconds ---" % (time.time() - start_time))
pos = ga_instance.best_pos
# peff = calculate_penalty.peff_penalty(G, optimize.convert_bvector_edges(pos, feasible_edges), E)
# print('mbe2 original', peff)
# t8 = time.process_time()
# peffog = calculate_penalty.peff_penalty(G,  optimize.convert_bvector_edges(pos, feasible_edges), E)
# print('\t', time.process_time() - t8)
# t9 = time.process_time()
# peff3 = calculate_penalty.peff_penalty3(G,  optimize.convert_bvector_edges(pos, feasible_edges), E, prmDict)
# print('\t', time.process_time() - t9)
# t10 = time.process_time()
# peff4 = calculate_penalty.peff_penalty4(G,  optimize.convert_bvector_edges(pos, feasible_edges), prmDict)
# print('\t', time.process_time() - t10)
# peff2 = calculate_penalty.peff_penalty3(G,optimize.convert_bvector_edges(pos, feasible_edges), E, prmDict)
# print('mbe2 effective', peff2)

symbolList, coordList, weightList, idList, hfragDict, fragNodes = miscellaneous.get_fragments(G,  optimize.convert_bvector_edges(pos, feasible_edges), coordinates)
# print('symbolList', symbolList)
# print('coordList', coordList)
# print('weightList', weightList)
# print('idList', idList)
smallestfrags = []
for count, weight in enumerate(weightList):
    if weight == -1:
        smallestfrags.append(idList[count])
smallestfrags =  set(smallestfrags)
print(','.join(str(x) for x in smallestfrags), file=open('smallestfrag.dat', 'a'))

molDict = {'fragments': {'nfrag': len(Counter(idList)), 'fragid': idList, 'fragment_charges': [0 for _ in range(len(Counter(idList)))], 'weights': weightList, 'broken_bonds': [] }, 'symbols': symbolList, 'geometry': coordList}
# print(molDict, file=open('frag.json', 'a'))
with open('frag.json', 'w') as fp:
    json.dump(molDict, fp, indent=4)
miscellaneous.fragment_xyz(symbolList, coordList, idList, G, coordinates, hfragDict, fragNodes)


