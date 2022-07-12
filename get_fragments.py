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
from collections import Counter
import os
import json
import sys
# import pygad
import pygadFY
import peff
from multiprocessing import Pool
mp.set_start_method('fork')

os.system('rm *.dat')
os.system('rm *.json')
os.system('rm -r fragxyz')
penNames = ['bo', 'aromaticity', 'penergy',  'conjugation', 'hyperconjugation', 'volume']


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
xyz_str = load_data.xyz_to_str(xyzFile)

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
# print('conjugated_edges', conjugated_edges)
# print('graph_characterisation time: ', time.process_time() - t1)

prmDict = load_data.read_prm()
betalist = [1,1,1,1,1,1]

def fitness_function(solution, solution_idx):
    solution1 = np.array(solution)
    # print(' '.join(str(j) for j in solution1),file=open('positions.dat', "a"))
    edges_to_cut_list = optimize.convert_bvector_edges(solution1, feasible_edges)
    
    # need to multiply by -1 because GA only accepts maximization functions
    penalty, edge_dij = - calculate_penalty.full_penalty_ga(solution1, atoms, G, edges_to_cut_list, conjugated_edges, donorDict, acceptorDict, connectionDict, aromaticDict, betalist, proxMatrix, desAtomNo, E, prmDict)
    # print(penalty)
    return round(penalty, 4)


def fitness_wrapper(solution):
    return fitness_function(solution, 0)

# '''connected components:'''
connected_sg = [G.subgraph(x) for x in nx.connected_components(G)]

idcount = 1
symbolList, coordList, idList, brokenBonds = [], [], [], []
hfragDict, fragNodes = {}, {}
for i, sg in enumerate(connected_sg):
    
    print('comp%d atoms: ' % (i+1), len(set(sg.nodes)))
    
    if len(sg.nodes) > desAtomNo:
        os.system('mkdir %d' % i)
        print(("%-20s " * len(penNames)) % tuple([str(i) for i in penNames]), file=open('penalties.dat', "a"))
        conjugated_edges_sg = [x for x in conjugated_edges if any([y in set(sg.nodes) for y in set(miscellaneous.flatten(x))  ])]
        print('conjugated_edges_sg', conjugated_edges_sg)
        cycleDict = rings.edgeList_dictionary(sg)
        cycleDict = boxing.classify_cycles(sg, cycleDict)
        aromaticDict, sg = aromaticity.classify_aromatic_systems(sg, conjugated_edges_sg, coordinates, cycleDict)
        donorDict, acceptorDict, connectionDict = hyperconj.classify_donor_acceptor_connections(sg, conjugated_edges_sg)
        xyz_str = peff.conv_graph_xyzstr(sg)
        t11 = time.process_time()
        E = peff.molecule_energy(xyz_str)
        print('E time', time.process_time() - t11)
        feasible_edges = optimize.get_feasible_edges(sg)
        fedges_idx = {edge: idx for idx, edge in enumerate(feasible_edges)}
        # print('feasible_edges', feasible_edges)
        print('\n'.join(str(i) for i in feasible_edges), file=open('%d/feasibleEdges_%d.dat' % (i, i), "a"))
        dim = len(feasible_edges)

        graph = sg

        inputpos = input('provide fragmentation bit vector file path / no) ')

        if inputpos == 'no':
            def fitness_function_sg(solution, solution_idx):
                solution1 = np.array(solution)
                # print('solution1', solution1)
                # print(' '.join(str(j) for j in solution1),file=open('positions.dat', "a"))
                edges_to_cut_list = optimize.convert_bvector_edges(solution1, feasible_edges)
                # for node in list(sg.nodes):
                #     print(node, sg.nodes[node])
                # need to multiply by -1 because GA only accepts maximization functions
                penalty, edge_dij = calculate_penalty.full_penalty_ga(solution1, atoms, sg, edges_to_cut_list, conjugated_edges_sg, donorDict, acceptorDict, connectionDict, aromaticDict, betalist, proxMatrix, desAtomNo, E, prmDict)
                penalty = - penalty
                # print(penalty)
                return round(penalty, 4), edge_dij, fedges_idx

            def fitness_wrapper_sg(solution):
                return fitness_function_sg(solution, 0)

            class PooledGA_SG(pygadFY.GA):
            # def __init__(self):
                best_fitness = -5.0
                #     self.best_pos = np.array([])

                def cal_pop_fitness(self):
                    global pool
                    
                    pop_fitness = pool.map(fitness_wrapper_sg, self.population)
                    pop_fitness = np.array(pop_fitness)
                    edge_dij = pop_fitness[:,1]
                    pop_fitness =  pop_fitness[:,0]

        
                    max_value = np.max(pop_fitness)
                    max_value_idx = np.argmax(pop_fitness)
                    
                    if max_value > self.best_fitness:
                        self.best_fitness = max_value
                        self.best_pos = np.array(self.population[max_value_idx])
                    
                    # print([round(x,4) for x in pop_fitness])
                    print('best fitness: ', self.best_fitness)
                    
                    return pop_fitness, edge_dij, fedges_idx

            init_pop = np.zeros((mp.cpu_count(), dim), dtype=int)
            idxfirst = []
            idxsecond = []
            for idx in range(mp.cpu_count()):
                second = np.r_[idx:dim:mp.cpu_count()+1]
                first = np.array([idx] * len(second))

                idxfirst.append(first)
                idxsecond.append(second)

            idx_first = np.concatenate(idxfirst, axis=0)
            idx_second = np.concatenate(idxsecond, axis=0)

            ii = (idx_first, idx_second)
            init_pop[ii] = 1

            start_time = time.time()
            ga_instance_sg = PooledGA_SG(num_generations=1000,
                                    initial_population=init_pop,
                                    num_parents_mating=2,
                                    # sol_per_pop=8,
                                    # num_genes=dim,
                                    fitness_func=fitness_function_sg,

                                    init_range_low=0,
                                    init_range_high=2,

                                    random_mutation_min_val=0,
                                    random_mutation_max_val=2,

                                    mutation_by_replacement=True,
                                    parent_selection_type="tournament",
                                    crossover_type="single_point",
                                    stop_criteria="reach_-1",

                                    gene_type=int)

            with Pool(processes=10) as pool:
                ga_instance_sg.run()

                # solution, solution_fitness, solution_idx = ga_instance.best_solution()
                print ('------------------------------------', file=open('%d/report_%d.log' % (i, i), 'a'))
                print("Parameters of the best solution : {solution}".format(solution=ga_instance_sg.best_pos))
                print("Parameters of the best solution : {solution}".format(solution=ga_instance_sg.best_pos), file=open('%d/report_%d.log' % (i, i), 'a'))
                print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=-1 * ga_instance_sg.best_fitness))
                print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=-1 * ga_instance_sg.best_fitness), file=open('%d/report_%d.log' % (i,i) , 'a'))
                # pos = np.array(solution)
                print('optimal edges to cut: ', optimize.convert_bvector_edges(ga_instance_sg.best_pos, feasible_edges))
                print('optimal edges to cut: ', optimize.convert_bvector_edges(ga_instance_sg.best_pos, feasible_edges), file=open('%d/report_%d.log' % (i, i), 'a'))
                end_time = time.time()
                print("--- %s seconds ---" % (end_time - start_time))
                print("--- %s seconds ---" % (end_time - start_time), file=open('%d/report_%d.log' % (i, i), 'a'))
                print ('------------------------------------', file=open('%d/report_%d.log' % (i, i), 'a'))
            pos = ga_instance_sg.best_pos
        else:
            with open(inputpos, 'r+') as f:
                data = f.read()
            pos = np.array(data.split(), dtype=int)
        symbolList1, coordList1, idList1, hfragDict1, fragNodes1, count = miscellaneous.get_fragments_sg(sg, optimize.convert_bvector_edges(pos, feasible_edges), idcount)
        symbolList.extend(symbolList1)
        coordList.extend(coordList1)
        idList.extend(idList1)
        hfragDict.update(hfragDict1)
        fragNodes.update(fragNodes1)
        brokenBonds.extend(miscellaneous.check_brokenbonds(sg, optimize.convert_bvector_edges(pos, feasible_edges)))
        idcount = count + 1
        os.system('mv positions.dat %d/positions_%d.dat' % (i, i))
        os.system('mv penalties.dat %d/penalties_%d.dat' % (i, i))
    
    else:
        symbolList1, coordList1, idList1, hfragDict1, fragNodes1, count = miscellaneous.get_fragments_sg(sg, [], idcount)
        symbolList.extend(symbolList1)
        coordList.extend(coordList1)
        idList.extend(idList1)
        hfragDict.update(hfragDict1)
        fragNodes.update(fragNodes1)
        idcount = count + 1
        

molDict = {'molecule': {'fragments': {'nfrag': len(Counter(idList)), 'fragid': idList, 'fragment_charges': [0 for _ in range(len(Counter(idList)))], 'broken_bonds': miscellaneous.flatten(brokenBonds) }, 'symbols': symbolList, 'geometry': coordList}, 'driver': 'energy', 'model': {'method': 'RHF', 'fragmentation': True, 'basis': '6-31G', 'aux_basis': 'cc-pVDZ'}, 'keywords': {'scf': {'niter': 50, 'ndiis': 10, 'dele': 1e-5, 'rmsd': 1e-6, 'dynamic_threshold': 10, 'debug': False, 'convergence_metric': 'diis'}, 'frag': {'method': 'MBE', 'level': 2, "ngpus_per_group": 8, 'dimer_cutoff': 40, 'trimer_cutoff': 10000}, 'rimp2': {'box_dim': 15}}}
# print(molDict, file=open('frag.json', 'a'))
with open('frag.json', 'w') as fp:
    json.dump(molDict, fp, indent=4)
# print('hfragDict', hfragDict)
miscellaneous.fragment_xyz(symbolList, coordList, idList, G, coordinates, hfragDict, fragNodes)
###########

# t3 = time.process_time()
# cycleDict = rings.edgeList_dictionary(G)
# # print('defining rings time: ', time.process_time() - t3)

# t4 = time.process_time()
# cycleDict = boxing.classify_cycles(G, cycleDict)
# # print('ring classification boxes time: ', time.process_time() - t4)
# # print('cycleDict', cycleDict)

# t5 = time.process_time()
# aromaticDict, G = aromaticity.classify_aromatic_systems(G, conjugated_edges, coordinates, cycleDict)
# # print('aromaticity classification time: ', time.process_time() - t5)

# t6 = time.process_time()
# donorDict, acceptorDict, connectionDict = hyperconj.classify_donor_acceptor_connections(G, conjugated_edges)

# # defining boxes
# t7 = time.process_time()
# donorDict, acceptorDict, aromaticDict = boxing.all_classification(G, donorDict, acceptorDict, cycleDict, aromaticDict) 

# # for node in list(G.nodes):
# #     print(node, G.nodes[node])

# # for edge in G.edges(data=True):
# #     print(edge)


# t11 = time.process_time()
# # E = uff.total_energy(G)
# E = peff.molecule_energy(xyz_str)
# print('E time', time.process_time() - t11)
# # peff_penalty = calculate_penalty.peff_penalty3(G, edges_to_cut_list, E, prmDict)
# # print('peff_penalty3', peff_penalty)
# # t8 = time.process_time()

# # total_penalty = calculate_penalty.full_penalty(atoms, G, edges_to_cut_list, conjugated_edges, donorDict, acceptorDict, connectionDict, aromaticDict, cycleDict, betalist, proxMatrix, minAtomNo)
# # print('total_penalty', total_penalty)
# # print('total penalty time', time.process_time() - t8)

# feasible_edges = optimize.get_feasible_edges(G)
# # print('feasible_edges: ', feasible_edges)
# print('\n'.join(str(i) for i in feasible_edges), file=open('feasibleEdges.dat', "a"))

# # mbe2wcs = calculate_penalty.peff_wcs(G, feasible_edges, E)
# dim = len(feasible_edges)
# # pos = optimize.run_optimizer(atoms, G, feasible_edges, conjugated_edges, donorDict, acceptorDict, connectionDict, aromaticDict, betalist, proxMatrix, desAtomNo, dim, E, prmDict)
# # pos = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0])
# # pos = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0])

# class PooledGA(pygad.GA):
#     # def __init__(self):
#         best_fitness = -5.0
#         #     self.best_pos = np.array([])

#         def cal_pop_fitness(self):
#             global pool
            
#             pop_fitness = pool.map(fitness_wrapper, self.population)
#             pop_fitness = np.array(pop_fitness)
#             max_value = np.max(pop_fitness)
#             max_value_idx = np.argmax(pop_fitness)
            
#             if max_value > self.best_fitness:
#                 self.best_fitness = max_value
#                 self.best_pos = np.array(self.population[max_value_idx])
            
#             print([round(x,4) for x in pop_fitness])
            
#             return pop_fitness

# start_time = time.time()
# ga_instance = PooledGA(num_generations=1000,
#                         num_parents_mating=2,
#                         sol_per_pop=8,
#                         num_genes=dim,
#                         fitness_func=fitness_function,

#                         init_range_low=0,
#                         init_range_high=2,

#                         random_mutation_min_val=0,
#                         random_mutation_max_val=2,

#                         mutation_by_replacement=True,
#                         parent_selection_type="tournament",
#                         crossover_type="single_point",

#                         gene_type=int)

# with Pool(processes=10) as pool:
#     ga_instance.run()

#     # solution, solution_fitness, solution_idx = ga_instance.best_solution()
#     print ('------------------------------------', file=open('report.log', 'a'))
#     print("Parameters of the best solution : {solution}".format(solution=ga_instance.best_pos))
#     print("Parameters of the best solution : {solution}".format(solution=ga_instance.best_pos), file=open('report.log', 'a'))
#     print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=-1 * ga_instance.best_fitness))
#     print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=-1 * ga_instance.best_fitness), file=open('report.log', 'a'))
#     # pos = np.array(solution)
#     print('optimal edges to cut: ', optimize.convert_bvector_edges(ga_instance.best_pos, feasible_edges))
#     print('optimal edges to cut: ', optimize.convert_bvector_edges(ga_instance.best_pos, feasible_edges), file=open('report.log', 'a'))
#     end_time = time.time()
#     print("--- %s seconds ---" % (end_time - start_time))
#     print("--- %s seconds ---" % (end_time - start_time), file=open('report.log', 'a'))
#     print ('------------------------------------', file=open('report.log', 'a'))
# pos = ga_instance.best_pos


# symbolList, coordList, weightList, idList, hfragDict, fragNodes = miscellaneous.get_fragments(G,  optimize.convert_bvector_edges(pos, feasible_edges), coordinates)
# broken_bonds = miscellaneous.check_brokenbonds(G, optimize.convert_bvector_edges(pos, feasible_edges))
# # print('symbolList', symbolList)
# # print('coordList', coordList)
# # print('weightList', weightList)
# # print('idList', idList)
# smallestfrags = []
# for count, weight in enumerate(weightList):
#     if weight == -1:
#         smallestfrags.append(idList[count])
# smallestfrags =  set(smallestfrags)
# print(','.join(str(x) for x in smallestfrags), file=open('smallestfrag.dat', 'a'))

# # molDict = {'molecule': {'fragments': {'nfrag': len(Counter(idList)), 'fragid': idList, 'fragment_charges': [0 for _ in range(len(Counter(idList)))], 'weights': weightList, 'broken_bonds': [] }, 'symbols': symbolList, 'geometry': coordList}, 'driver': 'energy', 'model': {'method': 'RHF', 'fragmentation': True, 'basis': '6-31G', 'aux_basis': 'cc-pVDZ'}, 'keywords': {'scf': {'niter': 50, 'ndiis': 10, 'dele': 1e-5, 'rmsd': 1e-6, 'dynamic_threshold': 10, 'debug': False, 'convergence_metric': 'diis'}, 'frag': {'method': 'MBE', 'level': 2, "ngpus_per_group": 8, 'dimer_cutoff': 40, 'trimer_cutoff': 10000}, 'rimp2': {'box_dim': 15}}}
# molDict = {'molecule': {'fragments': {'nfrag': len(Counter(idList)), 'fragid': idList, 'fragment_charges': [0 for _ in range(len(Counter(idList)))], 'broken_bonds': miscellaneous.flatten(broken_bonds) }, 'symbols': symbolList, 'geometry': coordList}, 'driver': 'energy', 'model': {'method': 'RHF', 'fragmentation': True, 'basis': '6-31G', 'aux_basis': 'cc-pVDZ'}, 'keywords': {'scf': {'niter': 50, 'ndiis': 10, 'dele': 1e-5, 'rmsd': 1e-6, 'dynamic_threshold': 10, 'debug': False, 'convergence_metric': 'diis'}, 'frag': {'method': 'MBE', 'level': 2, "ngpus_per_group": 8, 'dimer_cutoff': 40, 'trimer_cutoff': 10000}, 'rimp2': {'box_dim': 15}}}
# # print(molDict, file=open('frag.json', 'a'))
# with open('frag.json', 'w') as fp:
#     json.dump(molDict, fp, indent=4)
# miscellaneous.fragment_xyz(symbolList, coordList, idList, G, coordinates, hfragDict, fragNodes)
