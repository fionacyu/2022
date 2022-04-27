import calculate_penalty
import pyswarms as ps
import numpy as np
import multiprocessing as mp

def get_feasible_edges(graph): # edges that do not include hydrogen 
    return [e for e in graph.edges if graph.nodes[e[0]]['element'] != 'H' and graph.nodes[e[1]]['element'] != 'H']

def convert_bvector_edges(bvector, feasible_edges):
    mask = bvector == 1
    indList = np.where(mask)[0]
    edges_to_cut_list = [feasible_edges[i] for i in indList]
    return edges_to_cut_list

def run_optimizer(atoms, graph, feasible_edges, conjugated_edges, donorDict, acceptorDict, connectionDict, aromaticDict, cycleDict, betalist, proxMatrix, minAtomNo, dim):
    kwargs = {'atoms': atoms, 'feasible_edges': feasible_edges, 'graph': graph, 'conjugated_edges': conjugated_edges, 'donorDict': donorDict, 'acceptorDict': acceptorDict, 'connectionDict': connectionDict, 'aromaticDict': aromaticDict, 'cycleDict': cycleDict, 'betalist': betalist, 'proxMatrix': proxMatrix, 'minAtomNo': minAtomNo}
    options = {'c1': 0.5, 'c2': 0.5, 'w': 0.9, 'k':9, 'p': 1}
    optimizer = ps.discrete.binary.BinaryPSO(n_particles=10, dimensions=dim, options=options)
    _, pos = optimizer.optimize(calculate_penalty.full_penalty_opt, iters=1000, n_processes=None , **kwargs)
    return pos