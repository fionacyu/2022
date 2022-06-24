import calculate_penalty
import pyswarms as ps
import numpy as np
import collections
import multiprocessing as mp

def get_feasible_edges(graph): # edges that do not include hydrogen 
    return [e for e in graph.edges if graph.nodes[e[0]]['element'] != 'H' and graph.nodes[e[1]]['element'] != 'H' and graph[e[0]][e[1]]['bo'] == 1]

def convert_bvector_edges(bvector, feasible_edges):
    mask = bvector == 1
    indList = np.where(mask)[0]
    edges_to_cut_list = [feasible_edges[i] for i in indList]
    return edges_to_cut_list

def convert_bvector_edges1(bvector2d, feasible_edges):
    mask = bvector2d == 1
    indList1, indList2 = np.where(mask)[0], np.where(mask)[1]
    edges_to_cut_list = [[] for _ in range(bvector2d.shape[0])]
    for i in range(len(collections.Counter(indList1))):
        # print('i: ', i)
        if i == 0:
            start = 0
        else:
            start = sum([collections.Counter(indList1)[x] for x in range(i)])
        edges_to_cut_list[i].extend([feasible_edges[j] for j in indList2[start:start + collections.Counter(indList1)[i]]] )
    return edges_to_cut_list

def run_optimizer(atoms, graph, feasible_edges, conjugated_edges, donorDict, acceptorDict, connectionDict, aromaticDict, betalist, proxMatrix, minAtomNo, dim, E, prmDict):
    kwargs = {'feasible_edges': feasible_edges, 'atoms': atoms, 'graph': graph, 'conjugated_edges': conjugated_edges, 'donorDict': donorDict, 'acceptorDict': acceptorDict, 'connectionDict': connectionDict, 'aromaticDict': aromaticDict, 'betalist': betalist, 'proxMatrix': proxMatrix, 'minAtomNo': minAtomNo, 'E': E, 'prmDict': prmDict}
    options = {'c1': 0.5, 'c2': 0.5, 'w': 0.9, 'k':mp.cpu_count()-1, 'p': 1}
    # ipos = np.zeros(dim)
    # ipos[int(minAtomNo)-1:ipos.size:int(minAtomNo)]
    # ipos = np.tile(ipos, (mp.cpu_count(), 1))
    optimizer = ps.discrete.binary.BinaryPSO(n_particles=mp.cpu_count(), dimensions=dim, options=options)#, init_pos=ipos)
    _, pos = optimizer.optimize(calculate_penalty.full_penalty_opt, iters=1000, n_processes=None , **kwargs)
    return pos