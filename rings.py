import miscellaneous
import networkx as nx
import multiprocessing as mp
import time

class Cycle:
    def __init__(self):
        self.edgeList = []
        self.boxLabelList = []
    
    def add_edges(self, edgeList):
        self.edgeList.extend(edgeList)

    def add_boxLabels(self, labelList):
        self.boxLabelList.extend(labelList)

def _path_to_cycle(path):
    """
    Removes the edges from path that occur even number of times.
    Returns a set of edges
    """
    edges = set()
    # print('nx.utils.pairwise(path)', list(nx.utils.pairwise(path)))
    for edge in nx.utils.pairwise(path):
        # Toggle whether to keep the current edge.
        edges ^= {edge}
    return edges

def shortest_path(graph, node1, node2, cutoff):
    # BFS method, performs in linear time
    # taken from https://onestepcode.com/graph-shortest-path-python/?utm_source=rss&utm_medium=rss&utm_campaign=graph-shortest-path-python
    path_list = [[node1]]
    path_index = 0
    # To keep track of previously visited nodes
    previous_nodes = {node1}
    if node1 == node2:
        return path_list[0]
    
    dist = 1

    current_path = path_list[path_index]
    while path_index < len(path_list) and len(current_path) -1 < cutoff:
        current_path = path_list[path_index]
        last_node = current_path[-1]
        next_nodes = graph[last_node]
        # Search goal node
        # print('current_path', current_path)
        # print('dist', dist)
        if node2 in next_nodes:
            current_path.append(node2)
            return current_path
        # Add new paths
        for next_node in next_nodes:
            if not next_node in previous_nodes:
                new_path = current_path[:]
                new_path.append(next_node)
                path_list.append(new_path)
                # To avoid backtracking
                previous_nodes.add(next_node)
                dist += 1
        # Continue to next path in list
        path_index += 1
    # No path is found
    return []

def _min_cycle(G, orth, weight=None):
    """
    Computes the minimum weight cycle in G,
    orthogonal to the vector orth as per [p. 338, 1]
    """
    enodes = set(miscellaneous.flatten([ list(nx.dfs_preorder_nodes(G, source=x, depth_limit=10) ) for x,_ in orth]) )
    # print('enodes', enodes)
    uvedges = [e for e in G.edges if len(set(e).intersection(enodes)) > 0]
    # t2 = time.process_time()
    T = nx.Graph()
    offset = min(G.nodes) - 1
    
    nodes_idx = {node: idx for idx, node in enumerate(G.nodes())}
    idx_nodes = {idx + offset: node for node, idx in nodes_idx.items()}

    nnodes = len(nodes_idx)
    # nnodes = max(G.nodes)
    
    # print('offset', offset)
    # Add 2 copies of each edge in G to T. If edge is in orth, add cross edge;
    # otherwise in-plane edge
    # edgeList =[[(nodes_idx[u], nnodes + nodes_idx[v]), (nnodes + nodes_idx[u], nodes_idx[v])] if frozenset((u, v)) in orth else [(nodes_idx[u], nodes_idx[v]), (nnodes + nodes_idx[u], nnodes + nodes_idx[v])] for u, v in uvedges] 
    edgeList =[[(nodes_idx[u] + offset, nnodes + nodes_idx[v] + offset), (nnodes + nodes_idx[u] + offset, nodes_idx[v] + offset)] if frozenset((u, v)) in orth else [(nodes_idx[u] + offset, nodes_idx[v] + offset), (nnodes + nodes_idx[u] + offset, nnodes + nodes_idx[v] + offset)] for u, v in uvedges] 
    T.add_edges_from(miscellaneous.flatten(edgeList))
    # print('edgeList', miscellaneous.flatten(edgeList))
    T = {n: set(T.neighbors(n)) for n in T.nodes}
    # print('T', T)
    
    t1 = time.process_time()
    pool = mp.Pool(mp.cpu_count())
    results = pool.starmap_async(miscellaneous.shortest_path_length, [(T, n-1, nnodes + n - 1) for n in enodes]).get()
    pool.close()
    # print('results', results)
    # print('shortest_path_length time', time.process_time() - t1)
    # print('shortest_path_length', results)
    start = min(results, key = lambda t:t[1])[0]
    # print('start', start)
    # Now compute shortest paths in T, which translates to cyles in G
    end = nnodes + start
    min_path = shortest_path(T, start, end, 10)
    # print('min_path', min_path)

    # Now we obtain the actual path, re-map nodes in T to those in G
    min_path_nodes = [node if node < offset + nnodes else node - nnodes for node in min_path]
    # print('min_path_nodes', min_path_nodes)
    # Now remove the edges that occur two times
    mcycle_pruned = _path_to_cycle(min_path_nodes)
    # print('mcycle_pruned', mcycle_pruned)
    # print('_min_cycle', time.process_time() - t2)
    # print('cycle_edges', {frozenset((idx_nodes[u], idx_nodes[v])) for u, v in mcycle_pruned})
    return {frozenset((idx_nodes[u], idx_nodes[v])) for u, v in mcycle_pruned}

def _min_cycle_basis(comp, weight):
    # t3 = time.process_time()
    cb = []
    # We  extract the edges not in a spanning tree. We do not really need a
    # *minimum* spanning tree. That is why we call the next function with
    # weight=None. Depending on implementation, it may be faster as well
    # t4 = time.process_time()
    spanning_tree_edges = list(nx.minimum_spanning_edges(comp, weight=None, data=False))
    # print('min spanning tree: ', time.process_time() - t4)
    # edges_excl = [frozenset(e) for e in comp.edges() if e not in spanning_tree_edges]
    edges_excl = [frozenset(e) for e in list(set(comp.edges()) - set(spanning_tree_edges))]
    N = len(edges_excl)
    # print('edges_excl: ', edges_excl)

    # We maintain a set of vectors orthogonal to sofar found cycles
    set_orth = [{edge} for edge in edges_excl]
    # t1 = time.process_time()
    for k in range(N):
        # kth cycle is "parallel" to kth vector in set_orth
        # t = time.process_time()
        # print('set_orth[k]', set_orth[k])
        new_cycle = _min_cycle(comp, set_orth[k], weight=weight)
        # print('new_cycle', new_cycle)
        # print('list(set().union(*new_cycle))', [tuple(list(x)) for x in new_cycle])
        cb.append([tuple(list(x)) for x in new_cycle])
        # now update set_orth so that k+1,k+2... th elements are
        # orthogonal to the newly found cycle, as per [p. 336, 1]
        base = set_orth[k]
        set_orth[k + 1 :] = [
            orth ^ base if len(orth & new_cycle) % 2 else orth
            for orth in set_orth[k + 1 :]
        ]
    # print('_min_cycle_basis', time.process_time() - t3)
    # print('_min_cycle_basis time: ', time.process_time() - t1)
    cb = list(filter(None, cb))
    # print('cb', cb)
    return cb

def minimum_cycle_basis(G, weight=None):
    # We first split the graph in commected subgraphs
    return sum(
        (_min_cycle_basis(G.subgraph(c), weight) for c in nx.connected_components(G)),
        [],
    )

def classify_cycles(cycle_edge_list):
    cycleDict = {}
    count = 0 

    for edgeList in cycle_edge_list:
        cycle = Cycle()
        cycle.add_edges(edgeList)
        cycleDict['c%d' % count] = cycle
        count += 1
    
    return cycleDict

def edgeList_dictionary(graph):
    cycle_edge_list = minimum_cycle_basis(graph)
    cycleDict = classify_cycles(cycle_edge_list)

    return cycleDict