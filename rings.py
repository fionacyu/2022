import networkx as nx

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
    for edge in nx.utils.pairwise(path):
        # Toggle whether to keep the current edge.
        edges ^= {edge}
    return edges

def _min_cycle(G, orth, weight=None):
    """
    Computes the minimum weight cycle in G,
    orthogonal to the vector orth as per [p. 338, 1]
    """
    T = nx.Graph()

    nodes_idx = {node: idx for idx, node in enumerate(G.nodes())}
    idx_nodes = {idx: node for node, idx in nodes_idx.items()}

    nnodes = len(nodes_idx)

    # Add 2 copies of each edge in G to T. If edge is in orth, add cross edge;
    # otherwise in-plane edge
    for u, v, data in G.edges(data=True):
        uidx, vidx = nodes_idx[u], nodes_idx[v]
        edge_w = data.get(weight, 1)
        if frozenset((u, v)) in orth:
            T.add_edges_from(
                [(uidx, nnodes + vidx), (nnodes + uidx, vidx)], weight=edge_w
            )
        else:
            T.add_edges_from(
                [(uidx, vidx), (nnodes + uidx, nnodes + vidx)], weight=edge_w
            )

    all_shortest_pathlens = dict(nx.shortest_path_length(T, weight=weight))
    cross_paths_w_lens = {
        n: all_shortest_pathlens[n][nnodes + n] for n in range(nnodes)
    }

    # Now compute shortest paths in T, which translates to cyles in G
    start = min(cross_paths_w_lens, key=cross_paths_w_lens.get)
    end = nnodes + start
    min_path = nx.shortest_path(T, source=start, target=end, weight="weight")

    # Now we obtain the actual path, re-map nodes in T to those in G
    min_path_nodes = [node if node < nnodes else node - nnodes for node in min_path]
    # Now remove the edges that occur two times
    mcycle_pruned = _path_to_cycle(min_path_nodes)

    return {frozenset((idx_nodes[u], idx_nodes[v])) for u, v in mcycle_pruned}

def _min_cycle_basis(comp, weight):
    cb = []
    # We  extract the edges not in a spanning tree. We do not really need a
    # *minimum* spanning tree. That is why we call the next function with
    # weight=None. Depending on implementation, it may be faster as well
    spanning_tree_edges = list(nx.minimum_spanning_edges(comp, weight=None, data=False))
    # edges_excl = [frozenset(e) for e in comp.edges() if e not in spanning_tree_edges]
    edges_excl = [frozenset(e) for e in list(set(comp.edges()) - set(spanning_tree_edges))]
    N = len(edges_excl)

    # We maintain a set of vectors orthogonal to sofar found cycles
    set_orth = [{edge} for edge in edges_excl]
    for k in range(N):
        # kth cycle is "parallel" to kth vector in set_orth
        new_cycle = _min_cycle(comp, set_orth[k], weight=weight)
        # print('list(set().union(*new_cycle))', [tuple(list(x)) for x in new_cycle])
        cb.append([tuple(list(x)) for x in new_cycle])
        # now update set_orth so that k+1,k+2... th elements are
        # orthogonal to the newly found cycle, as per [p. 336, 1]
        base = set_orth[k]
        set_orth[k + 1 :] = [
            orth ^ base if len(orth & new_cycle) % 2 else orth
            for orth in set_orth[k + 1 :]
        ]
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