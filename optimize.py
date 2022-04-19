import pyswarms as ps

def get_feasible_edges(graph): # edges that do not include hydrogen 
    return [e for e in graph.edges if graph.nodes[e[0]]['element'] == 'H' or graph.ndoes[e[1]]['element'] == 'H']

