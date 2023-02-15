import copy
import networkx as nx

from Rule.traffic import *

def Rule_algorithm(graph: nx.Graph, traffic):
    fiber_graph = copy.deepcopy(graph)
    topo = nx.create_empty_copy(graph)
    traffic_path = {}
    load_graph = nx.Graph()
    for (node_i, node_j) in graph.edges:
        load_graph.add_edge(node_i, node_j, load=0)
    for f in traffic:
        src = f[0]
        dst = f[1]
        demand = traffic[f]
        # find two paths for flow
        path_1, path_2 = suurballe(fiber_graph, src, dst)
        path_1 = remove_cycle(path_1)
        path_2 = remove_cycle(path_2)

        if path_1 != [] and path_2 != []:
            for i in range(1, len(path_1)):
                fiber_graph[path_1[i - 1]][path_1[i]]['cost'] = 0
            for i in range(1, len(path_2)):
                fiber_graph[path_2[i - 1]][path_2[i]]['cost'] = 0
        path_1 = remove_cycle(path_1)
        path_2 = remove_cycle(path_2)
        traffic_path[f] = [copy.deepcopy(path_1), copy.deepcopy(path_2)]
        for i in range(1, len(path_1)):
            topo.add_edge(path_1[i - 1], path_1[i])
            load_graph[path_1[i-1]][path_1[i]]['load'] += demand
        for i in range(1, len(path_2)):
            topo.add_edge(path_2[i - 1], path_2[i])
            load_graph[path_2[i - 1]][path_2[i]]['load'] += demand
    # compute fiber cost
    cost = 0
    for (node_i, node_j) in topo.edges:
        cost += graph[node_i][node_j]['cost']


    return topo, load_graph, traffic_path, cost