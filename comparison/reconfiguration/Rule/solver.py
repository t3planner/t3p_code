import copy
import random
import gurobipy

import heapq as hp
import numpy as np
import networkx as nx

from Rule.traffic import *


def calaulate_congestion(original_graph: nx.Graph, traffic, SRLG):
    '''
        calculate the routing of the traffic on the graph and calculate the value of congestion
    '''
    graph = copy.deepcopy(original_graph)
    graph, SRLG_pair = SRLG_constrain(graph, SRLG)
    topo = nx.create_empty_copy(graph)  # Record result topology
    flow_path = {(f[0], f[1]): [] for f in traffic}
    flag = 0
    all_distances = compute_all_distances(graph, traffic)
    all_distances = sorted(all_distances.items(), key=lambda kv: (kv[1], kv[0]))
    for f in all_distances:
        src = f[0][0]
        dst = f[0][1]
        demand = traffic[(src, dst)]
        graph_new = demand_constrain(graph, demand)
        path_1, path_2 = suurballe(graph_new, src, dst)
        path_1 = remove_cycle(path_1)
        path_2 = remove_cycle(path_2)

        if path_1 != [] and path_2 != []:
            for i in range(1, len(path_1)):
                graph[path_1[i - 1]][path_1[i]]['cost'] += 10
                graph[path_1[i - 1]][path_1[i]]['load'] += traffic[(src, dst)]
            for i in range(1, len(path_2)):
                graph[path_2[i - 1]][path_2[i]]['cost'] += 10
                graph[path_2[i - 1]][path_2[i]]['load'] += traffic[(src, dst)]
        for node in path_1:
            if node in SRLG_pair.keys():
                path_1[path_1.index(node)] = SRLG_pair[node]
        for node in path_2:
            if node in SRLG_pair.keys():
                path_2[path_2.index(node)] = SRLG_pair[node]
        path_1 = remove_cycle(path_1)
        path_2 = remove_cycle(path_2)

        for i in range(1, len(path_1)):
            if (path_1[i-1], path_1[i]) not in topo.edges:
                topo.add_edge(path_1[i - 1], path_1[i], flow=[])

        for i in range(1, len(path_2)):
            if (path_2[i - 1], path_2[i]) not in topo.edges:
                topo.add_edge(path_2[i - 1], path_2[i], flow=[])
        if len(path_1) < len(path_2):
            for i in range(1, len(path_1)):
                original_graph[path_1[i - 1]][path_1[i]]['load'] += traffic[(src, dst)]
                topo[path_1[i - 1]][path_1[i]]['flow'].append([(src, dst), traffic[(src, dst)], len(path_1)])
                flow_path[(src, dst)] = [path_1, path_2, 1]
        else:
            for i in range(1, len(path_2)):
                original_graph[path_2[i - 1]][path_2[i]]['load'] += traffic[(src, dst)]
                topo[path_2[i - 1]][path_2[i]]['flow'].append([(src, dst), traffic[(src, dst)], len(path_2)])
                flow_path[(src, dst)] = [path_1, path_2, 2]
    congestion = max([original_graph[node_i][node_j]['load'] for (node_i, node_j) in topo.edges])
    unallacted_demand = 0
    flow = []
    for (node_i, node_j) in original_graph.edges:
        unallacted_demand += (original_graph[node_i][node_j]['cap']-original_graph[node_i][node_j]['load'])
        if original_graph[node_i][node_j]['load'] == congestion:
            flow = topo[node_i][node_j]['flow']
    unallacted_demand += sum([f[1]*f[2] for f in flow])  # Record the remaining bandwidth

    return original_graph, congestion, flow, flow_path, unallacted_demand


def Rule_algorithm(original_graph, graph, traffic, SRLG):
    '''
        regulate the traffic on the edge with the most load traffic
    '''
    graph_new = copy.deepcopy(graph)
    graph_new, congestion, flow, flow_path, unallacted_demand = calaulate_congestion(graph_new, traffic, SRLG)


    flow_path_old = copy.deepcopy(flow_path)
    allcated_demand = 0  # Record how much bandwidth is used
    topo = nx.create_empty_copy(original_graph)
    for (node_i, node_j) in original_graph.edges:
        topo.add_edge(node_i, node_j, load=0, cost=1, cap=np.inf)
    for f in flow:
        src = f[0][0]
        dst = f[0][1]
        demand = f[1]
        path = flow_path[(src, dst)][flow_path[(src, dst)][2]-1]
        for i in range(1, len(path)):
            graph_new[path[i - 1]][path[i]]['load'] -= demand
        for (node_i, node_j) in topo.edges:
            topo[node_i][node_j]['load'] = 0
            topo[node_i][node_j]['cost'] = 1
        for edge in topo.edges:
            if edge in graph_new.edges:
                if demand+graph_new[edge[0]][edge[1]]['load'] >= congestion-demand:
                    topo[edge[0]][edge[1]]['cost'] = np.inf
        fiber_graph = copy.deepcopy(topo)
        fiber_graph, SRLG_pair = SRLG_constrain(fiber_graph, SRLG)
        path_1, path_2 = suurballe(fiber_graph, src, dst)
        path_1 = remove_cycle(path_1)
        path_2 = remove_cycle(path_2)

        for node in path_1:
            if node in SRLG_pair.keys():
                path_1[path_1.index(node)] = SRLG_pair[node]
        for node in path_2:
            if node in SRLG_pair.keys():
                path_2[path_2.index(node)] = SRLG_pair[node]
        path_1 = remove_cycle(path_1)
        path_2 = remove_cycle(path_2)


        if len(path_1) < len(path_2):
            for i in range(1, len(path_1)):
                if (path_1[i-1], path_1[i]) not in graph_new.edges:
                    graph_new.add_edge(path_1[i-1], path_1[i], load=demand)
                else:
                    graph_new[path_1[i - 1]][path_1[i]]['load'] += demand
                allcated_demand += demand
                flow_path[(src, dst)] = [path_1, path_2, 1]
        else:
            for i in range(1, len(path_2)):
                if (path_2[i-1], path_2[i]) not in graph_new.edges:
                    graph_new.add_edge(path_2[i-1], path_2[i], load=demand)
                else:
                    graph_new[path_2[i - 1]][path_2[i]]['load'] += demand
                allcated_demand += demand
                flow_path[(src, dst)] = [path_1, path_2, 2]

    if allcated_demand > unallacted_demand or max([graph_new[node_i][node_j]['load'] for (node_i, node_j) in graph_new.edges]) > congestion:
        return graph, graph, flow_path_old
    return graph_new, graph_new, flow_path

