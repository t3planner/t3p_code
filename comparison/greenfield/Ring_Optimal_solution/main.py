import csv 
import copy
import time

import numpy as np
import networkx as nx

from Ring_Optimal_solution.solver import linear_programing


def SRLG_constrain(G:nx.Graph, graph:nx.Graph, SRLG:list): 
    SRLG_pair = {}
    for node, conflicts in SRLG: 
        if len(conflicts) == 2:
            node_i = conflicts.pop()
            node_j = conflicts.pop()

            if (node, node_i) in G.edges and (node, node_j) in G.edges: 
                node_s = max(graph.nodes)+1
                SRLG_pair[node_s] = node
                graph.add_edge(
                    node, node_s, 
                    cost=0, 
                    load=0, 
                    cap=min(G[node][node_i]['cap'], G[node][node_j]['cap'])
                )

                if node_i not in graph.adj[node]:
                    for neighbor in graph.adj[node_i]:
                        if neighbor in SRLG_pair.keys() and node in graph.adj[neighbor]: 
                            node_i = neighbor
                graph.add_edge(
                    node_s, node_i, 
                    cost=graph[node][node_i]['cost'], 
                    load=graph[node][node_i]['load'], 
                    cap=graph[node][node_i]['cap']
                )
                graph.remove_edge(node, node_i)

                if node_j not in graph.adj[node]:
                    for neighbor in graph.adj[node_j]:
                        if neighbor in SRLG_pair.keys() and node in graph.adj[neighbor]: 
                            node_j = neighbor
                graph.add_edge(
                    node_s, node_j, 
                    cost=graph[node][node_j]['cost'], 
                    load=graph[node][node_j]['load'], 
                    cap=graph[node][node_j]['cap']
                )
                graph.remove_edge(node, node_j)

        elif len(conflicts) == 3:
            node_i = conflicts.pop()
            node_j = conflicts.pop()
            node_k = conflicts.pop()

            if (node, node_i) in G.edges and (node, node_j) in G.edges and (node, node_k) in G.edges: 
                node_s = max(graph.nodes)+1
                SRLG_pair[node_s] = node
                graph.add_edge(
                    node, node_s, 
                    cost=0, 
                    load=0, 
                    cap=min(G[node][node_i]['cap'], G[node][node_j]['cap'])
                )

                if node_i not in graph.adj[node]:
                    for neighbor in graph.adj[node_i]:
                        if neighbor in SRLG_pair.keys() and node in graph.adj[neighbor]: 
                            node_i = neighbor
                graph.add_edge(
                    node_s, node_i, 
                    cost=graph[node][node_i]['cost'], 
                    load=graph[node][node_i]['load'], 
                    cap=graph[node][node_i]['cap']
                )
                graph.remove_edge(node, node_i)

                if node_j not in graph.adj[node]:
                    for neighbor in graph.adj[node_j]:
                        if neighbor in SRLG_pair.keys() and node in graph.adj[neighbor]: 
                            node_j = neighbor
                graph.add_edge(
                    node_s, node_j, 
                    cost=graph[node][node_j]['cost'], 
                    load=graph[node][node_j]['load'], 
                    cap=graph[node][node_j]['cap']
                )
                graph.remove_edge(node, node_j)

                if node_k not in graph.adj[node]:
                    for neighbor in graph.adj[node_k]:
                        if neighbor in SRLG_pair.keys() and node in graph.adj[neighbor]: 
                            node_k = neighbor
                graph.add_edge(
                    node_s, node_k, 
                    cost=graph[node][node_k]['cost'], 
                    load=graph[node][node_k]['load'], 
                    cap=graph[node][node_k]['cap']
                )
                graph.remove_edge(node, node_k)
    
    return graph, SRLG_pair


def save_results(dataset, traffic_paths:list=None, fiber_net:nx.Graph=None, load_graph:nx.Graph=None):
    if traffic_paths is not None: 
        with open('./Ring_Optimal_solution/results/{}_flow.csv'.format(dataset), 'w', newline='') as f:
            writer = csv.writer(f)
            for row in traffic_paths.values():
                writer.writerow(row)

    if fiber_net is not None:
        node_num = max(fiber_net.nodes)+1
        fiber_matrix = np.zeros((node_num, node_num))
        for (node_i, node_j) in fiber_net.edges:
            fiber_matrix[node_i, node_j] = 1
            fiber_matrix[node_j, node_i] = 1
        with open('./Ring_Optimal_solution/results/{}_fiber_matrix.csv'.format(dataset), 'w', newline='') as f:
            writer = csv.writer(f)
            for row in fiber_matrix:
                writer.writerow(row)

    if load_graph is not None:
        node_num = max(load_graph.nodes)+1
        load_matrix = np.zeros((node_num, node_num))
        for (node_i, node_j) in load_graph.edges:
            load_matrix[node_i, node_j] = load_graph[node_i][node_j]['load']
            load_matrix[node_j, node_i] = load_graph[node_i][node_j]['load']
        with open('./Ring_Optimal_solution/results/{}_load_matrix.csv'.format(dataset), 'w', newline='') as f:
            writer = csv.writer(f)
            for row in load_matrix:
                writer.writerow(row)


def Ring_Optimal_solution(
    dataset, original_graph:nx.Graph, original_traffic:dict, original_SRLG:list, 
    max_nodes=10, max_rings=30, time_limit=None
): 
    G = copy.deepcopy(original_graph)
    graph = copy.deepcopy(original_graph)
    traffic = copy.deepcopy(original_traffic)
    SRLG = copy.deepcopy(original_SRLG)

    graph, SRLG_pair = SRLG_constrain(G, graph, SRLG)
    for (node_s, node) in SRLG_pair.items():
        graph.add_node(node_s, type=graph.nodes[node]['type'])

    with open('./Ring_Optimal_solution/results/results.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'total_time', 'cost', 'none_backup', 'max_load'])
    
    start_time = time.time()

    fiber_net, load_graph, traffic_paths = linear_programing(G, graph, traffic, SRLG_pair, max_nodes, max_rings, time_limit)
    # fiber_net: final topology
    # traffic_paths: allocated paths for traffic demands
    # load_graph: load of links according to traffic_paths

    total_time = time.time() - start_time

    save_results(dataset, traffic_paths, fiber_net, load_graph)

    # cost of final topology
    cost = 0
    for (node_i, node_j) in fiber_net.edges:
        cost += G[node_i][node_j]['cost']

    # num of traffic without backup path
    none_backup = 0
    for paths in traffic_paths.values():
        if len(paths) < 2:
            none_backup += 1

    # max load of links
    max_load = 0
    for (node_i, node_j) in load_graph.edges: 
        if load_graph[node_i][node_j]['load'] > max_load: 
            max_load = load_graph[node_i][node_j]['load']

    with open('./Ring_Optimal_solution/results/results.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([dataset, total_time, cost, none_backup, max_load]) 