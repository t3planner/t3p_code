import os
import csv 
import copy
import time

import numpy as np
import pandas as pd
import networkx as nx

from Rule.solver import Rule_algorithm


def save_results(dataset, traffic_paths:list=None, fiber_net:nx.Graph=None, load_graph:nx.Graph=None):
    if traffic_paths is not None: 
        with open('./Rule/results/{}_flow.csv'.format(dataset), 'w', newline='') as f:
            writer = csv.writer(f)
            for row in traffic_paths.values():
                writer.writerow(row)

    if fiber_net is not None:
        node_num = max(fiber_net.nodes)+1
        fiber_matrix = np.zeros((node_num, node_num))
        for (node_i, node_j) in fiber_net.edges:
            fiber_matrix[node_i, node_j] = 1
            fiber_matrix[node_j, node_i] = 1
        with open('./Rule/results/{}_fiber_matrix.csv'.format(dataset), 'w', newline='') as f:
            writer = csv.writer(f)
            for row in fiber_matrix:
                writer.writerow(row)

    if load_graph is not None:
        node_num = max(load_graph.nodes)+1
        load_matrix = np.zeros((node_num, node_num))
        for (node_i, node_j) in load_graph.edges:
            load_matrix[node_i, node_j] = load_graph[node_i][node_j]['load']
            load_matrix[node_j, node_i] = load_graph[node_i][node_j]['load']
        with open('./Rule/results/{}_load_matrix.csv'.format(dataset), 'w', newline='') as f:
            writer = csv.writer(f)
            for row in load_matrix:
                writer.writerow(row)


def Rule(
    dataset, original_graph:nx.Graph, add_list:list, original_traffic:dict
): 
    G = copy.deepcopy(original_graph)
    graph = copy.deepcopy(original_graph)
    traffic = copy.deepcopy(original_traffic)

    with open('./Rule/results/results.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'total_time', 'cost', 'none_backup', 'max_load'])

    start_time = time.time()

    fiber_net, load_graph, traffic_paths, cost = Rule_algorithm(graph, traffic)
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
        if [] in paths:
            none_backup += 1

    # max load of links
    max_load = 0
    for (node_i, node_j) in load_graph.edges: 
        if load_graph[node_i][node_j]['load'] > max_load: 
            max_load = load_graph[node_i][node_j]['load']

    with open('./Rule/results/results.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([dataset, total_time, cost, none_backup, max_load])