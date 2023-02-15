import os
import csv 
import copy
import torch

import numpy as np
import pandas as pd
import networkx as nx
import torch.nn as nn

from NeuroPlan.env import Env
from NeuroPlan.model import GNNActorCritic
from NeuroPlan.VPG import policy_gradient
from NeuroPlan.traffic_allocation import allocate_traffic


def save_results(dataset, traffic_paths:list=None, fiber_net:nx.Graph=None, load_graph:nx.Graph=None):
    if traffic_paths is not None: 
        with open('./NeuroPlan/results/{}/flow.csv'.format(dataset), 'w', newline='') as f:
            writer = csv.writer(f)
            for row in traffic_paths.values():
                writer.writerow(row)

    if fiber_net is not None:
        node_num = max(fiber_net.nodes)+1
        fiber_matrix = np.zeros((node_num, node_num))
        for (node_i, node_j) in fiber_net.edges:
            fiber_matrix[node_i, node_j] = 1
            fiber_matrix[node_j, node_i] = 1
        with open('./NeuroPlan/results/{}/fiber_matrix.csv'.format(dataset), 'w', newline='') as f:
            writer = csv.writer(f)
            for row in fiber_matrix:
                writer.writerow(row)

    if load_graph is not None:
        node_num = max(load_graph.nodes)+1
        load_matrix = np.zeros((node_num, node_num))
        for (node_i, node_j) in load_graph.edges:
            load_matrix[node_i, node_j] = load_graph[node_i][node_j]['load']
            load_matrix[node_j, node_i] = load_graph[node_i][node_j]['load']
        with open('./NeuroPlan/results/{}/load_matrix.csv'.format(dataset), 'w', newline='') as f:
            writer = csv.writer(f)
            for row in load_matrix:
                writer.writerow(row)


def update_graph(fiber_net:nx.Graph, path):
    for i in range(len(path)-1):
        node_i = path[i]
        node_j = path[i+1]
        fiber_net.add_edge(node_i, node_j)
        
    return fiber_net


def NeuroPlan(
    dataset, original_graph:nx.Graph, original_traffic:dict, original_SRLG:list, 
    delta_load, max_deltas, feature_dim=1, time_limit=None
):
    graph = copy.deepcopy(original_graph)
    traffic = copy.deepcopy(original_traffic)
    SRLG = copy.deepcopy(original_SRLG)

    env = Env(graph, traffic, SRLG, delta_load, max_deltas, feature_dim, time_limit)
    ac = GNNActorCritic
    device = torch.device('cpu')
    ac_kwargs = dict(
        GNN_layers=2, GNN_hidden_size=256, 
        MLP_hidden_sizes=(256, 256), MLP_activation=nn.Tanh
    )
    logger_kwargs = dict(
        output_dir='./NeuroPlan/results/{}'.format(dataset), 
        output_name='progress.txt'
    )

    # sol: path_cost, load_cost, total_cost, acts
    sol = policy_gradient(
        env, ac, device, ac_kwargs=ac_kwargs, logger_kwargs=logger_kwargs, 
        gamma=0.99, lam=0.97, pi_lr=3e-4, v_lr=1e-3, v_iters=80,
        max_ep_len=600, steps_per_epoch=1200, epochs=50, save_freq=5, 
        seed=0, model_path=None
    )

    fiber_net = nx.Graph()
    fiber_net.add_nodes_from(original_graph)

    load_graph = nx.Graph()
    for (node_i, node_j) in original_graph.edges:
        load_graph.add_edge(
            node_i, node_j, 
            cost=original_graph[node_i][node_j]['cost'], 
            load=0, 
            cap=original_graph[node_i][node_j]['cap']
        )

    with open('./NeuroPlan/results/results.csv'.format(dataset), 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'cost', 'none_backup', 'max_load'])
    
    acts = sol['acts']  # action sequence the agent take, consist of edge (node_i, node_j) and delta_load on this edge
    for [(node_i, node_j), delta_load] in acts: 
        load_graph[node_i][node_j]['load'] += delta_load

    traffic_paths = allocate_traffic(load_graph, traffic, SRLG)
    for (src, dst) in traffic_paths.keys():
        paths = traffic_paths[(src, dst)]
        for j in range(len(paths)): 
            fiber_net = update_graph(fiber_net, paths[j])

    # fiber_net: final topology
    # traffic_paths: allocated paths for traffic demands
    # load_graph: load of links according to traffic_paths
    
    save_results(dataset, traffic_paths, fiber_net, load_graph)

    # cost of final topology
    cost = 0
    for (node_i, node_j) in fiber_net.edges:
        cost += original_graph[node_i][node_j]['cost']
    
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

    with open('./NeuroPlan/results/results.csv'.format(dataset), 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([dataset, cost, none_backup, max_load]) 