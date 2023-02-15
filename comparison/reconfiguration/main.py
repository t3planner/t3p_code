import os
import copy

import numpy as np
import pandas as pd
import networkx as nx

from utils import calculate_IP_cost
from configs import setup_configs
from ECMP.main import ECMP
from FFC.main import FFC
from Optimal_solution.main import Optimal_solution
from Ring_Optimal_solution.main import Ring_Optimal_solution
from OWAN.main import OWAN
from SMORE.main import SMORE
from Rule.main import Rule


def set_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    args = setup_configs()

    # construct dataset
    fiber_matrix = pd.read_csv('./datasets/{}/fiber.csv'.format(args.dataset))
    G_fiber = nx.Graph()
    for i in range(len(fiber_matrix)):
        node_i = int(fiber_matrix.iloc[i, 0]) - 1
        node_j = int(fiber_matrix.iloc[i, 1]) - 1
        cost = float(fiber_matrix.iloc[i, 2])
        G_fiber.add_edge(node_i, node_j, cost=cost, cap=np.inf)

    G_IP = nx.Graph()
    nodes = pd.read_csv('./datasets/{}/node.csv'.format(args.dataset))
    for i in range(len(nodes)):
        node = int(nodes.iloc[i, 0])-1
        node_type = nodes.iloc[i, 1]
        G_IP.add_node(node, type=node_type)
        
    # get original topology of IP layer
    IP_matrix = pd.read_csv('./datasets/{}/link.csv'.format(args.dataset))
    for i in range(len(IP_matrix)):
        node_i = int(IP_matrix.iloc[i, 0])-1
        node_j = int(IP_matrix.iloc[i, 1])-1
        G_IP.add_edge(node_i, node_j, cost=0, load=args.init_load, cap=args.cap, fiber=[])

    G_IP = calculate_IP_cost(G_fiber, G_IP)

    # get initial topology of IP layer
    graph = copy.deepcopy(G_IP)
    init_topo = pd.read_csv('./datasets/{}/init.csv'.format(args.dataset), header=None)
    for (node_i, node_j) in graph.edges: 
        if init_topo.iloc[node_i, node_j] == 0:
            graph.remove_edge(node_i, node_j)

    # get SRLG constrains
    SRLG = []
    with open('./datasets/{}/SRLG.txt'.format(args.dataset)) as f:
        for line in f:
            nodes = line.split()
            size = int(nodes[0])

            if size == 2: 
                node = int(nodes[1])-1
                node_i = int(nodes[2])-1
                node_j = int(nodes[4])-1
                conflicts = set([node_i, node_j]) & set(G_IP.adj[node])
            elif size == 3:
                node = int(nodes[1])-1
                node_i = int(nodes[2])-1
                node_j = int(nodes[4])-1
                node_k = int(nodes[6])-1
                conflicts = set([node_i, node_j, node_k]) & set(G_IP.adj[node])
            if len(conflicts) > 1:
                SRLG.append((node, conflicts))

    # get traffic demands
    traffic_matrix = pd.read_csv('./datasets/{}/traffic.csv'.format(args.dataset))
    traffic = {}
    for i in range(len(traffic_matrix)): 
        src = int(traffic_matrix.iloc[i, 0])-1
        dst = int(traffic_matrix.iloc[i, 1])-1
        if G_IP.degree[src] * G_IP.degree[dst] != 0:
            demand = float(traffic_matrix.iloc[i, 2])/1000
            if demand > 0:
                traffic[(src, dst)] = demand

    set_dir('./{}/results'.format(args.method))
    if args.method == 'ECMP': 
        ECMP(args.dataset, args.k, graph, traffic, SRLG)
    elif args.method == 'SMORE': 
        SMORE(args.dataset, args.k, graph, traffic, SRLG, args.time_limit)
    elif args.method == 'FFC': 
        FFC(args.dataset, args.k, args.fails, graph, traffic, SRLG, args.time_limit)
    elif args.method == 'Optimal_solution':
        Optimal_solution(args.dataset, graph, traffic, SRLG, args.time_limit)
    elif args.method == 'Ring_Optimal_solution':
        Ring_Optimal_solution(args.dataset, G_IP, traffic, SRLG, args.max_nodes, args.max_rings, args.total_load, args.time_limit)
    elif args.method == 'OWAN': 
        OWAN(args.dataset, G_IP, graph, traffic, SRLG, alpha=0.95, division=10, ring_len=10)
    elif args.method == 'Rule':
        Rule(args.dataset, G_IP, graph, traffic, SRLG)
    else: 
        print('Unsupported method.')
