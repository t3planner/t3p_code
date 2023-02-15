import os
import copy

import numpy as np
import pandas as pd
import networkx as nx

from configs import setup_configs
from OWAN.main import OWAN
from Rule.main import Rule


def set_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    args = setup_configs()

    # construct dataset
    # add_list: nodes to be added
    add_list = []
    G = nx.Graph()
    nodes = pd.read_csv('./datasets/{}/node.csv'.format(args.dataset))
    for i in range(len(nodes)):
        node = int(nodes.iloc[i, 0])-1
        node_type = nodes.iloc[i, 1]
        exist = nodes.iloc[i, 2]
        G.add_node(node, type=node_type)
        if not exist:
            add_list.append(node)

    # get initial topology
    init_topo = pd.read_csv('./datasets/{}/init.csv'.format(args.dataset))
    for i in range(len(init_topo)):
        node_i = int(init_topo.iloc[i, 0])-1
        node_j = int(init_topo.iloc[i, 1])-1
        dis = float(init_topo.iloc[i, 2])
        G.add_edge(node_i, node_j, cost=0, load=args.init_load, cap=args.cap)
    
    # get additional topology, including nodes & links to be added
    add_topo = pd.read_csv('./datasets/{}/link.csv'.format(args.dataset))
    for i in range(len(add_topo)):
        node_i = int(add_topo.iloc[i, 0])-1
        node_j = int(add_topo.iloc[i, 1])-1
        dis = float(add_topo.iloc[i, 2])
        G.add_edge(node_i, node_j, cost=dis, load=args.init_load, cap=args.cap)

    # get traffic demands
    traffic_matrix = pd.read_csv('./datasets/{}/traffic.csv'.format(args.dataset))
    traffic = {}
    for i in range(len(traffic_matrix)): 
        src = int(traffic_matrix.iloc[i, 0])-1
        dst = int(traffic_matrix.iloc[i, 1])-1
        if G.degree[src] * G.degree[dst] != 0:
            demand = float(traffic_matrix.iloc[i, 2])/1000
            if demand > 0:
                traffic[(src, dst)] = demand

    set_dir('./{}/results'.format(args.method))
    if args.method == 'OWAN': 
        OWAN(args.dataset, G, add_list, traffic, alpha=0.95, division=5, ring_len=args.max_nodes)
    elif args.method == 'Rule':
        Rule(args.dataset, G, add_list, traffic)
    else: 
        print('Unsupported method.')
