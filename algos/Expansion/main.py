from algos.ppo import ppo
from algos.Expansion.plan_env import GraphEnv
import networkx as nx
import pandas as pd
import numpy as np
import copy
from algos.Expansion.core import MLPActorCritic
from algos.traffic import update_traffic

def RL_addnode(dataset, adjust_num, stru, steps_per_epoch, epochs, max_ep_len, cuda):
    # construct dataset
    # add_list: nodes to be added
    node_csv = pd.read_csv("datasets//{}//node.csv".format(dataset))
    node_list = []
    node_num = len(node_csv)
    G_IP = nx.Graph()
    for node in range(node_num):
        if node_csv.iloc[node, 1] == "Core site":
            G_IP.add_node(int(node_csv.iloc[node, 0])-1, type='Core')
        elif node_csv.iloc[node, 1] == "Aggregation site":
            G_IP.add_node(int(node_csv.iloc[node, 0])-1, type='Agg')
        elif node_csv.iloc[node, 1] == "Access site":
            G_IP.add_node(int(node_csv.iloc[node, 0])-1, type='Acc')
        if node_csv.iloc[node, 2] == "FALSE":
            node_list.append(int(node_csv.iloc[node, 0])-1)

    # get initial topology
    file_name = './datasets/{}/init.csv'.format(dataset)
    fiber_matrix = pd.read_csv(file_name)
    for i in range(len(fiber_matrix)):
        node_i = int(fiber_matrix.iloc[i, 0]) - 1
        node_j = int(fiber_matrix.iloc[i, 1]) - 1
        G_IP.add_edge(node_i, node_j, cost=0, load=0, cap=np.inf)

    G_init = copy.deepcopy(G_IP)

    # get additional topology, including nodes & links to be added
    file_name = './datasets/{}/link.csv'.format(dataset)
    dis_matrix = pd.read_csv(file_name)
    for i in range(len(dis_matrix)):
        node_i = int(dis_matrix.iloc[i, 0]) - 1
        node_j = int(dis_matrix.iloc[i, 1]) - 1
        cost = dis_matrix.iloc[i, 2]
        G_IP.add_edge(node_i, node_j, cost=cost, load=0, cap=np.inf)

    SRLG = []

    # get traffic demands
    traffic_matrix = pd.read_csv('./datasets/{}/traffic.csv'.format(dataset))
    traffic = {}
    for i in range(len(traffic_matrix)):
        src = int(traffic_matrix.iloc[i, 0]) - 1
        dst = int(traffic_matrix.iloc[i, 1]) - 1
        demand = float(traffic_matrix.iloc[i, 2])
        traffic[(src, dst)] = demand
    # filter unreachable traffic
    traffic = update_traffic(G_IP, traffic, SRLG)

    env = GraphEnv(G_IP, traffic, SRLG, adjust_num, stru, G_init, node_list, cuda)

    ac_kwargs = dict(adjust_num=adjust_num, cuda=cuda)
    logger_kwargs = dict(
        output_dir="result//New sites expansion scenario", output_fname='progress.txt'
    )
    ac = MLPActorCritic
    ppo("addnode", env, actor_critic=ac, ac_kwargs=ac_kwargs, logger_kwargs=logger_kwargs, steps_per_epoch=steps_per_epoch,epochs=epochs, max_ep_len=max_ep_len)