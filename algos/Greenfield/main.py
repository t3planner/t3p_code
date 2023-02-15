from algos.ppo import ppo
import numpy as np
from algos.Greenfield.plan_env import GraphEnv
import networkx as nx
import pandas as pd
from algos.Greenfield.core import MLPActorCritic
from algos.traffic import update_traffic

def RL_cost(dataset, adjust_num, stru, gcn_outdim, steps_per_epoch, epochs, max_ep_len, cuda):

    # construct dataset
    fiber = pd.read_csv("datasets//{}//fiber.csv".format(dataset))
    G_fiber = nx.Graph()
    for i in range(len(fiber)):
        node_i = int(fiber.iloc[i, 0]) - 1
        node_j = int(fiber.iloc[i, 1]) - 1
        cost = fiber.iloc[i, 2]
        G_fiber.add_edge(node_i, node_j, cost=cost, cap=np.inf)

    node_csv = pd.read_csv("datasets//{}//node_IP.csv".format(dataset))
    node_num = len(node_csv)
    G_IP = nx.Graph()
    for node in range(node_num):
        if node_csv.iloc[node, 1] == "Core site":
            G_IP.add_node(int(node_csv.iloc[node, 0])-1, type='Core')
        elif node_csv.iloc[node, 1] == "Aggregation site":
            G_IP.add_node(int(node_csv.iloc[node, 0])-1, type='Agg')
        elif node_csv.iloc[node, 1] == "Access site":
            G_IP.add_node(int(node_csv.iloc[node, 0])-1, type='Acc')

    # get original topology of IP layer
    file_name = './datasets/{}/link.csv'.format(dataset)
    dis_matrix = pd.read_csv(file_name)
    for i in range(len(dis_matrix)):
        node_i = int(dis_matrix.iloc[i, 0]) - 1
        node_j = int(dis_matrix.iloc[i, 1]) - 1
        G_IP.add_edge(node_i, node_j, cost=0, load=0, cap=np.inf, fiber=[])

    # get SRLG constraints
    SRLG = []
    with open('./datasets/{}/SRLG.txt'.format(dataset)) as f:
        for line in f:
            nodes = line.split()
            size = int(nodes[0])

            if size == 2:
                node = int(nodes[1]) - 1
                node_i = int(nodes[2]) - 1
                node_j = int(nodes[4]) - 1
                conflicts = set([node_i, node_j]) & set(G_IP.adj[node])
            elif size == 3:
                node = int(nodes[1]) - 1
                node_i = int(nodes[2]) - 1
                node_j = int(nodes[4]) - 1
                node_k = int(nodes[6]) - 1
                conflicts = set([node_i, node_j, node_k]) & set(G_IP.adj[node])
            if len(conflicts) > 1:
                SRLG.append((node, conflicts))

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

    G_optimal_solution = None
    env = GraphEnv(G_fiber, G_IP, traffic, SRLG, adjust_num, stru, G_optimal_solution, cuda)

    ac_kwargs = dict(gcn_outdim=gcn_outdim, adjust_num=adjust_num, stru=stru, cuda=cuda)
    logger_kwargs = dict(
        output_dir="result//Greenfield scenario//{}//{}".format(dataset, stru), output_fname='progress.txt'
    )
    ac = MLPActorCritic
    ppo("cost", env, actor_critic=ac, ac_kwargs=ac_kwargs, logger_kwargs=logger_kwargs, steps_per_epoch=steps_per_epoch,epochs=epochs, max_ep_len=max_ep_len)