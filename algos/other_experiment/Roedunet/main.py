from algos.other_experiment.Roedunet.ppo import ppo
from algos.other_experiment.Roedunet.plan_env import GraphEnv
import networkx as nx
import pandas as pd
from algos.other_experiment.Roedunet.core import MLPActorCritic
import torch

def RL_Roedunet(dataset, del_num, gcn_outdim, steps_per_epoch, epochs, max_ep_len, cuda):

    G = nx.Graph()
    file_name = './datasets/{}/link.csv'.format(dataset)
    dis_matrix = pd.read_csv(file_name)
    for i in range(len(dis_matrix)):
        node_i = int(dis_matrix.iloc[i, 0]) - 1
        node_j = int(dis_matrix.iloc[i, 1]) - 1
        dis = float(dis_matrix.iloc[i, 2])
        G.add_edge(node_i, node_j, cost=dis)

    node_csv = pd.read_csv("datasets//{}//node.csv".format(dataset))
    node_list = []
    for node in range(len(node_csv)):
        if node_csv.iloc[node, 1] == "Aggregation site":
            node_list.append(int(node_csv.iloc[node, 0])-1)

    G_fiber = nx.Graph()
    file_name = './datasets/{}/fiber.csv'.format(dataset)
    dis_matrix = pd.read_csv(file_name)
    for i in range(len(dis_matrix)):
        node_i = int(dis_matrix.iloc[i, 0]) - 1
        node_j = int(dis_matrix.iloc[i, 1]) - 1
        dis = float(dis_matrix.iloc[i, 2])
        G_fiber.add_edge(node_i, node_j, cost=dis)

    traffic_matrix = pd.read_csv('./datasets/{}/traffic.csv'.format(dataset))
    traffic = {}
    for i in range(len(traffic_matrix)):
        src = int(traffic_matrix.iloc[i, 0]) - 1
        dst = int(traffic_matrix.iloc[i, 1]) - 1
        traffic[(src, dst)] = 0

    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    output_path = "result//motivation"
    env = GraphEnv(G, G_fiber, traffic, cuda, del_num, node_list, output_path)
    ac_kwargs = dict(gcn_outdim=gcn_outdim, del_num=del_num, cuda=cuda)
    logger_kwargs = dict(
        output_dir=output_path, output_fname='progress.txt'
    )
    ac = MLPActorCritic
    ppo(env, actor_critic=ac, ac_kwargs=ac_kwargs, logger_kwargs=logger_kwargs, steps_per_epoch=steps_per_epoch, epochs=epochs, max_ep_len=max_ep_len)