import gym
import torch
import copy
import numpy as np
import networkx as nx
from gym import spaces
from algos.cal_IPcost import Cal_IPcost
from algos.heur import Heur, Heur_cost, Heur_congestion

class GraphEnv(gym.Env):
    def __init__(
        self, G_fiber: nx.Graph, original_graph: nx.Graph, traffic: list, SRLG: list, adjust_num: int, stru: str, G_ILP: nx.Graph, cuda=False
    ):
        self.G_fiber = G_fiber
        self.original_graph = Cal_IPcost(self.G_fiber, original_graph)
        self.SRLG = SRLG

        self.adjust_num = adjust_num
        self.stru = stru

        # compute cost
        self.graph = copy.deepcopy(original_graph)  # state graph used to perform an action
        _, self.original_heur_topo, self.original_heur_topo_fiber, self.original_heur_traffic_path, obj = Heur_cost(self.G_fiber, self.original_graph, traffic, self.SRLG)

        self.traffic = traffic

        # edge numbering
        self.idx_to_edge = {}
        self.edge_to_idx = {}
        for i, (node_i, node_j) in enumerate(original_graph.edges):
            self.idx_to_edge[i] = (node_i, node_j)
            self.edge_to_idx[(node_i, node_j)] = i
        self.edge_num = len(original_graph.edges)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.edge_num,))
        self.action_space = spaces.Discrete(self.edge_num)


        self.original_obj = obj
        self.obj = obj
        self.done_topo = copy.deepcopy(self.original_heur_topo)  # save the resulting graph for each trajectory
        self.done_topo_fiber = copy.deepcopy(self.original_heur_topo_fiber)  # save the resulting graph for each trajectory

        self.done_traffic_path = copy.deepcopy(self.original_heur_traffic_path)  # save the resulting routing paths for each trajectory

        self.cum_reward = 0
        self.cuda = cuda

    def get_obs(self):
        self.graph = Cal_IPcost(self.G_fiber, self.graph)

        #  state 1: the cost of IP_link
        state_obj = torch.FloatTensor([self.graph[edge[0]][edge[1]]['cost'] if edge in self.graph.edges else 0.0 for edge in self.original_graph.edges]).reshape((self.action_space.n, 1))

        #  state 2: the existence of IP_link
        state_flag = torch.FloatTensor([1.0 if edge in self.graph.edges else 0.0 for edge in self.original_graph.edges]).reshape((self.edge_num, 1))

        obs = tuple([state_obj, state_flag])

        return obs

    def step(self, action, obs):
        done, info = True, None
        for action_i in action:
            node_i, node_j = self.idx_to_edge[int(action_i)]
            if (node_i, node_j) not in self.graph.edges:  # vaild edge
                reward = -5.0
            else:
                done = False
                self.graph.remove_edge(node_i, node_j)

        if done == False:
            #  heuristic method and determine whether the result meets the constraints
            flag, topo, topo_fiber, traffic_path, obj = Heur(self.stru, "cost", self.G_fiber, self.graph, self.traffic, self.SRLG)

            if flag == False:  # false edge
                done = True
                reward = -5.0
            else:
                if obj > self.obj:
                    print("vaild")
                    done = True
                    reward = -5.0
                else:
                    obs = self.get_obs()
                    reward = (self.obj - obj)
                    self.obj = obj
                    self.done_topo = copy.deepcopy(topo)
                    self.done_topo_fiber = copy.deepcopy(topo_fiber)
                    self.done_traffic_path = copy.deepcopy(traffic_path)

            self.cum_reward += reward

        return obs, reward, done, self.obj, self.done_topo, self.done_topo_fiber, self.done_traffic_path, info



    def reset(self):
        self.graph = copy.deepcopy(self.original_graph)
        self.done_topo = copy.deepcopy(self.original_heur_topo)
        self.done_topo_fiber = copy.deepcopy(self.original_heur_topo_fiber)
        obs = self.get_obs()
        self.obj = self.original_obj
        self.cum_reward = 0

        return obs

