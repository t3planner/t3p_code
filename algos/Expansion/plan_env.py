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
        self, original_graph: nx.Graph, traffic: list, SRLG: list, adjust_num: int, stru: str, G_init: nx.Graph, node_list: list, cuda=False
    ):
        self.SRLG = SRLG
        self.G_fiber = copy.deepcopy(original_graph)
        self.original_graph = Cal_IPcost(self.G_fiber, original_graph)
        self.adjust_num = adjust_num
        self.stru = stru

        # compute cost
        self.graph = copy.deepcopy(original_graph)  # state graph used to perform an action
        self.G_init = G_init
        self.node_list = node_list
        _, self.original_heur_topo, _, self.original_heur_traffic_path, obj = Heur_cost(self.G_fiber, original_graph, traffic, self.SRLG)
        self.original_heur_topo.add_edges_from(self.G_init.edges)
        self.traffic = traffic

        # edge numbering
        self.idx_to_edge = {}
        self.edge_to_idx = {}
        # only for the edges to be added
        for (node_i, node_j) in original_graph.edges:
            if (node_i, node_j) not in self.G_init.edges():
                self.idx_to_edge[len(self.idx_to_edge)] = (node_i, node_j)
                self.edge_to_idx[(node_i, node_j)] = len(self.edge_to_idx)
        self.edge_num = len(self.idx_to_edge)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.edge_num,))
        self.action_space = spaces.Discrete(self.edge_num)

        self.original_node_edge = {node: [] for node in self.original_graph.nodes}
        for edge in self.original_graph.edges:
            self.original_node_edge[edge[0]].append(edge)
            self.original_node_edge[edge[1]].append(edge)
        self.node_edge = copy.deepcopy(self.original_node_edge)

        self.original_obj = obj
        self.obj = obj
        self.done_topo = copy.deepcopy(self.original_heur_topo)  # save the resulting graph for each trajectory
        self.done_traffic_path = copy.deepcopy(self.original_heur_traffic_path)  # save the resulting routing paths for each trajectory

        self.cum_reward = 0
        self.cuda = cuda

    def get_obs(self):
        #  state 1: the cost or load of IP_link
        state_obj = torch.FloatTensor([self.graph[edge[0]][edge[1]]['cost'] if edge in self.graph.edges else 0.0 for edge in self.original_graph.edges if edge not in self.G_init.edges]).reshape((self.action_space.n, 1))

        #  state 2: the existence of IP_link
        state_flag = torch.FloatTensor([1.0 if edge in self.graph.edges else 0.0 for edge in self.original_graph.edges if edge not in self.G_init.edges]).reshape((self.edge_num, 1))

        state_tonggou = torch.FloatTensor(np.ones((self.edge_num, 1)))
        if self.stru == "dual-homing":
            for node in self.graph:
                if len(self.node_edge[node]) == 2:  # if node's degree is 2, the value of the two edges corresponding to this node is high
                    if self.node_edge[node][0] in self.edge_to_idx:
                        state_tonggou[self.edge_to_idx[self.node_edge[node][0]]] = 2
                    if self.node_edge[node][1] in self.edge_to_idx:
                        state_tonggou[self.edge_to_idx[self.node_edge[node][1]]] = 2
        elif self.stru == "mesh":
            for node in self.graph.nodes:
                if len(self.node_edge[node]) == 3:  # if node's degree is 2, the value of the three edges corresponding to this node is high
                    if self.node_edge[node][0] in self.edge_to_idx:
                        state_tonggou[self.edge_to_idx[self.node_edge[node][0]]] = 2
                    if self.node_edge[node][1] in self.edge_to_idx:
                        state_tonggou[self.edge_to_idx[self.node_edge[node][1]]] = 2
                    if self.node_edge[node][2] in self.edge_to_idx:
                        state_tonggou[self.edge_to_idx[self.node_edge[node][2]]] = 2
        elif self.stru == "hybrid ring-mesh":
            for node in self.graph:
                if len(self.node_edge[node]) == 2:  # if node's degree is 2, the value of the two edges corresponding to this node is high
                    if self.node_edge[node][0] in self.edge_to_idx:
                        state_tonggou[self.edge_to_idx[self.node_edge[node][0]]] = 2
                    if self.node_edge[node][1] in self.edge_to_idx:
                        state_tonggou[self.edge_to_idx[self.node_edge[node][1]]] = 2

            for node in self.graph.nodes:
                if self.graph.nodes()[node]['type'] != 'Acc' and len(self.node_edge[node]) == 3:
                    if self.node_edge[node][0] in self.edge_to_idx:
                        state_tonggou[self.edge_to_idx[self.node_edge[node][0]]] = 2
                    if self.node_edge[node][1] in self.edge_to_idx:
                        state_tonggou[self.edge_to_idx[self.node_edge[node][1]]] = 2
                    if self.node_edge[node][2] in self.edge_to_idx:
                        state_tonggou[self.edge_to_idx[self.node_edge[node][2]]] = 2
        if self.cuda:
            state_obj, state_flag, state_tonggou = state_obj.cuda(), state_flag.cuda(), state_tonggou.cuda()
        obs = tuple([state_obj, state_flag, state_tonggou])

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
                self.node_edge[node_i].remove((node_i, node_j))
                self.node_edge[node_j].remove((node_i, node_j))

        if done == False:
            #  heuristic method and determine whether the result meets the constraints
            for edge in self.graph.edges:
                self.graph[edge[0]][edge[1]]['load'] = 0
            flag, topo, topo_fiber, traffic_path, obj = Heur(self.stru, "addnode", self.G_fiber, self.graph, self.traffic, self.SRLG, node_list=self.node_list, G_init=self.G_init)

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
                    self.done_traffic_path = copy.deepcopy(traffic_path)

            self.cum_reward += reward

        return obs, reward, done, self.obj, self.done_topo, None, self.done_traffic_path, info



    def reset(self):
        self.graph = copy.deepcopy(self.original_graph)
        self.done_topo = copy.deepcopy(self.original_heur_topo)
        self.node_edge = copy.deepcopy(self.original_node_edge)
        obs = self.get_obs()
        self.obj = self.original_obj
        self.cum_reward = 0

        return obs

