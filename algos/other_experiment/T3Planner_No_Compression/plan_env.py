import gym
import torch
import copy
import numpy as np
import networkx as nx
from gym import spaces
from algos.GNN.utils import cal_GNN_input
from algos.cal_IPcost import Cal_IPcost
from algos.structure_encoder import stru_encoder
from algos.heur import Heur, Heur_cost, Heur_congestion
from algos.dual_homing.ring_isomorphism import subgraph_isomorphism_edge_counts, remove_edge

class GraphEnv(gym.Env):
    def __init__(
        self, G_fiber: nx.Graph, G_IP: nx.Graph, traffic: list, SRLG: list, adjust_num: int, stru: str, G_ILP: nx.Graph, cuda=False
    ):
        self.original_G_fiber = G_fiber
        self.original_G_IP = Cal_IPcost(G_fiber, G_IP)
        self.SRLG = SRLG

        self.adjust_num = adjust_num
        self.stru = stru

        # compute cost
        self.graph_fiber = copy.deepcopy(self.original_G_fiber)  # state graph used to perform an action
        self.graph_IP = copy.deepcopy(self.original_G_IP)  # state graph used to perform an action

        _, self.original_heur_topo, self.original_heur_topo_fiber, self.original_heur_traffic_path, obj = Heur_cost(self.original_G_fiber, self.original_G_IP, traffic, self.SRLG)

        self.traffic = traffic

        # edge numbering
        self.idx_to_edge = {}
        self.edge_to_idx = {}
        for i, (node_i, node_j) in enumerate(self.original_G_fiber.edges):
            self.idx_to_edge[i] = (node_i, node_j)
            self.edge_to_idx[(node_i, node_j)] = i
        self.edge_num = len(self.original_G_fiber.edges)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.edge_num,))
        self.action_space = spaces.Discrete(self.edge_num)

        if stru == "dual-homing" or stru == "hybrid ring-mesh":
            self.original_edge_to_ring, self.original_edge_feature = subgraph_isomorphism_edge_counts(self.original_G_fiber, length=10)
            self.edge_to_ring = copy.deepcopy(self.original_edge_to_ring)
            self.edge_feature = copy.deepcopy(self.original_edge_feature)

        self.original_node_edge = {node: [] for node in self.original_G_fiber.nodes}
        for edge in self.original_G_fiber.edges:
            self.original_node_edge[edge[0]].append(edge)
            self.original_node_edge[edge[1]].append(edge)
        self.node_edge = copy.deepcopy(self.original_node_edge)

        self.original_obj = obj
        self.obj = obj
        self.done_topo = copy.deepcopy(self.original_heur_topo)  # save the resulting graph for each trajectory
        self.done_topo_fiber = copy.deepcopy(self.original_heur_topo_fiber)  # save the resulting graph for each trajectory

        self.done_traffic_path = copy.deepcopy(self.original_heur_traffic_path)  # save the resulting routing paths for each trajectory

        self.cum_reward = 0
        self.cuda = cuda

    def get_obs(self):
        #  state 1: the cost of IP_link
        state_obj = torch.FloatTensor([self.graph_fiber[edge[0]][edge[1]]['cost'] if edge in self.graph_fiber.edges else 0.0 for edge in self.original_G_fiber.edges]).reshape((self.action_space.n, 1))

        #  state 2: the existence of IP_link
        state_flag = torch.FloatTensor([1.0 if edge in self.graph_fiber.edges else 0.0 for edge in self.original_G_fiber.edges]).reshape((self.edge_num, 1))

        #  the input of Graph filter
        X, adj_e, adj_v, T = cal_GNN_input(self.graph_fiber)

        #  the feature of edge  kernel value
        if self.stru == "dual-homing":
            Z = stru_encoder("dual-homing", self.original_G_fiber, self.graph_fiber, self.node_edge, self.edge_to_idx, self.edge_to_ring)
            state_dual = torch.FloatTensor(self.edge_feature)
            Z = torch.FloatTensor([[Z[i], 1] for i in range(len(state_flag)) if state_flag[i] == 1]).reshape((len(self.graph_fiber.edges), 2))

            if self.cuda:
                X, Z, adj_e, adj_v, T, state_obj, state_flag, state_dual = X.cuda(), Z.cuda(), adj_e.cuda(), adj_v.cuda(), T.cuda(), state_obj.cuda(), state_flag.cuda(), state_dual.cuda()
            obs = tuple([X, Z, adj_e, adj_v, T, state_obj, state_flag, state_dual])

        elif self.stru == "partial-mesh":
            Z = stru_encoder("mesh", self.original_G_fiber, self.graph_fiber, self.node_edge, self.edge_to_idx)
            state_mesh = torch.FloatTensor([Z[i] - 1 for i in range(len(state_flag)) if state_flag[i] == 1]).reshape((len(self.graph.edges), 1))
            Z = torch.FloatTensor([Z[i] for i in range(len(state_flag)) if state_flag[i] == 1]).reshape((len(self.graph.edges), 1))
            if self.cuda:
                X, Z, adj_e, adj_v, T, state_obj, state_flag, state_mseh = X.cuda(), Z.cuda(), adj_e.cuda(), adj_v.cuda(), T.cuda(), state_obj.cuda(), state_flag.cuda(), state_mesh.cuda()
            obs = tuple([X, Z, adj_e, adj_v, T, state_obj, state_flag, state_mesh])

        elif self.stru == "hybrid ring-mesh":
            Z = stru_encoder("hybrid ring-mesh", self.original_graph, self.graph, self.node_edge, self.edge_to_idx, self.edge_to_ring)

            state_dual = torch.FloatTensor(self.edge_feature)
            state_mesh = torch.FloatTensor([Z[i] - 1 for i in range(len(state_flag)) if state_flag[i] == 1]).reshape((len(self.graph.edges), 1))
            Z = torch.FloatTensor([[Z[i], 1] for i in range(len(state_flag)) if state_flag[i] == 1]).reshape((len(self.graph.edges), 2))

            if self.cuda:
                X, Z, adj_e, adj_v, T, state_obj, state_flag, state_dual, state_mseh = X.cuda(), Z.cuda(), adj_e.cuda(), adj_v.cuda(), T.cuda(), state_cost.cuda(), state_flag.cuda(), state_dual.cuda(), state_mesh.cuda()
            obs = tuple([X, Z, adj_e, adj_v, T, state_obj, state_flag, state_dual, state_mesh])

        return obs

    def step(self, action, obs):
        done, info = True, None
        for action_i in action:
            node_i, node_j = self.idx_to_edge[int(action_i)]
            if (node_i, node_j) not in self.graph_fiber.edges:  # vaild edge
                reward = -5.0
            else:
                done = False
                if self.stru == "dual-homing" or self.stru == "hybrid ring-mesh":
                    _, self.edge_to_ring, self.edge_feature = remove_edge(self.graph_fiber, self.edge_to_ring, self.edge_feature, (node_i, node_j))
                else:
                    self.graph_fiber.remove_edge(node_i, node_j)
                for (node_1, node_2) in self.graph_IP.edges:
                    if (node_i in self.graph_IP[node_1][node_2]['fiber']) and (node_j in self.graph_IP[node_1][node_2]['fiber']):
                        node_i_index = self.graph_IP[node_1][node_2]['fiber'].index(node_i)
                        node_j_index = self.graph_IP[node_1][node_2]['fiber'].index(node_j)
                        if abs(node_i_index - node_j_index) == 1:
                            self.graph_IP.remove_edge(node_1, node_2)

                self.node_edge[node_i].remove((node_i, node_j))
                self.node_edge[node_j].remove((node_i, node_j))

        if done == False:
            #  heuristic method and determine whether the result meets the constraints
            flag, topo, topo_fiber, traffic_path, obj = Heur(self.stru, "cost", self.graph_fiber, self.graph_IP, self.traffic, self.SRLG)

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
        self.graph_fiber = copy.deepcopy(self.original_G_fiber)
        self.graph_IP = copy.deepcopy(self.original_G_IP)
        self.done_topo = copy.deepcopy(self.original_heur_topo)
        self.done_topo_fiber = copy.deepcopy(self.original_heur_topo_fiber)
        if self.stru == "dual-homing" or self.stru == "hybrid ring-mesh":
            self.edge_to_ring = copy.deepcopy(self.original_edge_to_ring)
            self.edge_feature = copy.deepcopy(self.original_edge_feature)
        self.node_edge = copy.deepcopy(self.original_node_edge)
        obs = self.get_obs()
        self.obj = self.original_obj
        self.cum_reward = 0

        return obs

