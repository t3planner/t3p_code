import gym
import copy
import torch

import numpy as np
import networkx as nx

from gym import spaces

from NeuroPlan.utils import check_edge
from NeuroPlan.traffic_allocation import allocate_traffic


class Env(gym.Env): 
    def __init__(
        self, original_graph:nx.Graph, traffic:dict, SRLG:list, 
        delta_load, max_deltas, feature_dim, time_limit=None
    ):
        self.original_graph = original_graph
        self.graph = copy.deepcopy(self.original_graph)
        self.traffic = traffic
        self.SRLG = SRLG

        self.edges_num = len(self.original_graph.edges)
        self.delta_load = delta_load
        self.max_deltas = max_deltas
        self.feature_dim = feature_dim
        self.obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.edges_num, self.edges_num+self.feature_dim))
        self.act_space = spaces.Discrete(self.edges_num*self.max_deltas)

        self.time_limit = time_limit
        
        self.idx_to_edge = {}
        for idx, (node_i, node_j) in enumerate(self.original_graph.edges):
            self.idx_to_edge[idx] = (node_i, node_j)

        self.path_cost = 0  # cost of edges in all paths (i.e. cost of final topology)
        self.load_cost = 0  # cost of loads on all edges
        self.total_cost = 0

        self.solution = {'path_cost':np.inf, 'load_cost':np.inf, 'total_cost':np.inf, 'acts':[]}
        self.best_solution = {'path_cost':np.inf, 'load_cost':np.inf, 'total_cost':np.inf, 'acts':[]}


    # convert graph into adjacency matrix
    def preprocess(self, graph:nx.Graph): 
        edges = list(self.original_graph.edges)
        # edges = list(graph.edges)
        adj_matrix = np.zeros((self.edges_num, self.edges_num))
        for i in range(self.edges_num): 
            for j in range(i+1, self.edges_num): 
                (src_i, dst_i) = edges[i]
                (src_j, dst_j) = edges[j]
                if src_i == dst_j or dst_i == src_j: 
                    adj_matrix[i][j] = 1
                    adj_matrix[j][i] = 1
        return adj_matrix


    # edge feature: load and cost of each edge
    def get_edge_feature(self, graph:nx.Graph, feature_dim): 
        original_edges = list(self.original_graph.edges)
        edge_feature = np.zeros((self.edges_num, feature_dim))
        for i in range(self.edges_num): 
            (src, dst) = original_edges[i]
            if (src, dst) in graph.edges:
                edge_feature[i][0] = graph[src][dst]['cost']
                edge_feature[i][1] = graph[src][dst]['load']

        # standardization
        mean = np.mean(edge_feature)
        std = np.std(edge_feature)
        eps = np.finfo(np.float32).eps
        edge_feature = (edge_feature-mean)/(std+eps)
        return edge_feature


    # get mask for each action
    # 0: action is infeasible, will violate capacity constrain
    # 1: action is feasible
    def get_act_mask(self, graph:nx.Graph):
        original_edges = list(self.original_graph.edges)
        mask = [0]*self.edges_num*self.max_deltas
        for idx in range(self.edges_num):
            (src, dst) = original_edges[idx]
            for i in range(1, self.max_deltas+1): 
                delta_load = self.delta_load*i
                # load capacity constrain
                if (graph[src][dst]['load']+delta_load) <= graph[src][dst]['cap']:  
                    mask[idx*self.max_deltas+i-1] = 1
        return mask


    def get_obs(self, interm=False): 
        A = self.preprocess(self.graph)
        H = self.get_edge_feature(self.graph, self.feature_dim)

        A_hat = A + np.eye(A.shape[0])
        D = np.diag(np.sqrt(np.sum(A_hat, axis=1)**(-1)))
        A_adjust = np.matmul(np.matmul(D, A_hat), D)

        obs = np.concatenate((A_adjust, H), axis=1)
        mask = np.asarray(self.get_act_mask(self.graph))

        allocated = False
        traffic_paths = allocate_traffic(self.graph, self.traffic, self.SRLG, self.time_limit)

        if traffic_paths is not None:
            allocated = True    # ALL traffic demands are allocated

        path_cost = 0
        load_cost = 0
        for (node_i, node_j) in self.graph.edges: 
            if allocated:
                for paths in traffic_paths.values():
                    if check_edge(node_i, node_j, paths[0]): 
                        path_cost += self.graph[node_i][node_j]['cost']
                        break
                    elif check_edge(node_i, node_j, paths[1]): 
                        path_cost += self.graph[node_i][node_j]['cost']
                        break
            if self.graph[node_i][node_j]['load'] > 0:
                load_cost += self.graph[node_i][node_j]['load']

        total_cost = path_cost + load_cost

        if interm: 
            return obs, mask, allocated, path_cost, load_cost, total_cost
        else:
            return obs, mask


    def step(self, action): 
        obs, reward, done, info = None, None, False, None

        action = int(action)
        idx = action//self.max_deltas       # get idx of target edge
        delta_load = self.delta_load*((action%self.max_deltas)+1)       # get capacity increment of target edge
        (src, dst) = self.idx_to_edge[idx]  # convert idx into node pair
        if (src, dst) in self.graph.edges:
            self.graph[src][dst]['load'] += delta_load      # increase load capacity of target edge
            self.solution['acts'].append([(src, dst), delta_load])      # record action of current step
        
        obs, mask, allocated, path_cost, load_cost, total_cost = self.get_obs(interm=True)
        reward = (self.total_cost-total_cost)/100

        if allocated:
            print('Allocation success.')
            done = True

        print('Total Cost: {}'.format(total_cost))
        print('Reward: {}\n'.format(reward))

        self.path_cost = path_cost
        self.load_cost = load_cost
        self.total_cost = total_cost
        
        self.solution['path_cost'] = path_cost
        self.solution['load_cost'] = load_cost
        self.solution['total_cost'] = total_cost

        return obs, mask, reward, done, info


    def reset(self): 
        self.graph = copy.deepcopy(self.original_graph)

        obs, mask, allocated, path_cost, load_cost, total_cost = self.get_obs(interm=True)

        self.path_cost = path_cost 
        self.load_cost = load_cost
        self.total_cost = total_cost
        
        self.solution = {'path_cost':np.inf, 'load_cost':np.inf, 'total_cost':np.inf, 'acts':[]}

        return obs, mask