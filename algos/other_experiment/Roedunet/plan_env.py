import gym
import torch
import copy
import numpy as np
import networkx as nx
from gym import spaces
import scipy.sparse as sp
import pandas as pd
from algos.save_result import save_result
from algos.other_experiment.Roedunet.ring import rings_flag
from algos.other_experiment.Roedunet.traffic import calaulate_traffic, calaulate_traffic_original



def create_edge_adj(vertex_adj):
    '''
    create an edge adjacency matrix from vertex adjacency matrix
    '''
    vertex_adj.setdiag(0)
    edge_index = np.nonzero(sp.triu(vertex_adj, k=1))
    num_edge = int(len(edge_index[0]))
    edge_name = [x for x in zip(edge_index[0], edge_index[1])]

    edge_adj = np.zeros((num_edge, num_edge))
    for i in range(num_edge):
        for j in range(i, num_edge):
            if len(set(edge_name[i]) & set(edge_name[j])) == 0:
                edge_adj[i, j] = 0
            else:
                edge_adj[i, j] = 1
    adj = edge_adj + edge_adj.T
    np.fill_diagonal(adj, 1)
    return sp.csr_matrix(adj), edge_name


def create_transition_matrix(vertex_adj):
    '''create N_v * N_e transition matrix'''
    vertex_adj.setdiag(0)
    edge_index = np.nonzero(sp.triu(vertex_adj, k=1))
    num_edge = int(len(edge_index[0]))
    edge_name = [x for x in zip(edge_index[0], edge_index[1])]

    row_index = [i for sub in edge_name for i in sub]
    col_index = np.repeat([i for i in range(num_edge)], 2)

    data = np.ones(num_edge * 2)
    T = sp.csr_matrix((data, (row_index, col_index)),
               shape=(vertex_adj.shape[0], num_edge))

    return T

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1)).astype("float")
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


class GraphEnv(gym.Env):
    def __init__(
        self, original_graph:nx.Graph, fiber_graph:nx.Graph, traffic: list, cuda, del_num, node_list, output_path
    ):
        self.original_graph = original_graph
        self.graph = copy.deepcopy(original_graph)
        self.fiber_graph = fiber_graph
        self.SRLG = []
        self.node_list = node_list
        _, _, traffic, self.flow_path = calaulate_traffic_original(self.original_graph, traffic, self.SRLG, self.node_list)
        self.traffic = traffic
        self.edge_num = len(original_graph.edges)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.edge_num, ))
        self.action_space = spaces.Discrete(self.edge_num)
        self.del_num = del_num

        # edge numbering
        self.idx_to_edge = {}
        self.edge_to_idx = {}
        for i, (node_i, node_j) in enumerate(original_graph.edges):
            self.idx_to_edge[i] = (node_i, node_j)
            self.edge_to_idx [(node_i, node_j)] = i
        self.cum_reward = 0

        cost = 0
        for edge in self.graph.edges:
            cost += self.original_graph[edge[0]][edge[1]]['cost']

        self.original_cost = cost
        self.cost = cost
        self.best_cost = np.inf
        self.done_topo = copy.deepcopy(self.original_graph)
        print("fiber_cost=", self.cost)
        self.original_node_edge = {node: [] for node in self.original_graph.nodes}
        for edge in self.original_graph.edges:
            self.original_node_edge[edge[0]].append(edge)
            self.original_node_edge[edge[1]].append(edge)
        self.node_edge = copy.deepcopy(self.original_node_edge)

        self.cuda = cuda

        self.output_path = output_path

    def get_obs(self):
        state_cost = torch.tensor([self.original_graph[edge[0]][edge[1]]['cost'] for edge in self.original_graph.edges]).reshape((self.edge_num, 1))
        state_flag = torch.tensor([1.0 if edge in self.graph.edges else 0.0 for edge in self.original_graph.edges]).reshape((self.edge_num, 1))
        state_feature = torch.tensor([1.0 if (edge[0] not in self.node_list) and (edge[1] not in self.node_list) and ( edge in self.graph.edges) else 0.0 for edge in self.original_graph.edges]).reshape((self.edge_num, 1))
        X = torch.FloatTensor(np.ones((len(self.graph.nodes), 1)))
        Z = torch.FloatTensor(np.ones((len(self.graph.edges), 1)))
        adj = nx.adjacency_matrix(self.graph)
        adj_v = sparse_mx_to_torch_sparse_tensor(normalize(adj + sp.eye(adj.shape[0])))
        eadj, edge_name = create_edge_adj(adj)
        adj_e = sparse_mx_to_torch_sparse_tensor(normalize(eadj))
        T = create_transition_matrix(adj)
        T = sparse_mx_to_torch_sparse_tensor(T)
        if self.cuda:
            X, Z, adj_e, adj_v, T, state_cost, state_flag, state_feature = X.cuda(), Z.cuda(), adj_e.cuda(), adj_v.cuda(), T.cuda(), state_cost.cuda(), state_flag.cuda(), state_feature.cuda()
        obs = tuple([X, Z, adj_e, adj_v, T, state_cost, state_flag, state_feature])
        return obs

    def step(self, action, obs):
        ring_flag, done, info = 0, True, None
        for action_i in action:
            node_i, node_j = self.idx_to_edge[int(action_i)]
            if (node_i, node_j) not in self.graph.edges:
                reward = -5.0
            else:
                done = False
                self.graph.remove_edge(node_i, node_j)
                self.node_edge[node_i].remove((node_i, node_j))
                self.node_edge[node_j].remove((node_i, node_j))
        if done == False:
            topo, cost, traffic_flag, flow_path = calaulate_traffic(self.graph, self.traffic, self.SRLG, self.node_list)
            if traffic_flag == 1:
                print("traffic_vaild")
                done = True
                reward = -5.0
            else:
                ring_flag = rings_flag(topo, ring_len=4, node_list=self.node_list)
                if ring_flag == 1:
                    print("ring_vaild")
                    done = True
                    reward = -5.0
                else:
                    cost = 0
                    for edge in self.graph.edges:
                        cost += self.original_graph[edge[0]][edge[1]]['cost']
                    if cost > self.cost:
                        print("vaild")
                        done = True
                        reward = -5.0
                    else:
                        obs = self.get_obs()
                        reward = (self.cost - cost)/50
                        self.cost = cost
                        if self.best_cost > cost:
                            save_result(self.output_path, topo=topo, traffic_path=flow_path)

            self.cum_reward += reward

        return obs, reward, done, ring_flag, self.cost, info



    def reset(self):
        self.graph = copy.deepcopy(self.original_graph)
        self.node_edge = copy.deepcopy(self.original_node_edge)
        obs = self.get_obs()
        self.cost = self.original_cost
        self.cum_reward = 0

        return obs
