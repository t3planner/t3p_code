import numpy as np
import scipy.sparse as sp
import networkx as nx
import torch

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

def cal_GNN_input(graph: nx.Graph):
    #  the feature of node
    X = torch.FloatTensor(np.ones((len(graph.nodes), 1)))
    adj = nx.adjacency_matrix(graph)
    adj_v = sparse_mx_to_torch_sparse_tensor(normalize(adj + sp.eye(adj.shape[0])))
    eadj, edge_name = create_edge_adj(adj)
    adj_e = sparse_mx_to_torch_sparse_tensor(normalize(eadj))
    T = create_transition_matrix(adj)
    T = sparse_mx_to_torch_sparse_tensor(T)

    return X, adj_e, adj_v, T