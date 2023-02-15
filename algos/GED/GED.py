import os
import time
import torch

import numpy as np
import pandas as pd
import networkx as nx
import scipy.optimize as opt
import torch_geometric as pyg

from algos.GED.sinkhorn import Sinkhorn
from torch_geometric.utils import to_dense_batch, to_dense_adj, from_networkx


def hung_kernel(s:torch.Tensor, src_n=None, dst_n=None, mask=None):
    if mask is None:
        if src_n is None:
            src_n = s.shape[0]
        if dst_n is None:
            dst_n = s.shape[1]
        row, col = opt.linear_sum_assignment(s[:src_n, :dst_n])
    else:
        mask = mask.cpu()
        s_mask = s[mask]
        if s_mask.size > 0:
            dim0 = torch.sum(mask, dim=0).max()
            dim1 = torch.sum(mask, dim=1).max()
            row, col = opt.linear_sum_assignment(s_mask.reshape(dim0, dim1))
            row = torch.nonzero(torch.sum(mask, dim=1), as_tuple=True)[0][row]
            col = torch.nonzero(torch.sum(mask, dim=0), as_tuple=True)[0][col]
        else:
            row, col = [], []
    perm_mat = np.zeros_like(s)
    perm_mat[row, col] = 1
    return perm_mat


def hungarian(s:torch.Tensor, src_n=None, dst_n=None, mask=None):
    """
    Solve optimal LAP permutation by hungarian algorithm.
    :param s: input 3d tensor (first dimension represents batch)
    :param n1: [num of objs in dim1] (against padding)
    :param n2: [num of objs in dim2] (against padding)
    :param nproc: number of parallel processes (default =1 for no parallel)
    :return: optimal permutation matrix
    """
    if len(s.shape) == 2:
        s = s.unsqueeze(0)
        matrix_input = True
    elif len(s.shape) == 3:
        matrix_input = False
    else:
        raise ValueError('Invalid input data shape.')

    device = s.device
    batch_num = s.shape[0]

    if src_n is not None:
        src_n = src_n.cpu().numpy()
    else:
        src_n = [None]*batch_num
    if dst_n is not None:
        dst_n = dst_n.cpu().numpy()
    else:
        dst_n = [None]*batch_num
    
    if mask is None:
        mask = [None]*batch_num

    perm_mat = s.cpu().detach().numpy()*-1
    perm_mat = np.stack([hung_kernel(perm_mat[b], src_n[b], dst_n[b], mask[b]) for b in range(batch_num)])
    perm_mat = torch.from_numpy(perm_mat).to(device)

    if matrix_input:
        perm_mat.squeeze_(0)

    return perm_mat


def hungarian_lap(node_cost_mat, src_n, dst_n):
    device = node_cost_mat.device
    upper_left = node_cost_mat[:src_n, :dst_n]
    upper_right = torch.full((src_n, src_n), float('inf'), device=device)
    torch.diagonal(upper_right)[:] = node_cost_mat[:-1, -1]
    lower_left = torch.full((dst_n, dst_n), float('inf'), device=device)
    torch.diagonal(lower_left)[:] = node_cost_mat[-1, :-1]
    lower_right = torch.zeros((dst_n, src_n), device=device)

    large_cost_mat = torch.cat((torch.cat((upper_left, upper_right), dim=1),
                                torch.cat((lower_left, lower_right), dim=1)), dim=0)

    large_pred_x = hungarian(-large_cost_mat)
    pred_x = torch.zeros_like(node_cost_mat)
    pred_x[:src_n, :dst_n] = large_pred_x[:src_n, :dst_n]
    pred_x[:-1, -1] = torch.sum(large_pred_x[:src_n, dst_n:], dim=1)
    pred_x[-1, :-1] = torch.sum(large_pred_x[src_n:, :dst_n], dim=0)

    ged_lower_bound = torch.sum(pred_x*node_cost_mat)

    return pred_x, ged_lower_bound


def heuristic_prediction_hun(k, src_n, dst_n, partial_pmat=None):
    k_prime = k.reshape(-1, src_n+1, dst_n+1)
    node_costs = torch.empty(k_prime.shape[0], device=k.device)
    for i in range(k_prime.shape[0]):
        _, node_costs[i] = hungarian_lap(k_prime[i], src_n, dst_n)

    node_cost_mat = node_costs.reshape(src_n+1, dst_n+1)
    if partial_pmat is None:
        partial_pmat = torch.zeros_like(node_cost_mat)
    src_mask = ~partial_pmat.sum(dim=-1).to(dtype=torch.bool)
    dst_mask = ~partial_pmat.sum(dim=-2).to(dtype=torch.bool)
    src_mask[-1] = 1
    dst_mask[-1] = 1
    node_cost_mat = node_cost_mat[src_mask, :]
    node_cost_mat = node_cost_mat[:, dst_mask]

    x, lb = hungarian_lap(node_cost_mat, torch.sum(src_mask[:-1]), torch.sum(dst_mask[:-1]))
    return x, lb


def hungarian_ged(k, src_n, dst_n):
    x, _ = heuristic_prediction_hun(k, src_n, dst_n)
    return x


def calculate_ged(_x, _k):
    if len(_x.shape) == 3 and len(_k.shape) == 3:
        _batch = _x.shape[0]
        return torch.bmm(torch.bmm(_x.reshape(_batch, 1, -1), _k), _x.reshape(_batch, -1, 1)).view(_batch)
    elif len(_x.shape) == 2 and len(_k.shape) == 2:
        return torch.mm(torch.mm(_x.reshape( 1, -1), _k), _x.reshape( -1, 1)).view(1)
    else:
        raise ValueError('Input dimensions not supported.')


def ipfp_ged(k, src_n, dst_n, max_iter=100):
    v = hungarian_ged(k, src_n, dst_n)
    #v = torch.ones(n1 + 1, n2 + 1, dtype=k.dtype, device=k.device) / (n1 + 1) / (n2 + 1)
    last_v = v
    best_binary_sol = v
    src_n, dst_n = torch.tensor([src_n], device=k.device), torch.tensor([dst_n], device=k.device)

    #k_diag = torch.diag(k)
    #k_offdiag = k - torch.diag(k_diag)
    best_upper_bound = float('inf')

    for i in range(max_iter):
        cost = torch.mm(k, v.view(-1, 1)).reshape(src_n+1, dst_n+1)
        binary_sol = hungarian_lap(cost, src_n, dst_n)[0]
        upper_bound = calculate_ged(binary_sol, k)
        if upper_bound < best_upper_bound:
            best_binary_sol = binary_sol
            best_upper_bound = upper_bound
        alpha = torch.mm(torch.mm(v.view(1, -1), k), (binary_sol-v).view(-1, 1)) #+ \
                #torch.mm(k_diag.view(1, -1), (binary_sol - v).view(-1, 1))
        beta = calculate_ged(binary_sol - v, k)
        t0 = -alpha/beta
        if beta <= 0 or t0 >= 1:
            v = binary_sol
        else:
            v = v+t0*(binary_sol-v)
        last_v_sol = calculate_ged(last_v, k)
        if torch.abs(last_v_sol-torch.mm(cost.view(1, -1), binary_sol.view(-1, 1)))/last_v_sol < 1e-3:
            break
        last_v = v

    pred_x = best_binary_sol
    return pred_x


def rrwm_ged(k, src_n, dst_n, max_iter=100, sk_iter=100, alpha=0.2, beta=100):
    d = k.sum(dim=1, keepdim=True)
    d_max = d.max(dim=0, keepdim=True).values
    k = k/(d_max+d.min()*1e-5)

    #v = torch.ones(n1+1, n2+1, dtype=k.dtype, device=k.device) / (n1 + 1) / (n2 + 1)
    v = hungarian_ged(k, src_n, dst_n)
    v = v.reshape(-1, 1)
    src_n, dst_n = torch.tensor([src_n], device=k.device), torch.tensor([dst_n], device=k.device)

    for i in range(max_iter):
        last_v = v
        v = torch.mm(k, v)
        n = torch.norm(v, p=1, dim=0, keepdim=True)
        v = v/n
        v = alpha*Sinkhorn(max_iter=sk_iter, tau=v.max()/beta)(-v.view(src_n+1, -1), src_n+1, dst_n+1).reshape(-1, 1)+(1-alpha)*last_v
        n = torch.norm(v, p=1, dim=0, keepdim=True)
        v = torch.matmul(v, 1/n)

        if torch.norm(v-last_v) < 1e-3:
            break

    pred_x, lb = hungarian_lap(-v.view(src_n+1, -1), src_n, dst_n)
    return pred_x


def ga_ged(k, src_n, dst_n, max_iter=100, sk_iter=10, tau_init=1, tau_min=0.1, gamma=0.95):
    assert len(k.shape) == 2
    v = torch.ones(src_n+1, dst_n+1, dtype=k.dtype, device=k.device)/(src_n+1)/(dst_n+1)
    #v = hungarian_ged(k, n1, n2)
    v = v.reshape(-1, 1)
    src_n, dst_n = torch.tensor([src_n], device=k.device), torch.tensor([dst_n], device=k.device)
    tau = tau_init

    while tau >= tau_min:
        for i in range(max_iter):
            last_v = v
            v = torch.mm(k, v)
            v = Sinkhorn(max_iter=sk_iter, tau=tau)(-v.view(src_n+1, -1), src_n+1, dst_n+1).reshape(-1, 1)
            if torch.norm(v - last_v) < 1e-4:
                break
        tau = tau*gamma

    for i in range(max_iter):
        last_v = v
        v = torch.mm(k, v)
        v, _ = hungarian_lap(-v.reshape(src_n+1, -1), src_n, dst_n)
        v = v.view(-1, 1)
        if torch.norm(v-last_v) < 1e-4:
            break
    pred_x = v.reshape(src_n+1, -1)
    return pred_x


def construct_k(G_src, G_dst): 
    if isinstance(G_src, pyg.data.Data):
        G_src = pyg.data.Batch.from_data_list([G_src])
    if isinstance(G_dst, pyg.data.Data):
        G_dst = pyg.data.Batch.from_data_list([G_dst])

    src_edge_index = G_src.edge_index
    dst_edge_index = G_dst.edge_index
    if hasattr(G_src, 'edge_attr') and hasattr(G_dst, 'edge_attr'):
        src_edge_attr = G_src.edge_attr
        dst_edge_attr = G_dst.edge_attr
    else:
        src_edge_attr = None
        dst_edge_attr = None

    # src_node = G_src.x
    # dst_node = G_dst.x
    src_batch = G_src.batch if hasattr(G_src, 'batch') else torch.tensor((), dtype=torch.long).new_zeros(G_src.num_nodes)
    dst_batch = G_dst.batch if hasattr(G_dst, 'batch') else torch.tensor((), dtype=torch.long).new_zeros(G_dst.num_nodes)
    
    batch_num = G_src.num_graphs

    src_ns = torch.bincount(G_src.batch)
    dst_ns = torch.bincount(G_dst.batch)

    src_adj = to_dense_adj(src_edge_index, batch=src_batch, edge_attr=src_edge_attr)
    src_dummy_adj = torch.zeros(src_adj.shape[0], src_adj.shape[1]+1, src_adj.shape[2]+1)
    src_dummy_adj[:, :-1, :-1] = src_adj
    dst_adj = to_dense_adj(dst_edge_index, batch=dst_batch, edge_attr=dst_edge_attr)
    dst_dummy_adj = torch.zeros(dst_adj.shape[0], dst_adj.shape[1]+1, dst_adj.shape[2]+1)
    dst_dummy_adj[:, :-1, :-1] = dst_adj

    # src_node, _ = to_dense_batch(src_node, batch=src_batch)
    # src_dummy_node = torch.zeros(src_adj.shape[0], src_node.shape[1]+1, src_node.shape[-1])
    # src_dummy_node[:, :-1, :] = src_node
    # dst_node, _ = to_dense_batch(dst_node, batch=dst_batch)
    # dst_dummy_node = torch.zeros(dst_adj.shape[0], dst_node.shape[1]+1, dst_node.shape[-1])
    # dst_dummy_node[:, :-1, :] = dst_node

    # k_diag = node_metric(src_dummy_node, dst_dummy_node)

    src_mask = torch.zeros_like(src_dummy_adj)
    dst_mask = torch.zeros_like(dst_dummy_adj)
    for b in range(batch_num):
        src_mask[b, :src_ns[b]+1, :src_ns[b]+1] = 1
        src_mask[b, :src_ns[b], :src_ns[b]] -= torch.eye(src_ns[b], device=src_mask.device)
        dst_mask[b, :dst_ns[b]+1, :dst_ns[b]+1] = 1
        dst_mask[b, :dst_ns[b], :dst_ns[b]] -= torch.eye(dst_ns[b], device=dst_mask.device)

    a1 = src_dummy_adj.reshape(batch_num, -1, 1)
    a2 = dst_dummy_adj.reshape(batch_num, 1, -1)
    m1 = src_mask.reshape(batch_num, -1, 1)
    m2 = dst_mask.reshape(batch_num, 1, -1)
    k = torch.abs(a1-a2)*torch.bmm(m1, m2)
    k = k.reshape(batch_num, src_dummy_adj.shape[1], src_dummy_adj.shape[2], dst_dummy_adj.shape[1], dst_dummy_adj.shape[2])
    k = k.permute([0, 1, 3, 2, 4])
    k = k.reshape(batch_num, src_dummy_adj.shape[1]*dst_dummy_adj.shape[1], src_dummy_adj.shape[2]*dst_dummy_adj.shape[2])

    # for b in range(batch_num):
    #     k_diag_view = torch.diagonal(k[b])
    #     k_diag_view[:] = k_diag[b].reshape(-1)

    return k


def graph_edit_dist(G_src:nx.Graph, G_dst:nx.Graph, method='ipfp'):
    graph_src = from_networkx(G_src)
    graph_dst = from_networkx(G_dst)

    k = construct_k(graph_src, graph_dst).squeeze(0)
    if method == 'hungarian':
        x = hungarian_ged(k, graph_src.num_nodes, graph_dst.num_nodes)
    elif method == 'ipfp':
        x = ipfp_ged(k, graph_src.num_nodes, graph_dst.num_nodes)
    # elif method == 'beam':
    #     x = astar_ged(k, graph_src.num_nodes, graph_dst.num_nodes)
    elif method == 'rrwm':
        x = rrwm_ged(k, graph_src.num_nodes, graph_dst.num_nodes)
    elif method == 'ga':
        x = ga_ged(k, graph_src.num_nodes, graph_dst.num_nodes)
    else:
        raise NotImplementedError(f'{method} is not implemented.')

    cost = calculate_ged(x, k)
    return cost.item()
