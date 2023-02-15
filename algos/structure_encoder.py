import numpy as np
import torch

def stru_encoder(stru, original_graph, graph, node_edge, edge_to_idx, edge_to_ring=None):
    Z = torch.FloatTensor(np.ones((len(original_graph.edges), 1)))
    if stru == "dual-homing":
        for node in graph:
            if len(node_edge[node]) == 2:  # if node's degree is 2, the value of the two edges corresponding to this node is high
                Z[edge_to_idx[node_edge[node][0]]] = 2
                Z[edge_to_idx[node_edge[node][1]]] = 2
            else:  # find the most number of edges in the ring
                chongfu_num = [len(
                    set(edge_to_ring[node_edge[node][i]]) & set(edge_to_ring[node_edge[node][j]]))
                               for i in range(len(node_edge[node])) for j in range(len(node_edge[node])) if
                               i < j]
                sum_num = [len(set(edge_to_ring[node_edge[node][i]])) for i in
                           range(len(node_edge[node]))]
                max_num = sum(sum_num) - sum(chongfu_num)
                for i in range(len(node_edge[node])):
                    if len(edge_to_ring[node_edge[node][i]]) == max_num:
                        Z[edge_to_idx[node_edge[node][i]]] = 2
    elif stru == "mesh":
        for node in graph.nodes:
            if len(node_edge[node]) == 3:  # if node's degree is 2, the value of the three edges corresponding to this node is high
                Z[edge_to_idx[node_edge[node][0]]] = 2
                Z[edge_to_idx[node_edge[node][1]]] = 2
                Z[edge_to_idx[node_edge[node][2]]] = 2
    elif stru == "hybrid ring-mesh":
        for node in graph:
            if len(node_edge[node]) == 2:  # if node's degree is 2, the value of the two edges corresponding to this node is high
                Z[edge_to_idx[node_edge[node][0]]] = 2
                Z[edge_to_idx[node_edge[node][1]]] = 2
            else:  # find the most number of edges in the ring
                chongfu_num = [len(
                    set(edge_to_ring[node_edge[node][i]]) & set(edge_to_ring[node_edge[node][j]]))
                               for i in range(len(node_edge[node])) for j in range(len(node_edge[node])) if
                               i < j]
                sum_num = [len(set(edge_to_ring[node_edge[node][i]])) for i in
                           range(len(node_edge[node]))]
                max_num = sum(sum_num) - sum(chongfu_num)
                for i in range(len(node_edge[node])):
                    if len(edge_to_ring[node_edge[node][i]]) == max_num:
                        Z[edge_to_idx[node_edge[node][i]]] = 2
        for node in graph.nodes:
            if graph.nodes()[node]['type'] != 'Acc' and len(node_edge[node]) == 3:
                Z[edge_to_idx[node_edge[node][0]]] = 2
                Z[edge_to_idx[node_edge[node][1]]] = 2
                Z[edge_to_idx[node_edge[node][2]]] = 2

    return Z
