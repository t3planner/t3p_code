import copy
import networkx as nx
import pandas as pd
import numpy as np
import queue
import time
import csv
import torch

def ring_isomorphism(graph, path, Core_num, Agg_num, Acc_num, length):  # find the dual-homing ring in the graph
    for node in path:
        isvisited[node] = True
    n = path[-1]
    for node in graph.adj[n]:
        if isvisited[node] == False:
            path.append(node)
            isvisited[node] = True
            l = 0
            for i in path:
                if graph.nodes()[i]['type'] == 'Core' or graph.nodes()[i]['type'] == 'Agg' or graph.nodes()[i]['type']=='Acc':
                    l += 1
            if l > length:  # the length of the ring
                path.pop()
                isvisited[node]=False
                break
            if (node < (Core_num+Agg_num)) and(node >= Core_num):  # found the dual-homing ring
                ring=copy.deepcopy(path)
                if ring[0]<ring[-1]:
                    cycle_path.append(ring)
                    for index in range(len(ring)-1):
                        graph[ring[index]][ring[index+1]]['ring'].append(len(cycle_path)-1)
                        graph[ring[index]][ring[index+1]]['num'] += 1
                isvisited[path[-1]]=False
                path.pop()
            else:
                ring_isomorphism(graph, path, Core_num, Agg_num, Acc_num, length)
    isvisited[path[-1]] = False
    path.pop()
    return graph



def subgraph_isomorphism_edge_counts(graph, length = np.inf):
    '''
    count the id and the number of rings corresponding to each edge of the graph
    '''
    node_type = [graph.nodes()[i]['type'] for i in range(len(graph.nodes))]
    Core_num = node_type.count('Core')
    Agg_num = node_type.count('Agg')
    Acc_num = node_type.count('Acc')

    # convert the graph to a thumbnail with attributes
    new_graph = copy.deepcopy(graph)
    for edge in new_graph.edges:
        new_graph[edge[0]][edge[1]]['ring'] = []
        new_graph[edge[0]][edge[1]]['num'] = 0
    q = queue.Queue()
    q.queue.clear()
    for i in new_graph.nodes: q.put(i)
    edge_dict = {}
    while not q.empty():
        node = q.get()
        if (new_graph.degree(node) == 2) and (node >= (Core_num+Agg_num)) and ((list(new_graph.adj[node])[0], list(new_graph.adj[node])[1]) not in new_graph.edges):
            new_graph.add_edge(list(new_graph.adj[node])[0], list(new_graph.adj[node])[1], ring = [], num =0)
            edge_dict[(node, list(new_graph.adj[node])[0])] = (list(new_graph.adj[node])[0], list(new_graph.adj[node])[1])
            edge_dict[(node, list(new_graph.adj[node])[1])] = (list(new_graph.adj[node])[0], list(new_graph.adj[node])[1])
            q.put(list(new_graph.adj[node])[0])
            q.put(list(new_graph.adj[node])[1])
            new_graph.remove_edge(node, list(new_graph.adj[node])[0])
            new_graph.remove_edge(node, list(new_graph.adj[node])[0])

    nb_nodes = max(new_graph.nodes)+1
    global isvisited, cycle_path
    isvisited=[]
    cycle_path=[]

    node_list = range(Core_num, Core_num+Agg_num)  # the ring can only start at the Agg node
    for node in node_list:
        isvisited=[False for i in range(nb_nodes)]
        new_graph = ring_isomorphism(new_graph, [node], Core_num, Agg_num, Acc_num, length)
    # map the thumbnail to the original graph
    edge_ring_list = {}
    edge_num = {}
    for edge in graph.edges():
        parent_edge = edge
        while parent_edge not in new_graph.edges:
            if parent_edge in edge_dict:
                parent_edge = edge_dict[parent_edge]
            else:
                parent_edge = edge_dict[(parent_edge[1], parent_edge[0])]
        edge_ring_list[edge] = new_graph[parent_edge[0]][parent_edge[1]]['ring']
        edge_num[edge] = new_graph[parent_edge[0]][parent_edge[1]]['num']
    edge_feature = torch.Tensor([[edge_num[edge]] for edge in edge_num])
    return edge_ring_list, edge_feature


def remove_edge(graph, edge_ring_list, edge_feature, edge):
    '''
    remove the edge
    '''
    edge_num = {list(graph.edges)[index]: edge_feature[index][0] for index in range(len(graph.edges))}
    ring_list = copy.deepcopy(edge_ring_list[edge])
    graph.remove_edge(edge[0], edge[1])
    for e in graph.edges:
        edge_num[e] = edge_num[e]-len(set(ring_list) & set(edge_ring_list[e]))
        edge_ring_list[e] = list(set(edge_ring_list[e]) - set(ring_list) & set(edge_ring_list[e]))
    edge_feature = torch.Tensor([[edge_num[edge]] for edge in graph.edges])
    return graph, edge_ring_list, edge_feature
