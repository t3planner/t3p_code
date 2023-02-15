import copy
import random
import gurobipy

import heapq as hp
import numpy as np
import networkx as nx

from OWAN.utils import check_dual
from OWAN.traffic import allocate_traffic


def remove_cycle(path:list):
    final_path = []
    path_nodes = set()

    for node in path:
        if node not in path_nodes:
            path_nodes.add(node)
            final_path.append(node)
        else:
            idx = final_path.index(node)
            for i in range(idx+1, len(final_path)): 
                path_nodes.remove(final_path[i])

            del final_path[idx+1:]

    return final_path


def dijkstra(graph: nx.Graph, src, dst=None):
    priority_queue = [(0, src, None)]   # (cost, node, parent)
    node_parent = {node: None for node in graph.nodes}
    node_cost = {node: None for node in graph.nodes}

    while priority_queue:
        cost, node, parent = hp.heappop(priority_queue)
        if node_cost[node] is None:
            node_parent[node] = parent
            node_cost[node] = cost

            if node == dst:
                path = [dst]
                while node != src:
                    node = node_parent[node]
                    path.append(node)
                path.reverse()
                return path
            
            for neighbor in graph.adj[node]: 
                if node_parent[neighbor] is None:
                    hp.heappush(priority_queue, (cost+graph[node][neighbor]['cost'], neighbor, node))

    if dst != None and node_cost[dst] == None: 
        return []
    
    return node_parent, node_cost


def suurballe(G:nx.Graph, src, dst): 
    graph = G.to_directed()

    node_parent, node_cost = dijkstra(graph, src)
    node = dst
    path_1 = [dst]
    while node != src:
        node = node_parent[node]
        path_1.append(node)
    path_1.reverse()

    for node_i in graph.nodes:
        for node_j in graph.adj[node_i]:
            if node_cost[node_i]==None or node_cost[node_j]==None:
                graph[node_i][node_j]['cost'] = np.inf
            else:
                graph[node_i][node_j]['cost'] += node_cost[node_i]-node_cost[node_j]
    
    for i in range(len(path_1)-1):
        graph.remove_edge(path_1[i], path_1[i+1])
        graph[path_1[i+1]][path_1[i]]['cost'] = 0
    
    path_2 = dijkstra(graph, src, dst)

    done = False
    while not done: 
        end = len(path_2)-1
        for i in range(1, end):
            if path_2[i] in path_1:
                j = path_1.index(path_2[i])
                if path_1[j-1] == path_2[i+1]: 
                    p_1 = path_1[:j-1]+path_2[i+1:]
                    p_2 = path_2[:i]+path_1[j:]
                    path_1 = p_1
                    path_2 = p_2
                    break
        if i >= end-1:
            done = True

    path_1 = remove_cycle(path_1)
    path_2 = remove_cycle(path_2)

    return path_1, path_2


def check_edge(node_i, node_j, path):
    for i in range(len(path)-1):
        if node_i == path[i] and node_j == path[i+1]:
            return True
        elif node_j == path[i] and node_i == path[i+1]:
            return True

    return False


def update_graph(load_graph:nx.Graph, path, demand):
    for i in range(len(path)-1):
        node_i = path[i]
        node_j = path[i+1]
        load_graph[node_i][node_j]['load'] += demand
    
    return load_graph


# transmit delta_cap of load capacity from (src_i, src_j) to (dst_i, dst_j)
def state_transition(original_graph:nx.Graph, graph:nx.Graph, delta_cap=10):
    edges = []
    for (src_i, src_j) in graph.edges: 
        if graph[src_i][src_j]['cap']-delta_cap > 0:    # current capacity of (src_i, src_j) should be larger than delta_cap
            edges.append((src_i, src_j))
    (src_i, src_j) = random.sample(edges, 1)[0]
    graph[src_i][src_j]['cap'] = graph[src_i][src_j]['cap']-delta_cap

    transitions = list(original_graph.edges)
    (dst_i, dst_j) = random.sample(transitions, 1)[0]
    if (dst_i, dst_j) in graph.edges:
        graph[dst_i][dst_j]['cap'] = graph[dst_i][dst_j]['cap']+delta_cap
    else: 
        graph.add_edge(
            dst_i, dst_j, 
            cost=original_graph[dst_i][dst_j]['cost'], 
            load=0, 
            cap=delta_cap
        )

    return graph


# remove (src_i, src_j), transmit all of its capacity to (dst_i, dst_j)
# def state_transition(original_graph:nx.Graph, graph:nx.Graph):
#     edges = list(graph.edges)
#     (src_i, src_j) = random.sample(edges, 1)[0]

#     transitions = []
#     for (dst_i, dst_j) in original_graph.edges: 
#         if (dst_i, dst_j) not in graph.edges:
#             transitions.append((dst_i, dst_j))
#     (dst_i, dst_j) = random.sample(transitions, 1)[0]

#     graph.remove_edge(src_i, src_j)
#     graph.add_edge(
#         dst_i, dst_j, 
#         cost=original_graph[dst_i][dst_j]['cost'], 
#         load=0, 
#         cap=original_graph[dst_i][dst_j]['cap']
#     )

#     return graph


def annealing(original_graph:nx.Graph, init_graph:nx.Graph, T, iters, alpha, traffic:dict, SRLG_pair:dict={}, ring_len=10):
    graph = copy.deepcopy(init_graph)
    loads, max_util, unallocated, paths = allocate_traffic(graph, traffic, SRLG_pair)

    E_0 = max_util
    load_graph = loads
    traffic_paths = paths
    while True:
        T = alpha*T
        for i in range(iters): 
            graph = copy.deepcopy(init_graph)

            graph = state_transition(original_graph, graph)
            
            loads, max_util, unallocated, paths = allocate_traffic(graph, traffic, SRLG_pair)
            dual_num, agg_num = check_dual(graph, ring_len)
            
            # state transition
            if unallocated==0 and dual_num==agg_num:
                E_t = max_util
                print(E_t)
                if E_t < E_0: 
                    init_graph = graph
                    load_graph = loads
                    traffic_paths = paths
                    E_0 = E_t
                else: 
                    r = random.random()
                    e = np.exp((E_0-E_t)/T)
                    if r < e: 
                        init_graph = graph
                        load_graph = loads
                        traffic_paths = paths
                        E_0 = E_t

            # terminal condition
            if T < 0.1:
                return init_graph, load_graph, traffic_paths


def simulated_annealing(
    original_graph:nx.Graph, init_graph:nx.Graph, original_traffic:dict, SRLG_pair:dict={}, 
    alpha=0.95, division=10, ring_len=10
):
    fiber_net = nx.Graph()
    fiber_net.add_nodes_from(original_graph)

    load_graph = nx.Graph()
    for (node_i, node_j) in original_graph.edges:
        load_graph.add_edge(node_i, node_j, load=0)

    # initial temperature for annealing
    T = 0
    for demand in original_traffic.values(): 
        T += demand
    T = T/len(original_traffic)

    # num of iterations before temperature decrease
    iters = len(original_graph.edges)//division

    fiber_net, load_graph, traffic_paths = annealing(original_graph, init_graph, T, iters, alpha, original_traffic, SRLG_pair, ring_len)

    return fiber_net, load_graph, traffic_paths