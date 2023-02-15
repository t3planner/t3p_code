import copy
import random

import heapq as hp
import numpy as np
import networkx as nx

from OWAN.utils import check_dual


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


def dijkstra(graph:nx.Graph, src, dst=None):
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


def heuristic_solver(SRLG_graph:nx.Graph, traffic:dict):
    allocate_num = 0
    traffic_paths = {}
    for (src, dst) in traffic.keys():
        # traffic allocation, only with both main path and backup path will be accepted
        try: 
            path_1, path_2 = suurballe(SRLG_graph, src, dst)
            if len(path_1) > 0 and len(path_2) > 0:
                allocate_num += 1
                traffic_paths[(src, dst)] = [path_1, path_2]
        except:
            pass

    return allocate_num, traffic_paths


def annealing(SRLG_graph:nx.Graph, add_list:list, T, iters, alpha, traffic:dict, ring_len=10):
    E_0 = -T
    traffic_num = len(traffic)
    allocate_num, traffic_paths = heuristic_solver(SRLG_graph, traffic)

    # get availabe actions
    edges = []
    for (node_i, node_j) in SRLG_graph.edges: 
        if SRLG_graph[node_i][node_j]['cost'] > 0: 
            edges.append((node_i, node_j))

    while True:
        T = alpha*T
        for i in range(iters): 
            graph = copy.deepcopy(SRLG_graph)

            (node_i, node_j) = random.sample(edges, 1)[0]
            graph.remove_edge(node_i, node_j)

            cost = SRLG_graph[node_i][node_j]['cost']

            dual = check_dual(graph, add_list, ring_len)

            allocate_num, paths = heuristic_solver(graph, traffic)

            E_t = -cost
            # state transition 
            if E_t < E_0: 
                if dual and allocate_num==traffic_num:
                    SRLG_graph = graph
                    traffic_paths = paths
                    E_0 = E_t
                    edges.remove((node_i, node_j))
                else: 
                    edges.remove((node_i, node_j))
            else: 
                if dual and allocate_num==traffic_num:
                    r = random.random()
                    e = np.exp((E_0-E_t)/T)
                    if r < e: 
                        SRLG_graph = graph
                        traffic_paths = paths
                        E_0 = E_t
                        edges.remove((node_i, node_j))
                else: 
                    edges.remove((node_i, node_j))
                    
            # terminal condition
            if T < 1 or len(edges) < 1:
                return SRLG_graph, traffic_paths


def update_graph(fiber_net:nx.Graph, load_graph:nx.Graph, path, demand):
    for i in range(len(path)-1):
        node_i = path[i]
        node_j = path[i+1]

        fiber_net.add_edge(node_i, node_j)
        load_graph[node_i][node_j]['load'] += demand
    
    return fiber_net, load_graph


def simulated_annealing(
    original_graph:nx.Graph, SRLG_graph:nx.Graph, add_list:list, 
    original_traffic:dict, alpha=0.95, division=10, ring_len=10
):
    fiber_net = nx.Graph()
    fiber_net.add_nodes_from(original_graph)

    load_graph = nx.Graph()
    for (node_i, node_j) in original_graph.edges:
        load_graph.add_edge(node_i, node_j, load=0)

    # initial temperature for annealing
    T = 0
    for (node_i, node_j) in SRLG_graph.edges: 
        if SRLG_graph[node_i][node_j]['cost'] > T:
            T = SRLG_graph[node_i][node_j]['cost']
    
    # num of iterations before temperature decrease
    iters = len(SRLG_graph.edges)//division

    SRLG_fiber_net, SRLG_traffic_paths = annealing(SRLG_graph, add_list, T, iters, alpha, original_traffic, ring_len)

    traffic_paths = {}
    for (src, dst), paths in SRLG_traffic_paths.items(): 
        demand = original_traffic[(src, dst)]

        path_1 = paths[0]
        path_2 = paths[1]

        path_1 = remove_cycle(path_1)
        path_2 = remove_cycle(path_2)

        traffic_paths[(src, dst)] = [path_1, path_2]

        fiber_net, load_graph = update_graph(fiber_net, load_graph, path_1, demand)
        fiber_net, load_graph = update_graph(fiber_net, load_graph, path_2, demand)

    return fiber_net, load_graph, traffic_paths