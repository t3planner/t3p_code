import time
import copy

import heapq as hp
import numpy as np
import networkx as nx


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


def update_graph(fiber_net:nx.Graph, load_graph:nx.Graph, path, demand):
    for i in range(len(path)-1):
        node_i = path[i]
        node_j = path[i+1]
        fiber_net.add_edge(node_i, node_j)
        load_graph[node_i][node_j]['load'] += demand
    
    return fiber_net, load_graph


def remove_path(graph:nx.Graph, path:list):
    for i in range(len(path)-1):
        node_i = path[i]
        node_j = path[i+1]
        graph.remove_edge(node_i, node_j)
    return graph


def allocate_traffic(k, G:nx.Graph, src, dst):
    # try to find k disjoint paths for a traffic demand
    paths = []
    graph = copy.deepcopy(G)
    path_1, path_2 = suurballe(graph, src, dst)
    if path_1 != []:
        paths.append(path_1)
        graph = remove_path(graph, path_1)
    if path_2 != []:
        paths.append(path_2)
        graph = remove_path(graph, path_2)

    if len(paths) < k: 
        try: 
            path = nx.shortest_path(graph, src, dst, weight='cost')
            paths.append(path)
        except:
            pass
    
    return paths


def traffic_engineering(k, original_graph:nx.Graph, SRLG_graph:nx.Graph, original_traffic:dict, SRLG_pair:dict): 
    fiber_net = nx.Graph()
    fiber_net.add_nodes_from(original_graph)

    SRLG_fiber_net = nx.Graph()
    SRLG_fiber_net.add_nodes_from(SRLG_graph)

    load_graph = nx.Graph()
    for (node_i, node_j) in original_graph.edges:
        load_graph.add_edge(node_i, node_j, load=0)

    SRLG_load_graph = nx.Graph()
    for (node_i, node_j) in SRLG_graph.edges:
        SRLG_load_graph.add_edge(node_i, node_j, load=0)

    traffic_paths = {}

    traffic = []
    for key, value in original_traffic.items(): 
        traffic.append([key, value])
    traffic_orders = list(range(len(traffic)))
    traffic_orders.sort(key=lambda index: traffic[index][1], reverse=True)

    for index in traffic_orders:
        (src, dst), demand = traffic[index]

        graph = nx.Graph()
        for (node_i, node_j) in SRLG_load_graph.edges:
            # load capacity constrain
            if SRLG_load_graph[node_i][node_j]['load']+demand > SRLG_graph[node_i][node_j]['cap']: 
                graph.add_edge(node_i, node_j, cost=np.inf)
            else: 
                graph.add_edge(node_i, node_j, cost=SRLG_load_graph[node_i][node_j]['load'])

        paths = allocate_traffic(k, graph, src, dst)
        for i in range(len(paths)): 
            SRLG_fiber_net, SRLG_load_graph = update_graph(SRLG_fiber_net, SRLG_load_graph, paths[i], demand/len(paths))
        
            for node in paths[i]:
                if node in SRLG_pair.keys():
                    paths[i][paths[i].index(node)] = SRLG_pair[node]

            paths[i] = remove_cycle(paths[i])
            fiber_net, load_graph = update_graph(fiber_net, load_graph, paths[i], demand/len(paths))

        traffic_paths[(src, dst)] = paths
    
    return fiber_net, load_graph, traffic_paths

    