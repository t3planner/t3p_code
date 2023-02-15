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
    priority_queue = [(0, src, None)]  # (cost, node, parent)
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
                    hp.heappush(priority_queue, (cost + graph[node][neighbor]['cost'], neighbor, node))

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


def cap_constrain(G:nx.Graph, demand):
    graph = copy.deepcopy(G)
    for (node_i, node_j) in graph.edges:
        if graph[node_i][node_j]['load']+demand > graph[node_i][node_j]['cap']:
             graph[node_i][node_j]['cost'] = np.inf
    return graph


def SRLG_constrain(graph: nx.Graph, SRLG:list): 
    SRLG_pair = {}
    for node, conflicts in SRLG: 
        node_i = conflicts.pop()
        node_j = conflicts.pop()

        if (node, node_i) in graph.edges and (node, node_j) in graph.edges: 
            node_s = max(graph.nodes)+1
            SRLG_pair[node_s] = node
            graph.add_edge(
                node, node_s, 
                cost=0, 
                load=0, 
                cap=max(graph[node][node_i]['cap'], graph[node][node_j]['cap'])
            )

            if node_i not in graph.adj[node]:
                for neighbor in graph.adj[node_i]:
                    if neighbor in SRLG_pair.keys() and node in graph.adj[neighbor]: 
                        node_i = neighbor
            graph.add_edge(
                node_s, node_i, 
                cost=graph[node][node_i]['cost'], 
                load=graph[node][node_i]['load'], 
                cap=graph[node][node_i]['cap']
            )
            graph.remove_edge(node, node_i)

            if node_j not in graph.adj[node]:
                for neighbor in graph.adj[node_j]:
                    if neighbor in SRLG_pair.keys() and node in graph.adj[neighbor]: 
                        node_j = neighbor
            graph.add_edge(
                node_s, node_j, 
                cost=graph[node][node_j]['cost'], 
                load=graph[node][node_j]['load'], 
                cap=graph[node][node_j]['cap']
            )
            graph.remove_edge(node, node_j)
    
    return graph, SRLG_pair


def compute_all_distances(new_graph, traffic):
    nb_nodes = len(new_graph)
    all_distances = {}
    all_parent_list = []
    for initial_node in range(nb_nodes):
        parent_list, distances = dijkstra(new_graph, initial_node)
        for i in range(len(distances)):
            if i!=initial_node:
                a=(i, initial_node)
                if (a in traffic):
                    if distances[i]==None:
                        all_distances[(i, initial_node)] = np.inf
                    else:
                        all_distances[(i, initial_node)] = distances[i]
        all_parent_list.append(parent_list)

    return all_distances


def allocate_traffic(graph:nx.Graph, traffic, SRLG_pair:dict={}):
    unallocated = 0
    traffic_paths = {}
    load_graph = copy.deepcopy(graph)
    fiber_graph = copy.deepcopy(graph)
    topo = nx.create_empty_copy(graph)  # final topology
    all_distances = compute_all_distances(fiber_graph, traffic)
    all_distances = sorted(all_distances.items(), key=lambda kv: (kv[1], kv[0]))
    for f in all_distances:
        src = f[0][0]
        dst = f[0][1]
        demand = traffic[(src, dst)]
        fiber_graph_new = cap_constrain(fiber_graph, demand)
        path_1, path_2 = suurballe(fiber_graph_new, src, dst)
        path_1 = remove_cycle(path_1)
        path_2 = remove_cycle(path_2)

        if path_1 != [] and path_2 !=[]:
            for i in range(1, len(path_1)):
                fiber_graph[path_1[i - 1]][path_1[i]]['cost'] += 10
            for i in range(1, len(path_2)):
                fiber_graph[path_2[i - 1]][path_2[i]]['cost'] += 10
        for node in path_1:
            if node in SRLG_pair.keys():
                path_1[path_1.index(node)] = SRLG_pair[node]
        for node in path_2:
            if node in SRLG_pair.keys():
                path_2[path_2.index(node)] = SRLG_pair[node]
        path_1 = remove_cycle(path_1)
        path_2 = remove_cycle(path_2)
        if path_1 == [] or path_2 == []:
            unallocated += 1
        else:
            traffic_paths[(src, dst)] = [path_1, path_2]
            for i in range(1, len(path_1)):
                topo.add_edge(path_1[i - 1], path_1[i])

            for i in range(1, len(path_2)):
                topo.add_edge(path_2[i - 1], path_2[i])
            if len(path_1) < len(path_2):
                for i in range(1, len(path_1)):
                    load_graph[path_1[i - 1]][path_1[i]]['load'] += traffic[(src, dst)]
            else:
                for i in range(1, len(path_2)):
                    load_graph[path_2[i - 1]][path_2[i]]['load'] += traffic[(src, dst)]
    max_util = 0
    for (node_i, node_j) in topo.edges:
        util = load_graph[node_i][node_j]['load']
        if util > max_util: 
            max_util = util

    return load_graph, max_util, unallocated, traffic_paths