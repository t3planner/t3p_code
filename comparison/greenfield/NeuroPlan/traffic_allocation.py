import copy
# import gurobipy

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


# get cost of all traffic demands
def calculate_costs(graph:nx.Graph, traffic:dict, allocated:list):
    node_num = len(graph)
    cost_dict = {}
    for node in range(node_num):
        node_parent, node_cost = dijkstra(graph, node)
        for src in node_cost.keys():
            if src != node:
                OD_pair = (src, node)
                if OD_pair in traffic and OD_pair not in allocated:
                    if node_cost[src] is None:
                        cost_dict[OD_pair] = np.inf
                    else:
                        cost_dict[OD_pair] = node_cost[src]
    return cost_dict


def load_constrain(graph:nx.Graph, demand):
    for (node_i, node_j) in graph.edges:
        if graph[node_i][node_j]['load']+demand > graph[node_i][node_j]['cap']:
            graph[node_i][node_j]['cost'] = np.inf
    return graph


def update_graph(fiber_net:nx.Graph, load_graph:nx.Graph, path, demand):
    for i in range(len(path)-1):
        node_i = path[i]
        node_j = path[i+1]
        fiber_net.add_edge(node_i, node_j)
        load_graph[node_i][node_j]['load'] += demand
    
    return fiber_net, load_graph


def heuristic_solver(SRLG_graph:nx.Graph, SRLG_pair:dict, traffic:dict, k=1):
    SRLG_fiber_net = nx.Graph()
    SRLG_fiber_net.add_nodes_from(SRLG_graph)

    SRLG_load_graph = nx.Graph()
    for (node_i, node_j) in SRLG_graph.edges:
        SRLG_load_graph.add_edge(node_i, node_j, load=0)

    allocated = []
    traffic_paths = {}
    while len(allocated) < len(traffic):
        graph = nx.Graph()
        for (node_i, node_j) in SRLG_graph.edges:
            if (node_i, node_j) in SRLG_fiber_net.edges: 
                graph.add_edge(
                    node_i, node_j, 
                    cost=0, 
                    load=SRLG_load_graph[node_i][node_j]['load'], 
                    cap=SRLG_graph[node_i][node_j]['cap']
                )
            else: 
                graph.add_edge(
                    node_i, node_j, 
                    cost=SRLG_graph[node_i][node_j]['cost'], 
                    load=SRLG_load_graph[node_i][node_j]['load'], 
                    cap=SRLG_graph[node_i][node_j]['cap']
                )

        cost_dict = calculate_costs(graph, traffic, allocated)
        costs = sorted(cost_dict.items(), key=lambda kv:(kv[1], kv[0]), reverse=False)

        # allocate k traffic demands before update of graph
        for i in range(k): 
            flow = costs[i]
            src = flow[0][0]
            dst = flow[0][1]
            demand = traffic[(src, dst)]

            allocated.append((src, dst))
            graph = load_constrain(graph, demand)
            path_1, path_2 = suurballe(graph, src, dst)

            if len(path_1) < len(path_2):
                SRLG_fiber_net, SRLG_load_graph = update_graph(SRLG_fiber_net, SRLG_load_graph, path_1, demand)
                SRLG_fiber_net, SRLG_load_graph = update_graph(SRLG_fiber_net, SRLG_load_graph, path_2, 0)
            else:
                SRLG_fiber_net, SRLG_load_graph = update_graph(SRLG_fiber_net, SRLG_load_graph, path_1, 0)
                SRLG_fiber_net, SRLG_load_graph = update_graph(SRLG_fiber_net, SRLG_load_graph, path_2, demand)
            
            for node in path_1:
                if node in SRLG_pair.keys():
                    path_1[path_1.index(node)] = SRLG_pair[node]
            path_1 = remove_cycle(path_1)
            for node in path_2:
                if node in SRLG_pair.keys():
                    path_2[path_2.index(node)] = SRLG_pair[node]
            path_2 = remove_cycle(path_2)
            traffic_paths[(src, dst)] = [path_1, path_2]

            for (node_i, node_j) in SRLG_load_graph.edges:
                if SRLG_load_graph[node_i][node_j]['load'] > SRLG_graph[node_i][node_j]['cap']: 
                    return None

    return traffic_paths


def SRLG_constrain(G:nx.Graph, graph:nx.Graph, SRLG:list): 
    SRLG_pair = {}
    for node, conflicts in SRLG: 
        if len(conflicts) == 2:
            node_i = conflicts.pop()
            node_j = conflicts.pop()

            if (node, node_i) in G.edges and (node, node_j) in G.edges: 
                node_s = max(graph.nodes)+1
                SRLG_pair[node_s] = node
                graph.add_edge(
                    node, node_s, 
                    cost=0, 
                    load=0, 
                    cap=min(graph[node][node_i]['cap'], graph[node][node_j]['cap'])
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

        elif len(conflicts) == 3:
            node_i = conflicts.pop()
            node_j = conflicts.pop()
            node_k = conflicts.pop()

            if (node, node_i) in G.edges and (node, node_j) in G.edges and (node, node_k) in G.edges: 
                node_s = max(graph.nodes)+1
                SRLG_pair[node_s] = node
                graph.add_edge(
                    node, node_s, 
                    cost=0, 
                    load=0, 
                    cap=min(graph[node][node_i]['cap'], graph[node][node_j]['cap'])
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

                if node_k not in graph.adj[node]:
                    for neighbor in graph.adj[node_k]:
                        if neighbor in SRLG_pair.keys() and node in graph.adj[neighbor]: 
                            node_k = neighbor
                graph.add_edge(
                    node_s, node_k, 
                    cost=graph[node][node_k]['cost'], 
                    load=graph[node][node_k]['load'], 
                    cap=graph[node][node_k]['cap']
                )
                graph.remove_edge(node, node_k)
    
    return graph, SRLG_pair


def allocate_traffic(
    original_graph:nx.Graph, original_traffic:dict, original_SRLG:list, time_limit=None
): 
    traffic = copy.deepcopy(original_traffic)
    SRLG = copy.deepcopy(original_SRLG)

    SRLG_graph = nx.Graph()
    for (node_i, node_j) in original_graph.edges:
        SRLG_graph.add_edge(
            node_i, node_j, 
            cost=original_graph[node_i][node_j]['cost'], 
            load=0, 
            cap=original_graph[node_i][node_j]['load']
        )
    
    SRLG_graph, SRLG_pair = SRLG_constrain(original_graph, SRLG_graph, SRLG)

    traffic_paths = heuristic_solver(SRLG_graph, SRLG_pair, traffic)

    return traffic_paths
