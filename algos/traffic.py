import heapq as hp
import networkx as nx
import numpy as np
import copy

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
    '''
            Find the shortest path from src to dst in the graph
    '''
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


def suurballe(G: nx.Graph, src, dst):
    '''
        Find the shortest two non-consecting paths from src to dst in the graph
    '''
    graph = G.to_directed()
    node_parent, node_cost = dijkstra(graph, src)
    node = dst
    path_1 = [dst]
    while node != src:
        node = node_parent[node]
        if node == None:
            path_1 = []
            break
        path_1.append(node)
    path_1.reverse()

    for node_i in graph.nodes:
        for node_j in graph.adj[node_i]:
            if node_cost[node_i] == None or node_cost[node_j] == None:
                graph[node_i][node_j]['cost'] = np.inf
            else:
                graph[node_i][node_j]['cost'] += node_cost[node_i] - node_cost[node_j]

    for i in range(len(path_1) - 1):
        graph.remove_edge(path_1[i], path_1[i + 1])
        graph[path_1[i + 1]][path_1[i]]['cost'] = 0

    path_2 = dijkstra(graph, src, dst)
    for i in range(1, len(path_2) - 1):
        if path_2[i] in path_1:
            j = path_1.index(path_2[i])
            if path_1[j - 1] == path_2[i + 1]:
                p_1 = path_1[:j - 1] + path_2[i + 1:]
                p_2 = path_2[:i] + path_1[j:]
                path_1 = p_1
                path_2 = p_2
                break

    path_1 = remove_cycle(path_1)
    path_2 = remove_cycle(path_2)

    return path_1, path_2

def demand_constrain(graph: nx.Graph, demand:int):
    '''
        update the graph according to the bandwidth
    '''
    graph_new = copy.deepcopy(graph)
    for edge in graph_new.edges:
        if graph_new[edge[0]][edge[1]]['load'] + demand > graph_new[edge[0]][edge[1]]['cap']:
            graph_new[edge[0]][edge[1]]['cost'] = np.inf
    return graph_new


def SRLG_constrain(G: nx.Graph, SRLG: list):
    '''
        update the graph according to the SRLG
    '''
    graph = copy.deepcopy(G)
    SRLG_pair = {}
    for node, conflicts in SRLG:
        if len(conflicts) == 2:
            node_i = conflicts.pop()
            node_j = conflicts.pop()

            if (node, node_i) in G.edges and (node, node_j) in G.edges:
                node_s = max(graph.nodes) + 1
                SRLG_pair[node_s] = node
                graph.add_edge(
                    node, node_s,
                    cost=0,
                    load=0,
                    cap=min(G[node][node_i]['cap'], G[node][node_j]['cap'])
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
                node_s = max(graph.nodes) + 1
                SRLG_pair[node_s] = node
                graph.add_edge(
                    node, node_s,
                    cost=0,
                    load=0,
                    cap=min(G[node][node_i]['cap'], G[node][node_j]['cap'])
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

def compute_all_distances(new_graph, traffic):
    '''
        calculate the distance of the shortest path of all flow
    '''
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

def update_traffic(graph: nx.Graph, original_traffic, SRLG):
    '''
        filter unreachable traffic
    '''
    fiber_graph = copy.deepcopy(graph)
    fiber_graph, SRLG_pair = SRLG_constrain(fiber_graph, SRLG)
    traffic = copy.deepcopy(original_traffic)
    for f in original_traffic.keys():
        src = f[0]
        dst = f[1]
        path_1, path_2 = suurballe(fiber_graph, src, dst)
        path_1 = remove_cycle(path_1)
        path_2 = remove_cycle(path_2)

        if path_1 != [] and path_2 != []:
            for i in range(1, len(path_1)):
                fiber_graph[path_1[i - 1]][path_1[i]]['cost'] = 0
            for i in range(1, len(path_2)):
                fiber_graph[path_2[i - 1]][path_2[i]]['cost'] = 0
        for node in path_1:
            if node in SRLG_pair.keys():
                path_1[path_1.index(node)] = SRLG_pair[node]
        for node in path_2:
            if node in SRLG_pair.keys():
                path_2[path_2.index(node)] = SRLG_pair[node]
        path_1 = remove_cycle(path_1)
        path_2 = remove_cycle(path_2)
        if path_1 == [] or path_2 == []:
            print("No routing for ({}, {})".format(f[0], f[1]))
            traffic.pop(f)

    return traffic

