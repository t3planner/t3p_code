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

def short_path(G:nx.Graph, src, dst, node_list):
    graph = G.to_directed()

    node_parent, node_cost = dijkstra(graph, dst)

    path_1 = [src]
    cost = np.inf
    neighbor = None
    for node in G.adj[src]:
        if node in node_list and G[node][src]['cost']+node_cost[node] < cost:
            cost = G[node][src]['cost']+node_cost[node]
            neighbor = node
    path_1.append(neighbor)
    node = neighbor
    if node == None:
        return [], []
    while node != dst:
        node = node_parent[node]
        if node == None:
            path_1 = []
            break
        path_1.append(node)
    path_1.reverse()
    return path_1, copy.deepcopy(path_1)

def suurballe(G: nx.Graph, src, dst):
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


def SRLG_constrain(graph: nx.Graph, SRLG_o: list):
    SRLG_pair = {}
    SRLG = copy.deepcopy(SRLG_o)
    for node, conflicts in SRLG:
        node_i = conflicts.pop()
        node_j = conflicts.pop()

        if (node, node_i) in graph.edges and (node, node_j) in graph.edges:
            node_s = max(graph.nodes) + 1
            SRLG_pair[node_s] = node
            graph.add_edge(node, node_s, cost=0, load=0)

            if node_i not in graph.adj[node]:
                for neighbor in graph.adj[node_i]:
                    if neighbor in SRLG_pair.keys() and node in graph.adj[neighbor]:
                        node_i = neighbor
            graph.add_edge(node_s, node_i, cost=graph[node][node_i]['cost'], load=0)
            graph.remove_edge(node, node_i)

            if node_j not in graph.adj[node]:
                for neighbor in graph.adj[node_j]:
                    if neighbor in SRLG_pair.keys() and node in graph.adj[neighbor]:
                        node_j = neighbor
            graph.add_edge(node_s, node_j, cost=graph[node][node_j]['cost'], load=0)
            graph.remove_edge(node, node_j)

    return graph, SRLG_pair

def calaulate_traffic(graph: nx.Graph, traffic, SRLG, node_list):
    fiber_graph = copy.deepcopy(graph)
    fiber_graph, SRLG_pair = SRLG_constrain(fiber_graph, SRLG)
    topo = nx.create_empty_copy(graph)  # Record result topology
    flag = 0
    flow_path = {(f[0], f[1]): [] for f in traffic}
    for f in traffic:
        src = f[0]
        dst = f[1]

        if src not in node_list and dst in node_list:
            path_1, path_2 = short_path(fiber_graph, src, dst, node_list)
        else:
            path_1, path_2 = suurballe(fiber_graph, src, dst)

        path_1 = remove_cycle(path_1)
        path_2 = remove_cycle(path_2)

        if path_1 != [] and path_2 !=[]:
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
            flag = 1
            return None, 0, flag, None
        else:
            if src not in node_list and dst in node_list:
                flow_path[(src, dst)] = [path_1]
            else:
                flow_path[(src, dst)] = [path_1, path_2]
            for i in range(1, len(path_1)):
                topo.add_edge(path_1[i - 1], path_1[i])
            for i in range(1, len(path_2)):
                topo.add_edge(path_2[i - 1], path_2[i])
    cost = 0
    for (node_i, node_j) in topo.edges:
        cost += graph[node_i][node_j]['cost']
    return topo, cost, flag, flow_path

def calaulate_traffic_original(graph: nx.Graph, original_traffic, SRLG, node_list):
    fiber_graph = copy.deepcopy(graph)
    fiber_graph, SRLG_pair = SRLG_constrain(fiber_graph, SRLG)
    topo = nx.create_empty_copy(graph)  # Record result topology
    traffic = copy.deepcopy(original_traffic)
    flow_path = {(f[0], f[1]): [] for f in original_traffic}
    for f in original_traffic.keys():
        src = f[0]
        dst = f[1]

        if src not in node_list and dst in node_list:
            path_1, path_2 = short_path(fiber_graph, src, dst, node_list)
        else:
            path_1, path_2 = suurballe(fiber_graph, src, dst)


        path_1 = remove_cycle(path_1)
        path_2 = remove_cycle(path_2)

        if path_1 != [] and path_2 !=[]:
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
            traffic.pop(f)
        else:
            if src not in node_list and dst in node_list:
                flow_path[(src, dst)] = [path_1]
            else:
                flow_path[(src, dst)] = [path_1, path_2]
            for i in range(1, len(path_1)):
                topo.add_edge(path_1[i - 1], path_1[i])
            for i in range(1, len(path_2)):
                topo.add_edge(path_2[i - 1], path_2[i])
    cost = 0
    for (node_i, node_j) in topo.edges:
        cost += graph[node_i][node_j]['cost']

    return topo, cost, traffic, flow_path