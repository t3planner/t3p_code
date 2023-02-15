import networkx as nx
import numpy as np

def find_rings(graph :nx.Graph, path :list, ring_len, node_list):
    global is_visited, visited, ring_num
    for node in path:
        is_visited[node] = True

    src = path[-1]
    for neighbor in graph.adj[src]:
        if neighbor not in node_list:
            continue
        if neighbor == path[0]:
            path.append(neighbor)
            if len(path) == 3:
                path.pop()
                continue
            elif len(path) <= ring_len:
                ring_num += 1
                for i in range(len(path)):
                    if visited[path[i]] == False:
                        visited[path[i]] = True
                path.pop()
                break
                if sum(visited == True) == len(graph.nodes):
                    return
        elif is_visited[neighbor] is False:
            path.append(neighbor)
            if len(path) >= ring_len:
                path.pop()
                is_visited[neighbor] = False
                continue
            find_rings(graph, path, ring_len, node_list)
    is_visited[path[-1]] = False
    path.pop()


def rings_flag(graph :nx.Graph, ring_len=5, node_list=None):
    node_num = len(graph.nodes)
    global is_visited, visited, ring_num
    is_visited = [False for _ in range(max(graph.nodes)+1)]
    visited = np.zeros(max(graph.nodes)+1)
    ring_num = 0

    for node in node_list:
        is_visited = [False for _ in range(max(graph.nodes)+1)]
        find_rings(graph, [node], ring_len, node_list)
    dual_num = sum(visited == True)
    if dual_num != len(node_list):
        return 1
    for node in graph.nodes:
        if visited[node] == True:
            continue
        if node not in node_list:
            for n in graph.adj[node]:
                if n in node_list:
                    visited[node] = True
    dual_num = sum(visited==True)
    if dual_num == node_num:
        return 0
    else:
        return 1