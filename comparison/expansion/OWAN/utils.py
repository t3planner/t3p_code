import numpy as np
import networkx as nx


def find_rings(graph:nx.Graph, path:list, node_list, ring_len, Core_num, Agg_num, Acc_num):
    global is_visited, visited, ring_num
    for node in path:
        is_visited[node] = True

    src = path[-1]
    for neighbor in graph.adj[src]:
        if is_visited[neighbor] is False:
            path.append(neighbor)
            is_visited[neighbor] = True

            if len(path) > ring_len:
                path.pop()
                is_visited[neighbor] = False
                break

            if (neighbor >= Core_num) and (neighbor < (Core_num+Agg_num)):
                if path[0] < path[-1]:
                    ring_num += 1
                    for i in range(1, len(path) -1):
                        if path[i] >= (Core_num+Agg_num) and visited[path[i]] == False:
                            visited[path[i]] = True
                    ring_flag = 0
                    for k in node_list:
                        if visited[k] == False:
                            ring_flag = 1
                    if ring_flag == 0:
                        return
            find_rings(graph, path, node_list, ring_len, Core_num, Agg_num, Acc_num)

    is_visited[path[-1]] = False
    path.pop()

# dual-homing check
def check_dual(graph:nx.Graph, node_list, ring_len=10):
    node_num = len(graph.nodes)
    node_type = [graph.nodes()[i]['type'] for i in range(len(graph.nodes))]
    Core_num = node_type.count('Core site')
    Agg_num = node_type.count('Aggregation site')
    Acc_num = node_type.count('Access site')
    global is_visited, visited, ring_num
    is_visited = [False for _ in range(node_num)]
    visited = np.zeros(node_num)
    ring_num = 0

    for node in range(Core_num, Core_num+Agg_num):
        is_visited = [False for _ in range(node_num)]
        find_rings(graph, [node], node_list, ring_len, Core_num, Agg_num, Acc_num)
    
    ring_flag = True
    for node in node_list:
        if visited[node] == False:
            ring_flag = False

    return ring_flag
