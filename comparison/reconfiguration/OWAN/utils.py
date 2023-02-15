import numpy as np
import networkx as nx


def find_rings(graph :nx.Graph, path :list, ring_len, core_num, dis_num, agg_num):
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

            if (neighbor >= core_num) and (neighbor < (core_num+dis_num)):
                if path[0] < path[-1]:
                    ring_num += 1
                    for i in range(1, len(path) -1):
                        if path[i] >= (core_num+dis_num) and visited[path[i]] == False:
                            visited[path[i]] = True

                    if (visited[(core_num+dis_num):] == True).all():
                        return
            find_rings(graph, path, ring_len, core_num, dis_num, agg_num)

    is_visited[path[-1]] = False
    path.pop()

# dual-homing check
def check_dual(graph :nx.Graph, ring_len=10):
    node_num = len(graph.nodes)
    node_type = [graph.nodes[i]['type'] for i in range(len(graph.nodes))]
    core_num = node_type.count('core')
    dis_num = node_type.count('dis')
    agg_num = node_type.count('agg')

    global is_visited, visited, ring_num
    is_visited = [False for _ in range(node_num)]
    visited = np.zeros(node_num)
    ring_num = 0

    for node in range(core_num, core_num+dis_num):
        is_visited = [False for _ in range(node_num)]
        find_rings(graph, [node], ring_len, core_num, dis_num, agg_num)
    dual_num = sum(visited[(core_num+dis_num):]==True)
    
    return int(dual_num), agg_num