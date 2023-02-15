import numpy as np
import networkx as nx

from scipy.signal import lfilter


def combine_shape(length, shape=None):
    if shape is None:
        return (length,)
    elif np.isscalar(shape): 
        return (length, shape)
    else:
        return (length, *shape)


def cumulative_sum(x, discount): 
    return lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def count_params(model): 
    return sum([np.prod(p.shape) for p in model.parameters()])


def statistics_scalar(x, with_min_max=False): 
    x = np.array(x, dtype=np.float32)
    sum, n = [np.sum(x), len(x)]
    mean = sum/n

    sum_sqrt = np.sum((x-mean)**2)
    std = np.sqrt(sum_sqrt/n)
    
    if with_min_max: 
        min = np.min(x) if len(x)>0 else np.inf
        max = np.max(x) if len(x)>0 else -np.inf
        return mean, std, min, max
    
    return mean, std


def check_edge(node_i, node_j, path):
    for i in range(len(path)-1):
        if node_i == path[i] and node_j == path[i+1]:
            return True
        elif node_j == path[i] and node_i == path[i+1]:
            return True

    return False


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