import networkx as nx
import numpy as np

def find_rings(graph :nx.Graph, path :list, ring_len, Core_num, Agg_num, Acc_num, node_list=None):
    '''
        find the dual-homing ring in the graph
        the ring: the starting node is AGG, the end node is also Agg, and the middle node is Acc
    '''
    global is_visited, visited, ring_num
    for node in path:
        is_visited[node] = True

    src = path[-1]
    for neighbor in graph.adj[src]:
        if is_visited[neighbor] is False:
            path.append(neighbor)
            is_visited[neighbor] = True

            if len(path) > ring_len:  # the length of the ring
                path.pop()
                is_visited[neighbor] = False
                break

            if (neighbor >= Core_num) and (neighbor < (Core_num+Agg_num)):
                if path[0] < path[-1]:
                    ring_num += 1
                    for i in range(1, len(path) - 1):
                        if path[i] >= (Core_num+Agg_num) and visited[path[i]] == False:
                            visited[path[i]] = True
                    if node_list == None:
                        if (visited[(Core_num + Agg_num):] == True).all():
                            return
                    else:
                        ring_flag = 0
                        for k in node_list:
                            if visited[k] == False:
                                ring_flag = 1
                        if ring_flag == 0:
                            return

            find_rings(graph, path, ring_len, Core_num, Agg_num, Acc_num, node_list)

    is_visited[path[-1]] = False
    path.pop()


def rings_flag(graph: nx.Graph, ring_len=5, node_list=None):
    '''
    determine whether the topology meets the dual-homing ring constraints
    '''
    node_num = len(graph.nodes)
    node_type = [graph.nodes()[i]['type'] for i in graph.nodes]
    Core_num = node_type.count('Core')
    Agg_num = node_type.count('Agg')
    Acc_num = node_type.count('Acc')

    global is_visited, visited, ring_num
    is_visited = [False for _ in range(node_num)]
    visited = np.zeros(node_num)
    ring_num = 0

    for node in range(Core_num, Core_num+Agg_num):
        is_visited = [False for _ in range(node_num)]
        find_rings(graph, [node], ring_len, Core_num, Agg_num, Acc_num, node_list)

    if node_list == None:
        dual_num = sum(visited[(Core_num + Agg_num):] == True)
        if dual_num == Acc_num:
            return 0
        else:
            return 1
    else:
        ring_flag = 0
        for k in node_list:
            if visited[k] == False:
                ring_flag = 1
        return ring_flag
