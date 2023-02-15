import copy
import random
import gurobipy

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
        if (node_i, node_j) in graph.edges:
            graph.remove_edge(node_i, node_j)
    return graph


def oblivious_routing(k, SRLG_graph:nx.Graph, traffic:dict):
    traffic_paths = {}
    for (src, dst) in traffic.keys(): 
        # try to find k disjoint paths for a traffic demand
        traffic_paths[(src, dst)] = []
        graph = copy.deepcopy(SRLG_graph)
        path_1, path_2 = suurballe(graph, src, dst)
        if path_1 != []:
            traffic_paths[(src, dst)].append(path_1)
            graph = remove_path(graph, path_1)
        if path_2 != []:
            traffic_paths[(src, dst)].append(path_2)
            graph = remove_path(graph, path_2)

        if len(traffic_paths[(src, dst)]) < k: 
            try: 
                path = nx.shortest_path(graph, src, dst, weight='cost')
                traffic_paths[(src, dst)].append(path)
            except:
                pass

    return traffic_paths


def check_edge(node_i, node_j, path):
    for i in range(len(path)-1):
        if node_i == path[i] and node_j == path[i+1]:
            return True
        elif node_j == path[i] and node_i == path[i+1]:
            return True

    return False


def gurobi_solver(k, SRLG_graph:nx.Graph, traffic_demands:dict, traffic_paths:dict, time_limit=None, verbose=True): 
    traffic_list = list(traffic_paths.keys())
    traffic_num = len(traffic_list)

    demand_list = list(traffic_demands.values())
    path_list = list(traffic_paths.values())

    edge_list = list(SRLG_graph.edges)
    edge_num = len(edge_list)

    edge_flow = np.zeros((edge_num, traffic_num, k))
    for edge_idx in range(edge_num): 
        (node_i, node_j) = edge_list[edge_idx]
        for i in range(traffic_num): 
            paths = path_list[i]
            for j in range(len(paths)): 
                path = paths[j]
                if check_edge(node_i, node_j, path): 
                    edge_flow[edge_idx][i][j] = 1
    
    model = gurobipy.Model()
    model.modelSense = gurobipy.GRB.MINIMIZE
    model.Params.OutputFlag = verbose
    model.Params.Method = 2
    model.Params.TimeLimit = time_limit

    max_util = model.addVar(lb=0, vtype=gurobipy.GRB.CONTINUOUS, name='max_util')
    weights = model.addVars(traffic_num, k, lb=0, vtype=gurobipy.GRB.CONTINUOUS, name='weights')
    model.update()

    model.setObjective(max_util)
    
    for i in range(traffic_num):
        paths = path_list[i]
        model.addConstr(sum(weights[i, j] for j in range(len(paths))) == 1)
    for edge_idx in range(edge_num):
        model.addConstr(sum((demand_list[i]*weights[i, j]*edge_flow[edge_idx][i][j]) for i in range(traffic_num) for j in range(k)) <= SRLG_graph[node_i][node_j]['cap'])
        model.addConstr(sum((demand_list[i]*weights[i, j]*edge_flow[edge_idx][i][j]) for i in range(traffic_num) for j in range(k)) <= max_util)
    
    model.optimize()
    if model.status == gurobipy.GRB.Status.OPTIMAL or model.status == gurobipy.GRB.Status.TIME_LIMIT:
        path_weights = {}

        weights = model.getAttr('x', weights)
        for i in range(traffic_num): 
            (src, dst) = traffic_list[i]
            for j in range(k): 
                path_weights[(src, dst, j)] = weights[i, j]

        return path_weights

    else:
        print('Solver exit with status: ', model.status)
        return None


def traffic_engineering(k, original_graph:nx.Graph, SRLG_graph:nx.Graph, original_traffic:dict, SRLG_pair:dict, time_limit=None): 
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

    traffic_paths = oblivious_routing(k, SRLG_graph, original_traffic)
    path_weights = gurobi_solver(k, SRLG_graph, original_traffic, traffic_paths, time_limit)
    for (src, dst) in traffic_paths.keys():
        demand = original_traffic[(src, dst)]
        paths = traffic_paths[(src, dst)]
        for j in range(len(paths)): 
            for node in paths[j]:
                if node in SRLG_pair.keys():
                    paths[j][paths[j].index(node)] = SRLG_pair[node]

            paths[j] = remove_cycle(paths[j])
            weight = path_weights[(src, dst, j)]
            fiber_net, load_graph = update_graph(fiber_net, load_graph, paths[j], demand*weight)
        
        for j in range(len(paths)): 
            weight = path_weights[(src, dst, j)]
            paths.append(weight)

    return fiber_net, load_graph, traffic_paths