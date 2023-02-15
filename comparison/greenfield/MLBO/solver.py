import copy
import gurobipy

import heapq as hp
import numpy as np
import networkx as nx

from MLBO.utils import check_dual


# convert pairs of nodes into list of nodes
def convert_path(path:list, src, dst): 
    node = src
    converted_path = [node]
    while node != dst:
        for (node_i, node_j) in path: 
            if node == node_i: 
                converted_path.append(node_j)
                node = node_j
                path.remove((node_i, node_j))
                break
    return converted_path


def gurobi_solver(SRLG_graph:nx.Graph, traffic:dict, time_limit=None, verbose=True): 
    edge_list = []
    for (node_i, node_j) in SRLG_graph.edges: 
        edge_list.append((node_i, node_j))
        edge_list.append((node_j, node_i))

    commodity_list = list(traffic.keys())
    traffic_num = len(commodity_list)
    
    model = gurobipy.Model('netflow')
    model.modelSense = gurobipy.GRB.MINIMIZE
    model.Params.OutputFlag = verbose
    model.Params.TimeLimit = time_limit

    phy_link = model.addVars(edge_list, vtype=gurobipy.GRB.BINARY, name='phy_link')
    flow_1 = model.addVars(traffic_num, edge_list, vtype=gurobipy.GRB.BINARY, name='flow_1')
    flow_2 = model.addVars(traffic_num, edge_list, vtype=gurobipy.GRB.BINARY, name='flow_2')
    model.update()

    model.setObjective(gurobipy.quicksum((phy_link[node_i, node_j]*SRLG_graph[node_i][node_j]['cost']) for (node_i, node_j) in edge_list))
    
    model.addConstrs((flow_1[i, node_i, node_j]+flow_2[i, node_i, node_j]+flow_1[i, node_j, node_i]+flow_2[i, node_j, node_i]) <= phy_link[node_i, node_j] for i in range(traffic_num) for (node_i, node_j) in edge_list)
    for i, commodity in enumerate(commodity_list):
        src, dst = commodity
        for node in SRLG_graph.nodes:
            if node == dst:
                rhs = -1
            elif node == src:
                rhs = 1
            else:
                rhs = 0
            model.addConstr((flow_1.sum(i, node, '*') - flow_1.sum(i, '*', node) == rhs), '{}_{}'.format(node, src))
            model.addConstr((flow_2.sum(i, node, '*') - flow_2.sum(i, '*', node) == rhs), '{}_{}'.format(node, src))
    
    model.optimize()
    if model.status == gurobipy.GRB.Status.OPTIMAL or model.status == gurobipy.GRB.Status.TIME_LIMIT:
        traffic_paths = {}

        paths_1 = model.getAttr('x', flow_1)
        paths_2 = model.getAttr('x', flow_2)
        links = model.getAttr('x', phy_link)

        for i in range(traffic_num): 
            (src, dst) = commodity_list[i]
            path_1 = [(node_i, node_j) for (node_i, node_j) in edge_list if paths_1[i, node_i, node_j] > 0]
            path_2 = [(node_i, node_j) for (node_i, node_j) in edge_list if paths_2[i, node_i, node_j] > 0]
            path_1 = convert_path(path_1, src, dst)
            path_2 = convert_path(path_2, src, dst)
            traffic_paths[(src, dst)] = [path_1, path_2]

        return traffic_paths

    else:
        print('Solver exit with status: ', model.status)


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
    

def heuristic_solver(SRLG_graph:nx.Graph, traffic:dict):
    allocate_num = 0
    traffic_paths = {}
    for (src, dst) in traffic.keys():
        # traffic allocation, only with both main path and backup path will be accepted
        try: 
            path_1, path_2 = suurballe(SRLG_graph, src, dst)
            if len(path_1) > 0 and len(path_2) > 0:
                allocate_num += 1
                traffic_paths[(src, dst)] = [path_1, path_2]
        except: 
            pass
    
    return allocate_num, traffic_paths


# def evaluate_edges(G:nx.Graph, traffic:dict): 
#     edge_paths = {}
#     edge_rewards = {}
#     for (node_i, node_j) in G.edges:
#         graph = copy.deepcopy(G)
#         graph.remove_edge(node_i, node_j)

#         traffic_paths = heuristic_solver(graph, traffic)
#         # traffic_paths = gurobi_solver(graph, traffic, time_limit)
#         if traffic_paths is not None:
#             edge_paths[(node_i, node_j)] = traffic_paths
#             if check_dual(graph):
#                 edge_rewards[(node_i, node_j)] = G[node_i][node_j]['cost']
#             else: 
#                 edge_rewards[(node_i, node_j)] = -np.inf
#         else:
#             edge_rewards[(node_i, node_j)] = -np.inf
            
#     if max(edge_rewards.values()) < 0:
#         edge_paths = {}

#     return edge_paths, edge_rewards


def evaluate_edges(original_graph:nx.Graph, G:nx.Graph, traffic:dict): 
    edge_paths = {}
    edge_costs = {}
    traffic_num = len(traffic)
    max_cost = 0
    # find max cost of edges, will be used for normalization
    for (node_i, node_j) in original_graph.edges:
        cost = original_graph[node_i][node_j]['cost']
        if cost > max_cost: 
            max_cost = cost

    for (node_i, node_j) in original_graph.edges:
        if (node_i, node_j) not in G.edges: 
            graph = copy.deepcopy(G)
            graph.add_edge(node_i, node_j, cost=original_graph[node_i][node_j]['cost'])

            allocate_num, traffic_paths = heuristic_solver(graph, traffic)
            # traffic_paths = gurobi_solver(graph, traffic, time_limit)
            
            edge_paths[(node_i, node_j)] = traffic_paths
            edge_costs[(node_i, node_j)] = graph[node_i][node_j]['cost']/max_cost-allocate_num/traffic_num      # normalization of cost and allocate_num
            
    return edge_paths, edge_costs


def update_graph(fiber_net:nx.Graph, load_graph:nx.Graph, path, demand):
    for i in range(len(path)-1):
        node_i = path[i]
        node_j = path[i+1]
        fiber_net.add_edge(node_i, node_j)
        load_graph[node_i][node_j]['load'] += demand
    
    return fiber_net, load_graph


def greedy_algorithm(original_graph:nx.Graph, SRLG_graph:nx.Graph, original_traffic:dict, SRLG_pair:dict={}, ring_len=10):
    fiber_net = nx.Graph()
    fiber_net.add_nodes_from(original_graph)

    load_graph = nx.Graph()
    for (node_i, node_j) in original_graph.edges:
        load_graph.add_edge(node_i, node_j, load=0)

    graph = nx.Graph()
    for node in SRLG_graph.nodes: 
        graph.add_node(node, type=SRLG_graph.nodes[node]['type'])

    # done = False
    # while not done:
    #     edge_paths, edge_rewards = evaluate_edges(SRLG_graph, original_traffic)
    #     if len(edge_paths) > 0: 
    #         edges = list(edge_rewards.keys())
    #         rewards = list(edge_rewards.values())

    #         max_idx = rewards.index(max(rewards))
    #         (node_i, node_j) = edges[max_idx]
    #         SRLG_traffic_paths = edge_paths[(node_i, node_j)]

    #         SRLG_graph.remove_edge(node_i, node_j)
    #     else: 
    #         done = True

    done = False
    while not done: 
        edge_paths, edge_costs = evaluate_edges(SRLG_graph, graph, original_traffic)    # get cost of all edges

        edges = list(edge_costs.keys())
        costs = list(edge_costs.values())

        # add the edge with min cost to current topology
        min_idx = costs.index(min(costs))
        (node_i, node_j) = edges[min_idx]
        print((node_i, node_j))
        graph.add_edge(node_i, node_j, cost=SRLG_graph[node_i][node_j]['cost'])

        traffic_num = len(original_traffic)
        SRLG_traffic_paths = edge_paths[(node_i, node_j)]
        dual_num, agg_num = check_dual(graph, ring_len)

        # terminal condition
        if len(SRLG_traffic_paths)>traffic_num-1 and dual_num>agg_num-1: 
            done = True

    traffic_paths = {}
    for (src, dst), paths in SRLG_traffic_paths.items(): 
        demand = original_traffic[(src, dst)]

        for j in range(len(paths)): 
            for node in paths[j]:
                if node in SRLG_pair.keys():
                    paths[j][paths[j].index(node)] = SRLG_pair[node]

            paths[j] = remove_cycle(paths[j])
            load_graph = update_graph(fiber_net, load_graph, paths[j], demand)

        traffic_paths[(src, dst)] = paths

    return fiber_net, load_graph, traffic_paths