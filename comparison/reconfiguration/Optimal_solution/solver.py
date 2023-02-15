import gurobipy

import networkx as nx


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

    traffic_list = list(traffic.keys())
    traffic_num = len(traffic_list)

    demand_list = list(traffic.values())

    model = gurobipy.Model()
    model.modelSense = gurobipy.GRB.MINIMIZE
    model.Params.OutputFlag = verbose
    model.Params.Method = 2
    model.Params.TimeLimit = time_limit

    flow_1 = model.addVars(traffic_num, edge_list, vtype=gurobipy.GRB.BINARY, name='flow_1')
    flow_2 = model.addVars(traffic_num, edge_list, vtype=gurobipy.GRB.BINARY, name='flow_2')
    weights = model.addVars(traffic_num, 2, lb=0, vtype=gurobipy.GRB.CONTINUOUS, name='weights')
    
    loads = model.addVars(edge_list, vtype=gurobipy.GRB.CONTINUOUS, name='loads')
    max_util = model.addVar(lb=0, vtype=gurobipy.GRB.CONTINUOUS, name='max_util')
    
    model.update()

    model.setObjective(max_util)

    for (node_i, node_j) in edge_list:
        model.addConstr(loads[node_i, node_j] == loads[node_j, node_i])
        
        model.addConstr(sum([weights[i, 0]*demand_list[i]*(flow_1[i, node_i, node_j]+flow_1[i, node_j, node_i])+weights[i, 1]*demand_list[i]*(flow_2[i, node_i, node_j]+flow_2[i, node_j, node_i]) for i in range(traffic_num)]) <= max_util)
        model.addConstr(sum([weights[i, 0]*demand_list[i]*(flow_1[i, node_i, node_j]+flow_1[i, node_j, node_i])+weights[i, 1]*demand_list[i]*(flow_2[i, node_i, node_j]+flow_2[i, node_j, node_i]) for i in range(traffic_num)]) <= loads[node_i, node_j])
        model.addConstr(loads[node_i, node_j] <= SRLG_graph[node_i][node_j]['cap'])     # load capacity constrain
        for i in range(traffic_num):
            model.addConstr(flow_1[i, node_i, node_j]+flow_1[i, node_j, node_i]+flow_2[i, node_i, node_j]+flow_2[i, node_j, node_i] <= 1)   # main path and backup path should be disjoint
    for i in range(traffic_num):
        model.addConstr(sum(weights[i, j] for j in range(2)) == 1)
        (src, dst) = traffic_list[i]
        for node in SRLG_graph.nodes:
            if node == dst:
                rhs = -1
            elif node == src:
                rhs = 1
            else:
                rhs = 0
            model.addConstr((flow_1.sum(i, node, '*')-flow_1.sum(i, '*', node) == rhs))     # connectivity constrain
            model.addConstr((flow_2.sum(i, node, '*')-flow_2.sum(i, '*', node) == rhs))
    
    model.optimize()
    if model.status == gurobipy.GRB.Status.OPTIMAL or model.status == gurobipy.GRB.Status.TIME_LIMIT:
        traffic_paths = {}
        path_weights = {}

        paths_1 = model.getAttr('x', flow_1)
        paths_2 = model.getAttr('x', flow_2)
        w = model.getAttr('x', weights)

        for i in range(traffic_num): 
            (src, dst) = traffic_list[i]
            path_1 = [(node_i, node_j) for (node_i, node_j) in edge_list if paths_1[i, node_i, node_j] > 0]
            path_2 = [(node_i, node_j) for (node_i, node_j) in edge_list if paths_2[i, node_i, node_j] > 0]
            path_1 = convert_path(path_1, src, dst)
            path_2 = convert_path(path_2, src, dst)
            traffic_paths[(src, dst)] = [path_1, path_2]

            for j in range(2):
                path_weights[(src, dst, j)] = w[i, j]
    
        return traffic_paths, path_weights


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


def update_graph(fiber_net:nx.Graph, load_graph:nx.Graph, path, demand):
    for i in range(len(path)-1):
        node_i = path[i]
        node_j = path[i+1]
        fiber_net.add_edge(node_i, node_j)
        load_graph[node_i][node_j]['load'] += demand
    
    return fiber_net, load_graph


def linear_programing(original_graph:nx.Graph, SRLG_graph:nx.Graph, original_traffic:dict, SRLG_pair:dict, time_limit=None):
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

    traffic_paths, path_weights = gurobi_solver(SRLG_graph, original_traffic, time_limit)
    for (src, dst) in traffic_paths.keys():
        demand = original_traffic[(src, dst)]
        paths = traffic_paths[(src, dst)]
        for j in range(len(paths)): 
            for node in paths[j]:
                if node in SRLG_pair.keys():
                    paths[j][paths[j].index(node)] = SRLG_pair[node]

            paths[j] = remove_cycle(paths[j])
            weight = path_weights[(src, dst, j)]
            fiber_net, load_graph = update_graph(fiber_net, load_graph, paths[j], weight*demand)

    return fiber_net, load_graph, traffic_paths