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

    commodity_list = list(traffic.keys())
    traffic_num = len(commodity_list)
    
    model = gurobipy.Model('netflow')
    model.modelSense = gurobipy.GRB.MINIMIZE
    model.Params.OutputFlag = verbose
    model.Params.Method = 2
    model.Params.TimeLimit = time_limit

    phy_link = model.addVars(edge_list, vtype=gurobipy.GRB.BINARY, name='phy_link')
    flow_1 = model.addVars(traffic_num, edge_list, vtype=gurobipy.GRB.BINARY, name='flow_1')
    flow_2 = model.addVars(traffic_num, edge_list, vtype=gurobipy.GRB.BINARY, name='flow_2')
    model.update()

    model.setObjective(gurobipy.quicksum((phy_link[node_i, node_j]*SRLG_graph[node_i][node_j]['cost']) for (node_i, node_j) in edge_list))
    
    # main path and backup path should be disjoint
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
            model.addConstr((flow_1.sum(i, node, '*') - flow_1.sum(i, '*', node) == rhs), '{}_{}'.format(node, src))    # connectivity constrain
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

    load_graph = nx.Graph()
    for (node_i, node_j) in original_graph.edges:
        load_graph.add_edge(node_i, node_j, load=0)

    SRLG_traffic_paths = gurobi_solver(SRLG_graph, original_traffic, time_limit=time_limit)
    traffic_paths = {}
    for (src, dst), paths in SRLG_traffic_paths.items(): 
        demand = original_traffic[(src, dst)]

        path_1 = paths[0]
        path_2 = paths[1]

        for node in path_1:
            if node in SRLG_pair.keys():
                path_1[path_1.index(node)] = SRLG_pair[node]
        for node in path_2:
            if node in SRLG_pair.keys():
                path_2[path_2.index(node)] = SRLG_pair[node]
        path_1 = remove_cycle(path_1)
        path_2 = remove_cycle(path_2)

        traffic_paths[(src, dst)] = [path_1, path_2]

        fiber_net, load_graph = update_graph(fiber_net, load_graph, path_1, demand)
        fiber_net, load_graph = update_graph(fiber_net, load_graph, path_2, demand)

    return fiber_net, load_graph, traffic_paths