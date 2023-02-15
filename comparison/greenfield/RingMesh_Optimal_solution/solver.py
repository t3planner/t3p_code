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


def gurobi_solver(SRLG_graph:nx.Graph, traffic:dict, SRLG_pair:dict, max_nodes=10, max_rings=30, time_limit=None, verbose=True): 
    core = []
    dis = []
    agg = []
    node_list = list(SRLG_graph.nodes)
    node_num = len(node_list)
    for node in node_list:
        if SRLG_graph.nodes[node]['type'] == 'core':
            core.append(node)
        elif SRLG_graph.nodes[node]['type'] == 'dis':
            dis.append(node)
        else:
            agg.append(node)
    
    edge_list = []
    for (node_i, node_j) in SRLG_graph.edges: 
        edge_list.append((node_i, node_j))
        edge_list.append((node_j, node_i))

    traffic_list = list(traffic.keys())
    traffic_num = len(traffic_list)

    demand_list = list(traffic.values())
    
    model = gurobipy.Model('netflow')
    model.modelSense = gurobipy.GRB.MINIMIZE
    model.Params.OutputFlag = verbose
    # model.Params.Method = 2
    model.Params.TimeLimit = time_limit

    edges = model.addVars(edge_list, vtype=gurobipy.GRB.BINARY, name='edges')
    flow_1 = model.addVars(traffic_num, edge_list, vtype=gurobipy.GRB.BINARY, name='flow_1')
    flow_2 = model.addVars(traffic_num, edge_list, vtype=gurobipy.GRB.BINARY, name='flow_2')
    
    rings = model.addVars(node_num, max_rings, vtype=gurobipy.GRB.BINARY, name='rings')
    # ends = model.addVars(dis, max_rings, vtype=gurobipy.GRB.BINARY, name='ring_ends')
    closed = model.addVars(max_rings, vtype=gurobipy.GRB.BINARY, name='closed_rings')
    adj = model.addVars(edge_list, max_rings, vtype=gurobipy.GRB.BINARY, name='adj_nodes')
    linked = model.addVars(edge_list, max_rings, vtype=gurobipy.GRB.BINARY, name='linked_nodes')
    mu = model.addVars(node_list, max_rings, lb=0, vtype=gurobipy.GRB.CONTINUOUS, name='mu')

    model.update()

    model.setObjective(sum((edges[node_i, node_j]*SRLG_graph[node_i][node_j]['cost']) for (node_i, node_j) in edge_list))
    
    for (node_i, node_j) in edge_list: 
        model.addConstr(edges[node_i, node_j] == edges[node_j, node_i])
        model.addConstr(sum([demand_list[i]*(flow_1[i, node_i, node_j]+flow_1[i, node_j, node_i]+flow_2[i, node_i, node_j]+flow_2[i, node_j, node_i]) for i in range(traffic_num)]) <= SRLG_graph[node_i][node_j]['cap'])
        for i in range(traffic_num): 
            model.addConstr((flow_1[i, node_i, node_j]+flow_1[i, node_j, node_i]+flow_2[i, node_i, node_j]+flow_2[i, node_j, node_i]) <= edges[node_i, node_j])
    
    for i in range(traffic_num):
        (src, dst) = traffic_list[i]
        for node in SRLG_graph.nodes:
            if node == dst:
                rhs = -1
            elif node == src:
                rhs = 1
            else:
                rhs = 0
            model.addConstr((flow_1.sum(i, node, '*') - flow_1.sum(i, '*', node) == rhs))
            model.addConstr((flow_2.sum(i, node, '*') - flow_2.sum(i, '*', node) == rhs))

    for node in core:
        model.addConstr(sum(edges[node, neighbor] for neighbor in SRLG_graph.adj[node])>=3) # net structure constrain
    for node in dis:
        model.addConstr(sum(edges[node, neighbor] for neighbor in SRLG_graph.adj[node])>=3) # net structure constrain
    for node in agg:
        model.addConstr(sum(rings[node, r] for r in range(max_rings)) >= 1) # agg node should be at least in one ring
    
    
    model.addConstr(sum(closed[r] for r in range(max_rings)) >= 1)  # there should be at least one ring
    
    for r in range(max_rings):
        model.addConstr(sum(rings[node, r] for node in node_list) <= max_nodes) # max num of nodes in a ring
        model.addConstr((closed[r]==1) >> (sum(rings[node, r] for node in node_list)>=3))   # min num of nodes in a closed ring
        model.addConstr((closed[r]==0) >> (sum(rings[node, r] for node in node_list)==0))   # num of nodes in a empty ring

        # model.addConstr(sum(rings[node, r] for node in core) == 0)  # none core node in a ring
        model.addConstr((closed[r]==1) >> (sum(rings[node, r] for node in dis)==2)) # there should be at least two dis nodes in a closed ring
        for (node_s, node) in SRLG_pair.items():
            if node in dis: 
                model.addConstr(rings[node, r]+rings[node_s, r] <= 1)
        # model.addConstr((closed[r]==1) >> (sum(ends[node, r] for node in dis)==2))    # there should be and only be two ends in a closed ring
        # model.addConstr((closed[r]==0) >> (sum(ends[node, r] for node in dis)==0))    # num of ends in a empty ring

        for (node_i, node_j) in edge_list: 
            model.addConstr(adj[node_i, node_j, r] == rings[node_i, r]*rings[node_j, r])
            model.addConstr(linked[node_i, node_j, r] == adj[node_i, node_j, r]*edges[node_i, node_j])

        for node in agg+core:
            model.addConstr((rings[node, r]==1) >> (sum(linked[node, neighbor, r] for neighbor in SRLG_graph.adj[node])==2))    # degree of agg node in a closed ring should be 2
        for node in dis: 
            # model.addConstr((rings[node, r]==1) >> (ends[node, r]==1))  # dis nodes must be ends
            neighbors = set(SRLG_graph.adj[node]) & set(agg+core)
            if len(neighbors) > 0: 
                model.addConstr((rings[node, r]==1) >> (sum(linked[node, neighbor, r] for neighbor in neighbors)==1))     # degree of dis node (end) in a ring should be 1
            else:
                model.addConstr(rings[node, r]==0)
        
        for (node_i, node_j) in SRLG_graph.edges:
            model.addConstr((closed[r]==1) >> (mu[node_i, r]-mu[node_j, r]+node_num*linked[node_i, node_j, r]<=node_num-1))     # MTZ constrain
    
    model.optimize()
    if model.status == gurobipy.GRB.Status.OPTIMAL or model.status == gurobipy.GRB.Status.TIME_LIMIT:
        node_ring = {}
        rings = model.getAttr('x', rings)
        for node in node_list:
            for r in range(max_rings):
                if rings[node, r] > 0: 
                    try: 
                        node_ring[node].append(r)
                    except:
                        node_ring[node] = [r]
        
        fiber_net = nx.Graph()
        links = model.getAttr('x', edges)
        for (node_i, node_j) in edge_list:
            if links[node_i, node_j] > 0:
                fiber_net.add_edge(node_i, node_j)

        traffic_paths = {}
        paths_1 = model.getAttr('x', flow_1)
        paths_2 = model.getAttr('x', flow_2)
        for i in range(traffic_num): 
            (src, dst) = traffic_list[i]
            path_1 = [(node_i, node_j) for (node_i, node_j) in edge_list if paths_1[i, node_i, node_j] > 0]
            path_2 = [(node_i, node_j) for (node_i, node_j) in edge_list if paths_2[i, node_i, node_j] > 0]
            path_1 = convert_path(path_1, src, dst)
            path_2 = convert_path(path_2, src, dst)
            traffic_paths[(src, dst)] = [path_1, path_2]

        return fiber_net, traffic_paths


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


def update_graph(load_graph:nx.Graph, path, demand):
    for i in range(len(path)-1):
        node_i = path[i]
        node_j = path[i+1]
        load_graph[node_i][node_j]['load'] += demand
    
    return load_graph


def linear_programing(
    original_graph:nx.Graph, SRLG_graph:nx.Graph, original_traffic:dict, SRLG_pair:dict, 
    max_nodes=10, max_rings=30, time_limit=None
):
    fiber_net = nx.Graph()
    fiber_net.add_nodes_from(original_graph)

    load_graph = nx.Graph()
    for (node_i, node_j) in original_graph.edges:
        load_graph.add_edge(node_i, node_j, load=0)

    SRLG_fiber_net, SRLG_traffic_paths = gurobi_solver(SRLG_graph, original_traffic, SRLG_pair, max_nodes, max_rings, time_limit)
    for (node_i, node_j) in SRLG_fiber_net.edges:
        if node_i in SRLG_pair.keys(): 
            if node_j != SRLG_pair[node_i]:
                fiber_net.add_edge(SRLG_pair[node_i], node_j)
        elif node_j in SRLG_pair.keys(): 
            if node_i != SRLG_pair[node_j]:
                fiber_net.add_edge(node_i, SRLG_pair[node_j])
        else:
            fiber_net.add_edge(node_i, node_j)

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

        load_graph = update_graph(load_graph, path_1, demand)
        load_graph = update_graph(load_graph, path_2, demand)

    return fiber_net, load_graph, traffic_paths