from algos.dual_homing.ring import rings_flag
import networkx as nx
import copy
from algos.traffic import *

def Heur_cost(G_fiber: nx.Graph, graph: nx.Graph, traffic, SRLG):
    '''
        calculate the routing of the traffic on the graph and calculate the value of cost, the IP topo, the fiber topo
    '''
    fiber_graph = copy.deepcopy(graph)
    fiber_graph, SRLG_pair = SRLG_constrain(fiber_graph, SRLG)
    topo = nx.create_empty_copy(graph)
    traffic_path = {}
    flag = 0
    for f in traffic:
        src = f[0]
        dst = f[1]
        # find two paths for flow
        path_1, path_2 = suurballe(fiber_graph, src, dst)
        path_1 = remove_cycle(path_1)
        path_2 = remove_cycle(path_2)

        if path_1 != [] and path_2 != []:
            for i in range(1, len(path_1)):
                fiber_graph[path_1[i - 1]][path_1[i]]['cost'] = 0
            for i in range(1, len(path_2)):
                fiber_graph[path_2[i - 1]][path_2[i]]['cost'] = 0
        for node in path_1:
            if node in SRLG_pair.keys():
                path_1[path_1.index(node)] = SRLG_pair[node]
        for node in path_2:
            if node in SRLG_pair.keys():
                path_2[path_2.index(node)] = SRLG_pair[node]
        path_1 = remove_cycle(path_1)
        path_2 = remove_cycle(path_2)
        if path_1 == [] or path_2 == []:
            flag = 1
            return flag, None, None, None, 0
        else:
            traffic_path[f] = [copy.deepcopy(path_1), copy.deepcopy(path_2)]
            for i in range(1, len(path_1)):
                topo.add_edge(path_1[i - 1], path_1[i], fiber=graph[path_1[i-1]][path_1[i]]['fiber'])
            for i in range(1, len(path_2)):
                topo.add_edge(path_2[i - 1], path_2[i], fiber=graph[path_2[i-1]][path_2[i]]['fiber'])
    # compute fiber cost
    topo_fiber = nx.Graph()
    for (node_i, node_j) in topo.edges:
        fiber_path = topo[node_i][node_j]['fiber']
        for k in range(1, len(fiber_path)):
            topo_fiber.add_edge(fiber_path[k-1], fiber_path[k])
    cost = 0
    for (node_i, node_j) in topo_fiber.edges:
        cost += G_fiber[node_i][node_j]['cost']

    return flag, topo, topo_fiber, traffic_path, cost

def calaulate_congestion(original_graph: nx.Graph, traffic, SRLG):
    '''
        calculate the routing of the traffic on the graph and calculate the value of congestion
    '''
    graph = copy.deepcopy(original_graph)
    graph, SRLG_pair = SRLG_constrain(graph, SRLG)
    topo = nx.create_empty_copy(graph)  # Record result topology
    flow_path = {(f[0], f[1]): [] for f in traffic}
    flag = 0
    all_distances = compute_all_distances(graph, traffic)
    all_distances = sorted(all_distances.items(), key=lambda kv: (kv[1], kv[0]))
    for f in all_distances:
        src = f[0][0]
        dst = f[0][1]
        demand = traffic[(src, dst)]
        graph_new = demand_constrain(graph, demand)
        path_1, path_2 = suurballe(graph_new, src, dst)
        path_1 = remove_cycle(path_1)
        path_2 = remove_cycle(path_2)

        if path_1 != [] and path_2 != []:
            for i in range(1, len(path_1)):
                graph[path_1[i - 1]][path_1[i]]['cost'] += 10
                graph[path_1[i - 1]][path_1[i]]['load'] += traffic[(src, dst)]
            for i in range(1, len(path_2)):
                graph[path_2[i - 1]][path_2[i]]['cost'] += 10
                graph[path_2[i - 1]][path_2[i]]['load'] += traffic[(src, dst)]
        for node in path_1:
            if node in SRLG_pair.keys():
                path_1[path_1.index(node)] = SRLG_pair[node]
        for node in path_2:
            if node in SRLG_pair.keys():
                path_2[path_2.index(node)] = SRLG_pair[node]
        path_1 = remove_cycle(path_1)
        path_2 = remove_cycle(path_2)

        if path_1 == [] or path_2 == []:
            flag = 1
            return None, 0, None, None, 0, flag
        else:
            for i in range(1, len(path_1)):
                if (path_1[i-1], path_1[i]) not in topo.edges:
                    topo.add_edge(path_1[i - 1], path_1[i], flow=[])

            for i in range(1, len(path_2)):
                if (path_2[i - 1], path_2[i]) not in topo.edges:
                    topo.add_edge(path_2[i - 1], path_2[i], flow=[])
            if len(path_1) < len(path_2):
                for i in range(1, len(path_1)):
                    original_graph[path_1[i - 1]][path_1[i]]['load'] += traffic[(src, dst)]
                    topo[path_1[i - 1]][path_1[i]]['flow'].append([(src, dst), traffic[(src, dst)], len(path_1)])
                    flow_path[(src, dst)] = [path_1, path_2, 1]
            else:
                for i in range(1, len(path_2)):
                    original_graph[path_2[i - 1]][path_2[i]]['load'] += traffic[(src, dst)]
                    topo[path_2[i - 1]][path_2[i]]['flow'].append([(src, dst), traffic[(src, dst)], len(path_2)])
                    flow_path[(src, dst)] = [path_1, path_2, 2]
    congestion = max([original_graph[node_i][node_j]['load'] for (node_i, node_j) in topo.edges])
    unallacted_demand = 0
    flow = []
    for (node_i, node_j) in original_graph.edges:
        unallacted_demand += (original_graph[node_i][node_j]['cap']-original_graph[node_i][node_j]['load'])
        if original_graph[node_i][node_j]['load'] == congestion:
            flow = topo[node_i][node_j]['flow']
    unallacted_demand += sum([f[1]*f[2] for f in flow])  # Record the remaining bandwidth

    return original_graph, congestion, flow, flow_path, unallacted_demand, flag


def Heur_congestion(original_graph, graph, traffic, SRLG):
    '''
        regulate the traffic on the edge with the most load traffic
    '''
    graph_new = copy.deepcopy(graph)
    graph_new, congestion, flow, flow_path, unallacted_demand, flag = calaulate_congestion(graph_new, traffic, SRLG)

    if flag == 1:
        return flag, None, None, None, None

    flow_path_old = copy.deepcopy(flow_path)
    allcated_demand = 0  # Record how much bandwidth is used
    topo = nx.create_empty_copy(original_graph)
    for (node_i, node_j) in original_graph.edges:
        topo.add_edge(node_i, node_j, load=0, cost=1, cap=np.inf)
    for f in flow:
        src = f[0][0]
        dst = f[0][1]
        demand = f[1]
        path = flow_path[(src, dst)][flow_path[(src, dst)][2]-1]
        for i in range(1, len(path)):
            graph_new[path[i - 1]][path[i]]['load'] -= demand
        for (node_i, node_j) in topo.edges:
            topo[node_i][node_j]['load'] = 0
            topo[node_i][node_j]['cost'] = 1
        for edge in topo.edges:
            if edge in graph_new.edges:
                if demand+graph_new[edge[0]][edge[1]]['load'] >= congestion-demand:
                    topo[edge[0]][edge[1]]['cost'] = np.inf
        fiber_graph = copy.deepcopy(topo)
        fiber_graph, SRLG_pair = SRLG_constrain(fiber_graph, SRLG)
        path_1, path_2 = suurballe(fiber_graph, src, dst)
        path_1 = remove_cycle(path_1)
        path_2 = remove_cycle(path_2)

        for node in path_1:
            if node in SRLG_pair.keys():
                path_1[path_1.index(node)] = SRLG_pair[node]
        for node in path_2:
            if node in SRLG_pair.keys():
                path_2[path_2.index(node)] = SRLG_pair[node]
        path_1 = remove_cycle(path_1)
        path_2 = remove_cycle(path_2)

        if path_1 == [] or path_2 == []:
            return 0, graph, None, flow_path_old, congestion
        else:
            if len(path_1) < len(path_2):
                for i in range(1, len(path_1)):
                    if (path_1[i-1], path_1[i]) not in graph_new.edges:
                        graph_new.add_edge(path_1[i-1], path_1[i], load=demand)
                    else:
                        graph_new[path_1[i - 1]][path_1[i]]['load'] += demand
                    allcated_demand += demand
                    flow_path[(src, dst)] = [path_1, path_2, 1]
            else:
                for i in range(1, len(path_2)):
                    if (path_2[i-1], path_2[i]) not in graph_new.edges:
                        graph_new.add_edge(path_2[i-1], path_2[i], load=demand)
                    else:
                        graph_new[path_2[i - 1]][path_2[i]]['load'] += demand
                    allcated_demand += demand
                    flow_path[(src, dst)] = [path_1, path_2, 2]

    if allcated_demand > unallacted_demand or max([graph_new[node_i][node_j]['load'] for (node_i, node_j) in graph_new.edges]) > congestion:
        return 0, graph, None, flow_path_old, congestion
    return 0, graph_new, None, flow_path, max([graph_new[node_i][node_j]['load'] for (node_i, node_j) in graph_new.edges])


def Heur(stru, sceniors, fiber_graph, graph, traffic, SRLG, node_list=None, original_graph=None, G_init=None):
    flag = True
    if sceniors == "cost":
        traffic_flag, topo, topo_fiber, traffic_path, obj = Heur_cost(fiber_graph, graph, traffic, SRLG)  # determine whether the topology meets the traffic constraints
    elif sceniors == "congestion":
        traffic_flag, topo, topo_fiber, traffic_path, obj = Heur_congestion(original_graph, graph, traffic, SRLG)
    elif sceniors == "addnode":
        traffic_flag, topo, topo_fiber, traffic_path, obj = Heur_cost(fiber_graph, graph, traffic, SRLG)  # determine whether the topology meets the traffic constraints
        if traffic_flag == 0:
            topo.add_edges_from(G_init.edges)
    else:
        print("No sceniors!")

    if traffic_flag == 1:  # false edge
        print("traffic_vaild")
        flag = False
        return flag, None, None, None, 0
    else:
        if sceniors == "addnode":
            stru_flag = rings_flag(topo, ring_len=10, node_list=node_list)
        elif stru == "dual-homing":
            stru_flag = rings_flag(topo, ring_len=10)   # determine whether the topology meets the structural constraints
        elif stru == "partial-mesh":
            topo = graph
            stru_flag = 0  # determine whether the topology meets the structural constraints
            for node in graph.nodes:
                if graph.degree(node) < 3:
                    stru_flag = 1
                    break
        elif stru == "hybrid ring-mesh":
            topo = graph
            stru_flag = 0
            ring_flag = rings_flag(graph, ring_len=10)  # determine whether the topology meets the structural constraints
            mesh_flag = 0
            for node in graph.nodes:
                if graph.nodes()[node]['type'] != 'Agg' and graph.degree(node) < 3:
                    mesh_flag = 1
                    break
            if ring_flag == 1 or mesh_flag == 1:
                stru_flag = 1

        if stru_flag == 1:  # false edge
            print("stru_vaild")
            flag = False
            return flag, None, None, None, 0

        return flag, topo, topo_fiber, traffic_path, obj
