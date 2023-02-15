import copy

import heapq as hp
import networkx as nx


def dijkstra(graph:nx.Graph, src, dst=None):
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

# calculate cost of IP links
def calculate_IP_cost(G_fiber, G_IP):
    graph = copy.deepcopy(G_fiber)
    for (node_i, node_j) in G_IP.edges():
        path = dijkstra(graph, node_i, node_j)
        for j in range(len(path)-1):
            graph[path[j]][path[j+1]]['cost'] += 1
        G_IP[node_i][node_j]['fiber'] = copy.deepcopy(path)
        cost = 0
        for j in range(len(path)-1):
            cost += G_fiber[path[j]][path[j+1]]['cost']
        G_IP[node_i][node_j]['cost'] = cost
    return G_IP



