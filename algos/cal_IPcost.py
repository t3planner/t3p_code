import copy
from algos.traffic import dijkstra

def Cal_IPcost(G_fiber, G_IP):
    '''
        calculate cost and the fiber path of IP links
    '''
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



