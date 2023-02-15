import csv
import numpy as np
import networkx as nx

def save_result(output_dir, topo: nx.Graph=None, topo_fiber: nx.Graph=None, traffic_path: list=None):
    if traffic_path is not None:
        with open('{}//routing.csv'.format(output_dir), 'w', newline='') as f:
            writer = csv.writer(f)
            for row in traffic_path.values():
                writer.writerow(row)

    if topo is not None:
        node_num = max(topo.nodes)+1
        fiber_matrix = np.zeros((node_num, node_num))
        for (node_i, node_j) in topo.edges:
            fiber_matrix[node_i, node_j] = 1
            fiber_matrix[node_j, node_i] = 1
        with open('{}//IP_topo.csv'.format(output_dir), 'w', newline='') as f:
            writer = csv.writer(f)
            for row in fiber_matrix:
                writer.writerow(row)

    if topo_fiber is not None:
        node_num = max(topo_fiber.nodes)+1
        fiber_matrix = np.zeros((node_num, node_num))
        for (node_i, node_j) in topo_fiber.edges:
            fiber_matrix[node_i, node_j] = 1
            fiber_matrix[node_j, node_i] = 1
        with open('{}//optical_topo.csv'.format(output_dir), 'w', newline='') as f:
            writer = csv.writer(f)
            for row in fiber_matrix:
                writer.writerow(row)