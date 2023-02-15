import torch
import argparse

import numpy as np
import torch.nn as nn

def setup_configs(): 
    parser = argparse.ArgumentParser(description='Configurations for Greenfield Scenario')

    parser.add_argument('-dataset', type=str, default='NetD', choices=['NetD', 'ion', 'roedunet', 'ussignal'], required=False, help='Available datasets for greenfield scenario.')
    parser.add_argument('-method', type=str, default='Optimal_solution', choices=['MLBO', 'Optimal_solution', 'Mesh_Optimal_solution', 'NeuroPlan', 'Ring_Optimal_solution', 'RingMesh_Optimal_solution', 'OWAN', 'OWAN_roedunet', 'Rule'],  required=False, help='Available methods for greenfield scenario.')

    # params for dual-homing check
    parser.add_argument('-max_nodes', type=int, default=10, required=False, help='Max number of nodes in a ring.')
    parser.add_argument('-max_rings', type=int, default=30, required=False, help='Max number of rings to be considered.')

    # time limit for gurobi solver
    parser.add_argument('-time_limit', type=int, default=60*20, required=False)

    # params for initial topology
    parser.add_argument('-init_load', type=float, default=0, required=False, help='Initial load of links.')
    parser.add_argument('-cap', type=float, default=np.inf, required=False, help='Load capacity of links.')

    # params for NeuroPlan
    parser.add_argument('-delta_load', type=float, default=200, required=False, help='Basic step the agent will take in one interaction')
    parser.add_argument('-max_deltas', type=int, default=4, required=False, help='Max number of steps the agent will take in one interaction')
    parser.add_argument('-feature_dim', type=int, default=2, required=False, help='Feature dimension of input topology.')

    args = parser.parse_args()
    # args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return args