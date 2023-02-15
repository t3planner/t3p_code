import argparse

import numpy as np
import torch.nn as nn

def setup_configs(): 
    parser = argparse.ArgumentParser(description='Configurations for Traffic Based Reconfiguration Scenario')

    parser.add_argument('-dataset', type=str, default='NetD', choices=['NetD', 'ion', 'ussignal'], required=False, help='Available datasets for reconfiguration scenario.')
    parser.add_argument('-method', type=str, default='Optimal_solution', choices=['ECMP', 'FFC', 'Optimal_solution', 'Ring_Optimal_solution', 'OWAN', 'SMORE', 'Rule'], required=False, help='Available methods for reconfiguration scenario.')

    parser.add_argument('-k', type=int, default=3, required=False, help='Max number of paths for a traffic demand.')

    # time limit for gurobi solver
    parser.add_argument('-time_limit', type=int, default=60*3, required=False)

    # params for initial topology
    parser.add_argument('-init_load', type=float, default=0, required=False, help='Initial load of links.')
    parser.add_argument('-cap', type=float, default=np.inf, required=False, help='Load capacity of links.')
    parser.add_argument('-total_load', type=float, default=4484, required=False, help='Total capacity of links.')

    # params for dual-homing check
    parser.add_argument('-max_nodes', type=int, default=10, required=False, help='Max number of nodes in a ring.')
    parser.add_argument('-max_rings', type=int, default=30, required=False, help='Max number of rings to be considered.')
    
    # params for FFC
    parser.add_argument('-fails', type=int, default=2, required=False, help='Number of links will fail.')

    args = parser.parse_args()
    # args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return args