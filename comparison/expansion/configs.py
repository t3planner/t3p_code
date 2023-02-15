import torch
import argparse

import numpy as np
import torch.nn as nn

def setup_configs(): 
    parser = argparse.ArgumentParser(description='Configurations for Expansion Scenario')

    parser.add_argument('-dataset', type=str, default='expansion', choices=['expansion'], required=False, help='Available dataset for expansion scenario.')
    parser.add_argument('-method', type=str, default='Rule', choices=['OWAN', 'Rule'],  required=False, help='Available method for expansion scenario.')

    # params for dual-homing check
    parser.add_argument('-max_nodes', type=int, default=10, required=False, help='Max number of nodes in a ring.')

    # params for initial topology
    parser.add_argument('-init_load', type=float, default=0, required=False, help='Initial load of links.')
    parser.add_argument('-cap', type=float, default=np.inf, required=False, help='Load capacity of links.')

    args = parser.parse_args()
    # args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return args