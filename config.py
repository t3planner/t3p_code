import argparse
import torch
import numpy as np
import torch.nn as nn

def setup_configs():
    parser = argparse.ArgumentParser(description='T3Planner')
    '''
        if the scenario is 'Greenfield scenario', 'Traffic  based reconfiguration scenario', 'No GNN', 'No Kernel', 'No Rule', 'No Compression',
            then the datasets can be selected in 'NetD', 'ION', 'US signal'.
        if the scenario is 'New sites expansion scenario', then the datasets is 'expansion'.
        if the scenario is 'motivation', then the datasets is 'Roedunet'.
    '''
    parser.add_argument('-scenario', type=str, default='Greenfield scenario', choices=['Greenfield scenario', 'Traffic based reconfiguration scenario', 'New sites expansion scenario', 'No GNN', 'No Kernel', 'No Rule', 'No Compression', 'motivation'], required=False, help='Available scenario.')
    parser.add_argument('-dataset', type=str, default='NetD', choices=['NetD', 'ION', 'Roedunet', 'US signal', 'expansion'], required=False, help='Available datasets.')
    parser.add_argument('-stru', type=str, default='dual-homing', choices=['dual-homing', 'partial-mesh', 'hybrid ring-mesh'], required=False, help='Available structure.')

    #  if scenario != "Traffic  based reconfiguration scenario"
    parser.add_argument('-del_num', type=int, default=1, required=False)
    #  if scenario == "Traffic  based reconfiguration scenario"
    parser.add_argument('-init_load', type=float, default=0, required=False)
    parser.add_argument('-cap', type=float, default=40000, required=False)
    parser.add_argument('-adjust_num', type=int, default=5, required=False)
    parser.add_argument('-adjust_demand', type=int, default=10, required=False)


    parser.add_argument('-gcn_outdim', type=int, default=1, required=False)
    parser.add_argument('-steps_per_epoch', type=int, default=20, required=False)
    parser.add_argument('-epochs', type=int, default=2000, required=False)
    parser.add_argument('-max_ep_len', type=int, default=5, required=False)

    # if stru == "dual-homing"
    parser.add_argument('-ring_len', type=int, default=10, required=False)
    parser.add_argument('-max_rings', type=int, default=30, required=False)

    args = parser.parse_args()
    args.cuda = True if torch.cuda.is_available() else False

    return args