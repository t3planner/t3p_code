from config import setup_configs
from algos.Greenfield.main import RL_cost
from algos.Reconfiguration.main import RL_congestion
from algos.Expansion.main import RL_addnode
from algos.other_experiment.Roedunet.main import RL_Roedunet
from algos.other_experiment.T3Planner_No_GNN.main import RL_T3Planner_No_GNN
from algos.other_experiment.T3Planner_No_ker.main import RL_T3Planner_No_ker
from algos.other_experiment.T3Planner_No_Rule.main import RL_T3Planner_No_Rule
from algos.other_experiment.T3Planner_No_Compression.main import RL_T3Planner_No_Compression

if __name__ == '__main__':
    args = setup_configs()
    if args.dataset == "Roedunet" or args.scenario == "motivation":
        RL_Roedunet("Roedunet", args.del_num, args.gcn_outdim, args.steps_per_epoch, args.epochs, args.max_ep_len, args.cuda)
    elif args.dataset == "expansion" or args.scenario == "New sites expansion scenario":
        RL_addnode("New_sites_expansion_scenario", args.del_num, args.stru, args.steps_per_epoch, args.epochs, args.max_ep_len,args.cuda)
    elif args.scenario == "Greenfield scenario":
        RL_cost(args.dataset, args.del_num, args.stru, args.gcn_outdim, args.steps_per_epoch, args.epochs, args.max_ep_len, args.cuda)
    elif args.scenario == "Traffic based reconfiguration scenario":
        RL_congestion(args.dataset, args.cap, args.adjust_num, args.adjust_demand, args.stru, args.gcn_outdim, args.steps_per_epoch, args.epochs, args.max_ep_len, args.cuda)
    elif args.scenario == "No GNN":
        RL_T3Planner_No_GNN(args.dataset, args.del_num, args.stru, args.gcn_outdim, args.steps_per_epoch, args.epochs, args.max_ep_len, args.cuda)
    elif args.scenario == "No Kernel":
        RL_T3Planner_No_ker(args.dataset, args.del_num, args.stru, args.gcn_outdim, args.steps_per_epoch, args.epochs,args.max_ep_len, args.cuda)
    elif args.scenario == "No Rule":
        RL_T3Planner_No_Rule(args.dataset, args.del_num, args.stru, args.gcn_outdim, args.steps_per_epoch, args.epochs, args.max_ep_len, args.cuda)
    elif args.scenario == "No Compression":
        RL_T3Planner_No_Compression(args.dataset, args.del_num, args.stru, args.gcn_outdim, args.steps_per_epoch, args.epochs, args.max_ep_len, args.cuda)
    else:
        print("No Scenario!")