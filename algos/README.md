## T3Planner Algorithms

Here, we provide the implementations of T3Planner algorithms for three scenarios: 

* **Greenfield planning Scenario (`./Greenfield`):** Given initial topology and traffic demands, construct a topology with minimal cost of fibers and not violate any constraints (e.g. SRLG constraints, structural constraints, load capacity constraints). 
  * **Objective: **building cost minimization
  * **State: **current IP topology, structural feature per IP link, building cost per IP link
  * **Action: **remove an IP link
  * **Reward:** change in building cost obtained by Min-Cost approach

* **Traffic-Based Reconfiguration Scenario (`./Reconfiguration`):** Given initial topology and traffic demands, construct a topology with minimal congestion of edges and not violate any constraints (e.g. SRLG constraints, structural constraints, load capacity constraints). 
  * **Objective: **congestion minimization
  * **State: **current IP topology, structural feature per IP link, throughput per IP link
  * **Action: ** remove an IP link and add a potential IP link
  * **Reward:** change in congestion obtained by Min-Congestion approach

* **New Sites Expansion Scenario (`./Expansion `):** Given initial topology, traffic demands, and nodes to be added, add these nodes to the initial topology with minimal cost of added edges and not violate any constraints (e.g. SRLG constraints, structural constraints, load capacity constraints). 
  * **Objective: **migration cost minimization
  * **State: **current IP topology, structural feature per IP link, migration cost per IP link
  * **Action: ** remove an IP link
  * **Reward:** change in migration cost obtained by Min-Cost approach


### Content

* **`./dual_homing`:** Dual-homing check algorithm. Judge whether the topology meets the dual-homing structure.
* **`./Expansion`:**  
  * `core.py`: Implementation of Actor-Critic model in RL controller model. 
  * `plan_env.py`: Simulation of RL environment, outputs rewards according to actions. 
  * `main.py`: The main script of the new sites expansion scenario, including input of initial topology, traffic demands, and SRLG constraints.
* **`./GED`:** Implementation of the graph edit distance algorithm.
* **`./GNN`:** Implementation of the GNN model.
* **`./Greenfield`:** 
  * `core.py`: Implementation of the Actor-Critic model in RL controller model. 
  * `plan_env.py`: Simulation of RL environment, outputs rewards according to actions. 
  * `main.py`: The main script of the greenfield planning scenario, including input of initial topology, traffic demands, and SRLG constraints.

* **`./other_experiment`:**
  * `./Roedunet`: Implementation of the experiment to address motivation two.
  * `./T3Planner_No_Compression`: Implementation of T3Planner without top-down planning.
  * `./T3Planner_No_GNN`: Implementation of T3Planner without structure-aware GNN.
  * `./T3Planner_No_ker`: Implementation of T3Planner without graph kernel encoding.
  * `./T3Planner_No_Rule`: Implementation of T3Planner without heuristic-based rewarding.

* **`./Reconfiguration`:**  
  * `core.py`: Implementation of the Actor-Critic model in RL controller model. 
  * `plan_env.py`: Simulation of RL environment, outputs rewards according to actions. 
  * `main.py`: The main script of the traffic-based reconfiguration scenario, including input of initial topology, traffic demands, and SRLG constraints.

* **`./cal_IPcost.py`:** Includes algorithm calculating greenfield of IP links according to fiber net.
* **`./heur.py`:** Implementation of heuristic-driven evaluator model. 
* **`./ppo.py`:** Implementation of policy gradient algorithm in RL controller model. 
* **`./save_result.py`:** Implementation of saving results. 
* **`./structure_encoder.py`:** Implementation of the structure encoder model. 
* **`./traffic.py`:** Mainly includes the traffic allocation algorithm. 

