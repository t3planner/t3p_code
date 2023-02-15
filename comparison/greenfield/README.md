## Greenfield Scenario

This part contains the source code of comparison algorithms for the greenfield minimization scenario.

### Content

* **`./datasets`:** Available datasets of addition scenario. 
* **`./MLBO`:** 
  * `main.py`: The main script used to call the MLBO algorithm and save results. 
  * `solver.py`: Essential script including implementation of MLBO algorithm. 
  * `utils.py`: Mainly consists of the dual-homing check algorithm. 
* **`./Optimal_solution`:** 
  * `main.py`: The Main script used to call the Optimal_solution algorithm and save results. 
  * `solver.py`: Essential script including implementation of Optimal_solution algorithm. 
* **`./Mesh_Optimal_solution`:** 
  * `main.py`: The main script used to call the Mesh_Optimal_solution algorithm and save results. 
  * `solver.py`: Essential script including implementation of Mesh_Optimal_solution algorithm.  
* **`./NeuroPlan`:** 
  - `env.py`: Simulation of RL environment, outputs rewards according to actions. 
  - `logger.py`: Utilities for logging results. 
  - `main.py`: The main script used to call the NeuroPlan algorithm and save results. 
  - `model.py`: Implementation of Actor-Critic model. 
  - `traffic_allocation.py`: Mainly includes traffic allocation algorithm. 
  - `utils.py`: Mainly consists of the dual-homing check algorithm. 
  - `VPG.py`: Implementation of policy gradient algorithm. 
* **`./Ring_Optimal_solution:** 
  * `main.py`: The main script used to call the Ring_Optimal_solution algorithm and save results. 
  * `solver.py`: Essential script including implementation of Ring_Optimal_solution algorithm.  
* **`./RingMesh_Optimal_solution`:** 
  * `main.py`: The main script used to call the RingMesh_Optimal_solution algorithm and save results. 
  * `solver.py`: Essential script including implementation of RingMesh_Optimal_solution algorithm.  
* **`./OWAN`:** 
  * `main.py`: The main script used to call the OWAN algorithm and save results. 
  * `solver.py`: Essential script including implementation of simulated annealing algorithm. 
  * `utils.py`: Mainly consists of the dual-homing check algorithm. 
* **`./OWAN_RoEdu`:** 
  * `main.py`: The main script used to call the OWAN algorithm and save results. 
  * `solver.py`: Essential script including implementation of simulated annealing algorithm. 
  * `utils.py`: Mainly consists of the dual-homing check algorithm. 
* **`./Rule`:** 
  * `main.py`: The main script used to call the Rule algorithm and save results. 
  * `solver.py`: Essential script including implementation of Rule algorithm. 
  * `traffic.py`: Mainly includes traffic allocation algorithm. 
* **`./configs.py`:** Configurations of greenfield scenario, including dataset, method, time limit, etc. 
* **`./main.py`:** The main script of the greenfield scenario, including input of initial topology, traffic demands, and SRLG constrain.
* **`./utils.py`:** Includes algorithm calculating greenfield of IP links according to fiber net.

### Note

* `Ring_Optimal_solution` is a variant of `Optimal_solution`, considering dual-homing ring structure constrain. 
* `Mesh_Optimal_solution` is a variant of `Optimal_solution`, considering partial mesh structure constrain. 
* `RingMesh_Optimal_solution` is a variant of `Optimal_solution`, considering hybrid ring-mesh structure constrain. 
* For `roedunet` dataset, only `OWAN_roedunet` method is available, and the implementation of `OWAN_roedunet` is different from `OWAN` for other datasets.
* For `NeuroPlan` method, advanced congratulations are available, including `-delta_load`, `-max_deltas`, and `-feature_dim`. Please check `./configs.py` for details. 