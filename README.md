## T3Planner Algorithms

Here, we provide the implementations of T3Planner algorithms.

T3Planner is a practical machine learning solver to the network topology planning problem.

It can generate multi-layer and structure-constrained topology plans for multi planning phases, without hand-tuned strategy.


### Content

* **`./algos`:**  The program of T3Planner algorithm.
* **`./compression`:**  The program of comparison algorithms.
* **`./datasets`:** Available datasets of the experiment.
* **`./spinup`:**  The program of RL.
* **`./configs.py`:** Configurations of the experiment, including dataset, method, etc. 
* **`./main.py`:** The main script of the experiment.

### Dependencies

```
 gurobipy
 gym
 networkx
 pandas
 numpy
 scipy
 torch-geometric
 torch == 1.7.0
 mpi4py == 3.0.3
 tensorflow == 1.15.4
```

### Run T3Planner Algorithms

* Run `python main.py -scenario {scenario} -dataset {dataset}  `. You can replace `{sceniros}` and `{dataset}` with different scenarios (e.g. Greenfield scenarios, Traffic  based reconfiguration scenario, New sites expansion scenario, No GNN, No Kernel, No Rule, No Compression) and dataset (e.g. NetD, US signal, ION, Roedunet, expansion). 

* The results are stored in the `./result` folder.

  **Note:**

* If the scenario is 'Greenfield scenario', 'Traffic-based reconfiguration scenario, 'No GNN', 'No Kernel', 'No Rule' or 'No Compression', then the datasets can be selected in 'NetD', 'ION', 'US signal'.

* If the scenario is 'New sites expansion scenario', then the dataset is 'expansion'.

* If the scenario is 'motivation', then the dataset is 'Roedunet'.
