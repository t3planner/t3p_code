## Comparison Algorithms

Here, we provide the implementations of comparison algorithms for three scenarios: 

* **Greenfield Scenario (`./greenfield`):** Given initial topology and traffic demands, construct a topology with minimal cost of edges and not violate any constraints (e.g. SRLG constrain, structural constrain, load capacity constrain). 
* **Traffic-Based Reconfiguration Scenario (`./reconfiguration`):** Given initial topology and traffic demands, construct a topology with minimal congestion of edges and not violate any constraints (e.g. SRLG constrain, structural constrain, load capacity constrain). 
* **New Sites Expansion Scenario (`./expansion `):** Given initial topology, traffic demands, and nodes to be added, add these nodes to the initial topology with minimal cost of added edges and not violate any constraints (e.g. SRLG constrain, structural constrain, load capacity constrain). 

### Dependencies

```
 gurobipy
 gym
 networkx
 torch == 1.10.2
 torch-geometric == 2.0.4
```

### Run Comparison Algorithms

* **Greenfield Scenario:** Make sure your current path is `./greenfield`, and run `python main.py -method {method} -dataset {dataset}  `. You can replace `{method}` and `{dataset}` with different methods (e.g. MLBO, Optimal solution, NeuroPlan, OWAN, Rule) and dataset (e.g. NetD, ION, Roedunet, US signal). 
* **Traffic-Based Reconfiguration Scenario:** Make sure your current path is `./reconfiguration `, and run `python main.py -method {method} -dataset {dataset}  `. You can replace `{method}` and `{dataset}` with different methods (e.g. ECMP, Optimal solution, OWAN, SMORE, Rule) and dataset (e.g. NetD, ION, US signal). 
* **New Sites Expansion Scenario:**  Make sure your current path is `./expansion `, and run `python main.py -method {method} -dataset {dataset}  `, in the expansion scenario, only OWAN and Rule method are available. 
