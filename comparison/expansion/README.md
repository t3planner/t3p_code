## New Sites Expansion Scenario

This part contains the source code of comparison algorithms for new sites expansion scenario.

### Content

* **`./datasets`:** Available datasets of the expansion scenario.
* **`./OWAN`:** 
  * `main.py`: The main script used to call the OWAN algorithm and save results. 
  * `solver.py`: Essential script including implementation of the simulated annealing algorithm. 
  * `utils.py`: Mainly consists of the dual-homing check algorithm. 

* **`./Rule`:** 
  * `main.py`: The main script used to call the Rule algorithm and save results. 
  * `solver.py`: Essential script including implementation of the Rule algorithm. 
  * `traffic.py`: Mainly includes the traffic allocation algorithm. 
* **`./configs.py`:** Configurations of the expansion scenario, including dataset, method, etc. 
* **`./main.py`:** The main script of expansion scenario, including input of initial topology, traffic demands and SRLG constrain.


### Note

* In expansion scenario, only OWAN and Rule algorithm is available. 
