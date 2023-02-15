## Traffic-Based Reconfiguration Scenario

This part contains the source code of comparison algorithms for traffic-based reconfiguration scenario.

### Content

* **`./datasets`:** Available datasets of the reconfiguration scenario.
* **`./ECMP`:** 
  * `main.py`: The main script used to call the ECMP algorithm and save results. 
  * `solver.py`: Essential script including implementation of the ECMP algorithm. 
* **`./FFC`:** 
  * `main.py`: The main script used to call the FFC algorithm and save results. 
  * `solver.py`: Essential script including implementation of the FFC algorithm. 
* **`./Optimal_solution`:** 
  * `main.py`: The Main script used to call the Optimal_solution algorithm and save results. 
  * `solver.py`: Essential script including implementation of the Optimal_solution algorithm. 
* **`./Ring_Optimal_solution`:** 
  * `main.py`: The main script used to call the Ring_Optimal_solution algorithm and save results. 
  * `solver.py`: Essential script including implementation of the Ring_Optimal_solution algorithm.  
* **`./OWAN`:** 
  * `main.py`: The main script used to call the OWAN algorithm and save results. 
  * `solver.py`: Essential script including implementation of the OWAN algorithm. 
  * `traffic.py`: Mainly includes the traffic allocation algorithm. 
  * `utils.py`: Mainly consists of the dual-homing check algorithm. 
* **`./Rule`:** 
  * `main.py`: The main script used to call the Rule algorithm and save results. 
  * `solver.py`: Essential script including implementation of the Rule algorithm. 
  * `traffic.py`: Mainly includes the traffic allocation algorithm. 
* **`./SMORE`:** 
  * `main.py`: The main script used to call the SMORE algorithm and save results. 
  * `solver.py`: Essential script including implementation of the SMORE algorithm. 
* **`./configs.py`:** Configurations of the reconfiguration scenario, including dataset, method, time limit, etc. 
* **`./main.py`:** The main script of the reconfiguration scenario, including input of initial topology, traffic demands and SRLG constrain.
* **`./utils.py`:** Includes algorithm calculating cost of IP links according to fiber net.

### Note

* `Ring_Optimal_solution` is a variant of `Optimal_solution`, considering dual-homing ring structure constrain. 