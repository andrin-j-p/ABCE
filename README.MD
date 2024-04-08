# Overview
This repository contains the code for the agent-based simulation as described in the masters's thesis "An Agent-Based Exploration of Uncondtional Cash Transfers". 

# How to run the Simulation 
Before running the simulation please make sure all dependencies as listed in 'dependencies.txt' are satisfied.

In VSCode, the easiest way to run the simulation is via the 'validation.py' file which is located in the directory 'Simulation'. The 'validation.py' file is organized as a jupyter notebook and can be executed by running the constituent code cells as described [here](https://code.visualstudio.com/docs/python/jupyter-support-py). This allows to reproduce the outcomes reported in the thesis step-by-step and reproduce the corresponding plots.  

The interactive visualization can be run from the terminal by naviaging to the 'Simulation' directory and executing the 'visualize.py' file via the following comand line code:
````
python -m visualize
````
This will first run the simulation and then creates a link to a local dash application. Following the link should automatically open the interactive visualization in the default browser. 

# Organization 
The core implementation of the simulation can be found in 'ABM.py'. Its execution relies on the following additional files:

* `read_data.py:' Implements some of the functionalities used to initialize the model, such as the spatial layout of the simulation.    

* `datacollector.py:' Implements the classes and functions used to extract agent- and aggregate level data during runtime.

* `intervention_handler.py:' Implements the randomization proceedure and intervention. 

The calibraiton proceedure described in the thesis is implemented in the 'calibration.py' file. The version employing the surrogate model is located in the options folder together with the code used to create some of the
additional figures and illustrations. 

Finally, the 'test' folder contains a set of unittests to analyise the functionality of the simulation.

# Data
All data files, as compiled by Egger et. al. (2022), and used to calibrate and validate the model are located in the directory Simulation/data/raw_data.
