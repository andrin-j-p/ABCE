#%%%
import numpy as np
import pandas as pd
import ABM
import timeit
import dill 


class Model(ABM.Sugarscepe):
  def __init__(self):
    super().__init__()
    print('init was called')

  def set_parameters(self, fm_sample):
    """
    Type:        Method
    Description: Overwites the model parameters as specified in the 'parameters' argument
    """
    for firm in self.all_firms:
      for parameter, value in fm_sample.items():
        setattr(firm, parameter, value)

  def run_simulation(self, steps):

    """
    Type: Method
    Description: overwrite run simulaiton in parent class to avoid print statement
    """
    for i in range(steps):
      self.step()

  
    
def run_model(steps):
  simulation = Model()
  with open('../data/simulation_initial_state.dill', 'wb') as file:
    dill.dump(simulation, file)

  for i in range(steps):
    start = timeit.default_timer()
    with open('../data/simulation_initial_state.dill', 'rb') as file:
      sim_copy = dill.load(file)

    sim_copy.run_simulation(10)
    _, _, df_md, _ = sim_copy.datacollector.get_data()
    print(df_md)

    stop = timeit.default_timer()
    print('Time: ', stop - start)  

#run_model(5)


# %%
