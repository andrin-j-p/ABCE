#%%
from tester import  run_model
#from calibration import test
import numpy as np
np.random.seed(0) # enable global seed


def main():
  run_model(2)

  print('')

  # Save the simulation object
  #joblib.dump(simulation, "simulation.pkl")

  # Later, to continue the simulation
  #simulation = joblib.load("simulation.pkl")
if __name__ == "__main__":
    main()

# %%
