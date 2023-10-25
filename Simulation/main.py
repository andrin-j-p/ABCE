from ABM import run_simulation 
import numpy as np
import joblib

np.random.seed(0) # enables global seed


def main():
  run_simulation()

  # Save the simulation object
  #joblib.dump(simulation, "simulation.pkl")

  # Later, to continue the simulation
  #simulation = joblib.load("simulation.pkl")
if __name__ == "__main__":
    main()
