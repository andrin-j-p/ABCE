
from ABM import Sugarscepe
import numpy as np
import random 
import copy

def main():
  np.random.seed(0) # enable global seed
  random.seed(0)
  steps = 50
  model = Sugarscepe()
  model.run_simulation(steps)
  model_copy = copy.deepcopy(model)
  model.run_simulation(10)
  model_copy.run_simulation(15)
  hh_data, fm_data, md_data, _ = model.datacollector.get_data()
  print(md_data[['average_stock', 'unemployment_rate', 'average_income', 'average_price', 
                'trade_volume', 'no_worker_found', 'no_dealer_found', 'worker_fired', ]].head(100))

  hh_data, fm_data, md_data, _ = model_copy.datacollector.get_data()
  print(md_data[['average_stock', 'unemployment_rate', 'average_income', 'average_price', 
                'trade_volume', 'no_worker_found', 'no_dealer_found', 'worker_fired', ]].head(100))
if __name__ == "__main__":
    main()

