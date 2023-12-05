
from ABM import Sugarscepe
import numpy as np
import random 
np.random.seed(0) 
random.seed(0)

#Questions
# why different money after one step
def main():
  steps = 80
  model = Sugarscepe()
  model.run_simulation(steps)
  

  print("_______________________________________")


  hh_data, fm_data, md_data, _ = model.datacollector.get_data()
  print(md_data[['average_stock', 'unemployment_rate', 'average_income', 'average_price', 
                'trade_volume', 'no_worker_found', 'no_dealer_found', 'worker_fired', ]].head(100))
#
if __name__ == "__main__":
    main()

