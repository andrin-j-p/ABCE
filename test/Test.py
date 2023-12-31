import unittest
import sys
import os
import pandas as pd

# Get the current directory (where the test file is located)
current_directory = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory (containing the 'Sugarsim' folder) to the module search path
parent_directory = os.path.abspath(os.path.join(current_directory, '..'))
sys.path.append(parent_directory)

# Add the Simulation directory (i.e. the 'Simulationo' folder) to the module search path
Simulation_directory = os.path.abspath(os.path.join(parent_directory, 'Simulation'))
sys.path.append(Simulation_directory)

####################
# TEST READ_DATA.PY 
####################

# Import the functionalities from 'read_data.py'
from Simulation import read_data

# load raw datasets
df_hh_raw = read_data.read_dataframe('GE_HHLevel_ECMA.dta', 'df')
df_fm_raw = read_data.read_dataframe('GE_Enterprise_ECMA.dta', 'df')
df_mk_raw = read_data.read_dataframe('GE_MarketData_Panel_ProductLevel_ECMA.dta', 'df')
df_vl_raw = read_data.read_dataframe("GE_VillageLevel_ECMA.dta", "df")
df_nrst_mk = read_data.read_dataframe("Village_NearestMkt_PUBLIC.dta", "df")

class TestReadData(unittest.TestCase):
    """
    Name:        TestReadDataFrame
    Description: Unittesting for functions in read_data.py
    """
    def test_columns_exist(self):
        """
        Test if the DataFrame contains certain columns
        """
        # List of expected columns
        expected_columns = ['treat']

        # Check if all expected columns exist in the DataFrame
        for col in expected_columns:
            self.assertIn(col, df_hh_raw.columns)

    def test_no_hh_twice(self):
        pass

    def test_hhs_same_in_all_dfs(self):
        pass
    

####################
# TEST SIMULATION.PY 
####################

# Import the functionalities from the 'Sugarsim.py' folder
from Simulation import ABM

class Testsim(ABM.Sugarscepe):
  """
  Type:         Child Class Sugarscape
  Description:  Mock model instance for testing. 
                Extends Sugarscape class
  """
  def __init__(self):
    super().__init__()
    self.fm_dct = {firm.unique_id: firm for firm in self.all_firms}
    self.hh_dct = {hh.unique_id: hh for hh in self.all_agents}

steps = 100
model = Testsim()
model.run_simulation(steps=steps)
df_hh_sim, df_fm_sim, df_md_sim, df_td_sim = model.datacollector.get_data()
    

class TestABM(unittest.TestCase):
    """
    Name:        TestABM
    Description: Unittesting for functions in ABM.py
    """
### Model 
 
    def test_village_instantiation(self):
        """
        test if there are markets without vendors
        """
        for mk in model.all_markets:
            self.assertGreater(len(mk.vendors), 0)

    def test_market_village_mapping(self):
        """
        test if markets are mapped to the correct village
        """
        for vl in model.all_villages:
            val_sim = vl.market.unique_id
            val_exp = f"m_{int(df_nrst_mk.loc[df_nrst_mk['village_code']==vl.unique_id]['market_id'].iloc[0])}"
            self.assertEqual(val_sim, val_exp)

### Firm

    def test_all_firms_on_market(self):
        """
        test if all vendors have a firm
        """
        all_vendors = []
        for mk in model.all_markets:
            for vendor in mk.vendors:
                all_vendors.append(vendor)
        all_firms = model.all_firms

        self.assertEqual(set(all_vendors), set(all_firms))

    def test_firm_owner_in_hh(self):
        """
        test if owner hhid_key is an hhid_key existing in hh dataset
        """
        i = 0
        owner_lst = [firm.owner for firm in model.all_firms]
        for owner in owner_lst:
            if owner not in [agent for agent in model.all_agents]:
                i+=1
        self.assertEqual(i, 0)

    def test_all_firms_have_owner(self):
        """
        test Firm __init__ 
        """
        for firm in model.all_firms:
            self.assertNotEqual(firm.owner, None)
            
    def test_same_amount_of_firms_and_owners(self):
        """
        test if all firms have an owner
        -> aggregate test
        """
        owner_lst = [owner for owner in model.all_agents if owner.firm != None ]
        self.assertEqual(len(owner_lst), len(model.all_firms))

    def test_owners_are_employed_by_own_firm(self):
        """
        test if firm owners are employed at their own firm
        """
        for hh in model.all_agents:
            if hh.firm != None:
                self.assertEqual(hh.employer, hh.firm)

# Agents

    def test_hh_variables(self):
        """
        test1: hh money needs to be at least zero
        test2: hh demand cannot exceed money
        test3: hh income is zero or positive
        test4: hh income same as productivity
        """
        for hh in model.all_agents:
            self.assertGreaterEqual(hh.money, 0)
            self.assertGreaterEqual(hh.money, hh.demand)
            self.assertGreaterEqual(hh.income, 0)

            if hh.firm == None and hh.employer != None:
                self.assertEqual(hh.income, hh.productivity)

### Trade 

    def test_price_larger_min_price(self):
        """
        test1 if trades happen at a price larger than min price
        test2 if there are trades with negative volume = price * demand
        """
        for transaction in df_td_sim.itertuples():
            transaction_price = transaction.price
            price_min = model.hh_dct[transaction.parties[1]].firm.marginal_cost
            self.assertLess(price_min, transaction_price, f"Price in transaction {transaction} is lower than minimum")
            self.assertGreater(transaction.volume, 0)

    def test_buyer_dealer_on_same_market(self):
        """
        test if for each trade, buyer and vendor have the same market affilitaion
        """
        for transaction in df_td_sim.itertuples():
            buyer_mk = model.hh_dct[transaction.parties[0]].village.market
            vendor_mk = model.hh_dct[transaction.parties[1]].firm.market
            self.assertEqual(buyer_mk, vendor_mk)

    def test_best_dealers(self):
        """
        test1: list of preferred dealers does not contain duplicates
        test2: all agents in the same village should either have 3 or 0 best dealers

        """
        for hh in model.all_agents:
            # test1
            self.assertEqual(len(set(hh.best_dealers)), len(hh.best_dealers))

            l = len(hh.best_dealers)
            if l == 0:
                for person in hh.village.population:
                    self.assertEqual(len(person.best_dealers), 0)      

### Intervention 

    def test_treated_agents(self):
        # test if after token money is at least 150
        # test if after first intervention money is at least 840 
        # test if after second intervention money is at least 840 
        pass

### Semantics

    def test_prices_convergance(self):
        """
        rationale: average transaction price is expected to decrease
        """
        initial_price = df_md_sim[df_md_sim['step'] == 1]['average_price'].mean()
        final_price = df_md_sim[df_md_sim['step'] == steps-1]['average_price'].mean()
        self.assertLess(final_price, initial_price)


    def test_sales_converge(self):
        pass

    def test_unemployment_rate_in_range(self):
        pass

    def test_demand_satisfied_in_range(self):
        pass




#####################
# TEST CALIBRATION.PY 
#####################

#from Simulation import calibration
"""
@unittest.skip("Does not hold in data")
class TestCalibration(unittest.TestCase):
    
    Name:        TestABM
    Description: Unittesting for functions in ABM.py
    
    
    def test_sobol_sampler(self):
        data = pd.DataFrame({
        'Type': ['Firm', 'Firm', 'Firm'], 
        'Name': ['alpha', 'beta', 'gamma'],
        'Bounds': [(1,10), (-1,5), (0.1, 0.5)]})

        samples = calibration.create_sample_parameters(data, m=4)
        self.assertEqual(len(samples[0]), 3)
        para_1 = [s[0] for s in samples]
        para_2 = [s[1] for s in samples]
        para_3 = [s[2] for s in samples]
        self.assertLessEqual(max(para_1), 10)
        self.assertLessEqual(max(para_2), 5)
        self.assertLessEqual(max(para_3), 0.5)
        self.assertGreaterEqual(min(para_1),1 )
        self.assertGreaterEqual(min(para_2), -1)
        self.assertGreaterEqual(min(para_3), 0.1)
    
    def test_set_parameter(self):
        sample = {'nu': 5, 'theta': 0, 'phi_l':0.4}
        model = calibration.Model()
        model.set_parameters(sample)
        for firm in model.all_firms:
            self.assertEqual(firm.nu, 5)
            self.assertEqual(firm.theta, 0)
            self.assertEqual(firm.phi_l, 0.4)

    def test_mse(self):
        pass

"""
if __name__ == '__main__':

    unittest.main()

