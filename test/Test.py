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

class Testsim(ABM.Model):
  """
  Type:         Child Class Sugarscape
  Description:  Mock model instance for testing. 
                Extends Sugarscape class
  """
  def __init__(self):
    super().__init__()
    self.fm_dct = {firm.unique_id: firm for firm in self.all_firms}
    self.hh_dct = {hh.unique_id: hh for hh in self.all_agents}

steps = 1000
model = Testsim()
model.run_simulation(steps=steps)
df_hh_sim, df_fm_sim, df_md_sim, df_td_sim = model.datacollector.get_data()
    
class TestABM(unittest.TestCase):
    """
    Name:        TestABM
    Description: Unittesting for functions in ABM.py
    """
### Model 

    def test_market_instantiation(self):
        """
        test1: no markets without vendors
        test2: no marktes without conntection to villages
        test3: market village association 
        test4: all vendors have a firm
        """
        for mk in model.all_markets:
            self.assertGreater(len(mk.vendors), 0)
            self.assertGreater(len(mk.villages), 0)
            for vl in mk.villages:
                self.assertEqual(vl.market, mk)

            for vendor in mk.vendors:
                self.assertIsNotNone(vendor)

    def test_village_instantiation(self):
        """
        test1: villages have a population of at least 50 people
        test2: household village association
        """
        for vl in model.all_villages:
            self.assertGreater(len(vl.population), 50)
            for hh in vl.population:
                self.assertEqual(hh.village, vl)

    def test_agent_instantiation(self):
        """
        test1: household is part of the village populaiton
        test2: firm of owner exist (if applicable)
        test3: owner is employed by own firm 
        """
        for hh in model.all_agents:
            self.assertIn(hh, hh.village.population)
            if hh.firm != None:
                self.assertIn(hh.firm, model.all_firms)
                self.assertEqual(hh.firm.owner, hh)
            
                
    def test_firm_instantiation(self):
        """
        test1: firm has owner
        test2: owner exists
        test3: firm owner association 
        test4: firm village is same as owner village
        test5: market firm operates on exists
        test6: firm owner is in market vendor list
        """
        for fm in model.all_firms:
            owner = fm.owner
            self.assertIsNotNone(owner)
            self.assertIn(owner, model.all_agents)
            self.assertEqual(owner.firm, fm)
            self.assertEqual(owner.village, fm.village)
            self.assertIn(fm.market, model.all_markets)
            self.assertIn(fm, fm.market.vendors)

        self.assertEqual(len(model.all_firms), len([1 for hh in model.all_agents if hh.firm != None]))


### Firm
    def test_attributes(self):
        """
        test1: stock weakly positive
        test2: revenue is weakly positve
        test3: money not negative
        test4: firm has owner
        test5: owner is in all_agents
        test6: test no emploee twice
        """
        for fm in model.all_firms:
            self.assertGreaterEqual(fm.stock, 0)
            self.assertGreaterEqual(fm.money, -1)
            self.assertGreaterEqual(fm.revenue, 0)
            self.assertIsNotNone(fm.owner)
            self.assertIn(fm.owner, model.all_agents)
            self.assertIn(fm.village, model.all_villages)
            self.assertEqual(len(fm.employees), len(set(fm.employees)))

    def test_set_price(self):
        """
        test1: price greater or equal marginal cost
        test2: price smaller max price
        """
        for fm in model.all_firms:
            price = fm.price
            self.assertGreaterEqual(price, fm.marginal_cost)
            self.assertLessEqual(price, fm.max_price)

    def test_produce(self):
        """
        test1: production strictly positive
        """
        for fm in model.all_firms:
            self.assertGreater(fm.output, 0)

    def test_set_labor(self):
        """
        """
        pass
            
    def test_distribute_profit(self):
        """
        test1:
        """
        pass

    def test_firm_step(self):
        """
        test if firm owners are employed at their own firm
        """
        pass

# Agents

    def test_hh_attributes(self):
        """
        test1: hh money weakly positive
        test2: hh income is zero or positive
        test3: hh no best dealer twice
        test4: hh income same as productivity
        """
        for hh in model.all_agents:
            self.assertGreaterEqual(hh.money, -1, f"money {hh.money}") # -1 due to floating point errors
            self.assertGreaterEqual(hh.income, 0, f"income {hh.income}")
            self.assertEqual(len(hh.best_dealers), len(set(hh.best_dealers)))
            self.assertGreaterEqual(hh.demand, 0)
            if hh.firm == None and hh.employer != None:
                self.assertEqual(hh.income, hh.productivity, f"empyler {hh.employer} firm {hh.firm}")

### Trade 

    def test_transaction_price(self):
        """
        test1: if trades happen at a price larger than min price
        test2: volume strictly positive volume = price * demand
        """
        for transaction in df_td_sim.itertuples():
            transaction_price = transaction.price
            price_min = model.hh_dct[transaction.parties[1]].firm.marginal_cost
            self.assertLess(price_min, transaction_price)
            self.assertGreater(transaction.volume, 0)

    def test_transaction_market(self):
        """
        test1 if for each trade, buyer and vendor have the same market affilitaion
        """
        for transaction in df_td_sim.itertuples():
            buyer_mk = model.hh_dct[transaction.parties[0]].village.market
            vendor_mk = model.hh_dct[transaction.parties[1]].firm.market
            self.assertEqual(buyer_mk, vendor_mk)

    def test_best_dealers(self):
        """
        test1: list of prefered dealers does not contain duplicates
        test2: agents in the same village have the same number of best dealers
        """
        for hh in model.all_agents:
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

    def test_treated_villages(self):
        """
        test 1: vl on low saturation mk have low sat status
        test 2: vl on high saturaiton mk have high saturation status
        """
        for vl in model.all_villages:
            self.assertEqual(vl.saturation, vl.market.saturation)

    def test_treated_market(self):
        """
        test1: fraction treatment villages high saturation market
        test2: fraction treatment villages low saturation market
        test3: 33 markets with high saturation
        test4: 28 markets wiht low saturation 
        """
        high_sat_count = 0
        low_sat_count =  0
        for mk in model.all_markets:

            if mk.saturation == 1:
                high_sat_count += 1

                treated_vl = 0
                for vl in mk.villages:
                    if vl.treated == 1:
                        treated_vl += 1

                self.assertEqual(int(len(mk.villages)*(2/3)), treated_vl)
            
            if mk.saturation == 0:
                low_sat_count += 1

                control_vl = 0
                for vl in mk.villages:
                    if vl.treated == 1:
                        control_vl += 1
                
                self.assertEqual(int(len(mk.villages)*(1/3)), control_vl)

        
        self.assertEqual(high_sat_count, 33)
        self.assertEqual(low_sat_count, 28)

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

