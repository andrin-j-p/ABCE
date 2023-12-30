import random
import numpy as np
import math

def create_list_partition(input_list):
    """
    Type:         Helper function 
    Description:  Splits an input list into exponentially decaying sublists
    """
    output_list = []
    n = math.floor(len(input_list))

    # while the list is larger than 10 split the input list into exponentially decaying sublist
    while n > 10:
        n = math.floor(len(input_list)/3)
        output_list.append(input_list[:n])
        input_list = input_list[n:]

    # split the remaining elements into equally sized sublists and append them
    remainders = np.array_split(input_list, len(input_list)/3)
    output_list.extend([sublist.tolist() for sublist in remainders])
    return output_list


class Intervention_handler():
  """
  Type:        Helper Calss
  Description: Performs the Unconditional Cash Transfer (UCT)
  """
  def __init__(self, model):
    self.model = model
    self.control = False
    self.treated_agents = []
    self.UCT_1 = []
    self.UCT_2 = []
    self.UCT_3 = []

  def UCT(self, current_step):
    """
    Type:        Method
    Description: Coordinates the time of the UCTs
    """    
    intervention_step = 360
    # Assign treatment status at step 350
    if current_step == intervention_step:
      # assign treatment status 
      self.assign_treatement_status()

    if self.control == True:
      return

    # Start weekly rollout of transfers at step 350 
    if current_step >= intervention_step and current_step%7==0:

      # Start rollout of the token 15 days after week s after treatment assignment
      if current_step >= intervention_step and len(self.UCT_1) > 0:

        # Get batch of agents receiving the token in the current week
        agents = self.UCT_1.pop(0)
        self.UCT_2.append(agents)

        # distribute the token (USD 150 PPP)
        self.intervention(80, agents) 

      # Start first UCT rollout 2 months after token
      if current_step >= intervention_step + 60 and len(self.UCT_2) > 0:

        # Get the batch of agents to receiving UCT 1 in the current week
        agents = self.UCT_2.pop(0)
        self.UCT_3.append(agents)

        # distribute first handout (USD 860 PPP)
        self.intervention(460, agents)
        
      # Start second UCT rollout 8 months after token
      if current_step >= intervention_step + 240 and len(self.UCT_3) > 0:

        # Get the batch of agents to receiving UCT 2 in the current week
        agents = self.UCT_3.pop(0)

        # distribute first handout (USD 860 PPP)
        self.intervention(460, agents)


  def assign_treatement_status(self):
    """
    Type:        Method
    Description: Assigns saturation and treatment status on market, village and agnet level
    """

### Level 1 randomization

    # assign high saturation status to 30 random markets (without replacment)
    high_sat_mk = random.sample(self.model.all_markets, k=34)
    for mk in high_sat_mk:
      setattr(mk, 'saturation', 1)

### Level 2 randomizaton

    # assign treatment status to 1/3 of villages in low saturation mks 
    # assign treatment status to 2/3 of villages in high saturation mks 
    treatment_villages  = []
    for mk in self.model.all_markets:
      # choose treatment villages fraction depending on market saturation status
      treat_frac = 2/3 if mk.saturation == 1 else 1/3
      treatment_villages.extend(random.sample(mk.villages, k = int(len(mk.villages) * treat_frac)))

    # assign treatment status to the selected villages
    for vl in treatment_villages:
      setattr(vl, 'treated', 1)

### Level 3 randomization 

    # for each village identify the 30 poorest households
    for vl in treatment_villages:
      sorted_population = sorted(vl.population, key=lambda x: x.income )
      eligible_hh = sorted_population[:33]
      self.treated_agents.extend(eligible_hh)
    
    # assign treatment status to the selected agents
    for hh in self.treated_agents:
      setattr(hh, 'treated', 1)

    # declare the stack for 3 phase treatment rollout
    self.UCT_1 = create_list_partition(self.treated_agents)

    print(f"# treated hhs: {len(self.treated_agents)}, # treated vls: {len(treatment_villages)}")


  def intervention(self, amount, agents):
    """
    Type:        Method
    Description: Distributes the UCT to the eligible agents
    """
    for agent in agents:
      agent.money += amount