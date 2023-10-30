#%%
import matplotlib.pyplot as plt
import ABM
from  read_data import create_agent_data
import pandas as pd

df_hh, df_fm, df_vl, df_mk = create_agent_data()


#%%
plt.hist(df_fm['prof_year'], bins=70, color='blue', edgecolor='black')
plt.xlabel('Profit')
plt.ylabel('Frequency')
plt.title('Profit Distribution Histogram BL')
plt.grid(True)
plt.show()


#%%
model = ABM.Sugarscepe()
for hh in model.all_agents:
  if hh.firm == None and hh.income < 0:
    print(hh)

model.run_simulation(100)
df_hh, df_fm, df_md, df_td = model.datacollector.get_data()

#%%
plt.hist(df_hh[df_hh['step']==0]['income'], bins=70, color='blue', edgecolor='black')
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.title('Income Distribution Histogram Step 0')
plt.grid(True)
#plt.show()

# Histogram of income distribution after 100 epochs
plt.hist(df_hh[df_hh['step']==99]['income'], bins=70, color='blue', edgecolor='black')
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.title('Income Distribution Histogram Step 99')
plt.grid(True)
#plt.show()

plt.hist(df_fm[df_fm['step'] == 99]['money'], bins=70, color='blue', edgecolor='black')
plt.xlabel('Profit')
plt.ylabel('Frequency')
plt.title('Profit Distribution Histogram Step 99')
plt.grid(True)
plt.show()

# %%
