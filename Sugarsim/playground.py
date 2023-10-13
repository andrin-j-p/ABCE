#%%
import pandas as pd
from read_data import read_dataframe
import networkx as nx

# set display options. Not imperative for exectution
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', None)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.width', 10000)


itr = pd.read_stata('../data/GE_HH-EL_hhexpenditure.dta', iterator=True)
dct = itr.variable_labels()
f = open("variable_description_expenditure_EL.txt", "w")
f.write("{\n")
for k in dct.keys():
    f.write("'{}':'{}'\n".format(k, dct[k]))
f.write("}")
f.close()
# %%
df = read_dataframe('GE_HH-EL_hhexpenditure.dta', 'df')
df[['s1_hhid_key','s12_q1_cerealsamt_12mth']].head()

