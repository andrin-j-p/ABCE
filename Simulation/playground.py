#%%
import pandas as pd
import read_data
import numpy as np
import networkx as nx
import statsmodels.api as sm
from read_data import read_dataframe
#%%
# set display options. Not imperative for exectution
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', None)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.width', 10000)


itr = pd.read_stata('../data/GE_Enterprise_ECMA.dta', iterator=True)
dct = itr.variable_labels()
f = open("var_des_ent_ECMA.txt", "w")
f.write("{\n")
for k in dct.keys():
    f.write("'{}':'{}'\n".format(k, dct[k]))
f.write("}")
f.close()

# %%
df, _, _= read_data.create_agent_data()

# Define independent (predictor) variables and the dependent (response) variable
X = df[["own_land_acres","h1_11_landvalue","h1_2_agtools", "h1_1_livestock", "h1_12_loans"]]
y = df["p3_totincome"]

# Add a constant term to the independent variables (intercept)
X = sm.add_constant(X)

# Fit the multiple regression model
model = sm.OLS(y, X).fit()

# Print the summary statistics
print(model.summary())

# %%
df_revenue = read_dataframe("../data/GE_HH-BL_income_revenue.dta", "df")
print(len(df_revenue.loc[df_revenue["selfemp"] == 1.0]))
print(len(df_revenue['selfemp']))





# %%
a = np.random.choice([], size=1)[0]
print(a)
# %%
