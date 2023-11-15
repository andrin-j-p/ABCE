#%%
import pandas as pd
import read_data
import numpy as np
import networkx as nx
import statsmodels.api as sm
from read_data import read_dataframe
import arviz as az
import matplotlib.pyplot as plt


# set display options. Not imperative for exectution
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', None)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.width', 10000)
#%%

itr = pd.read_stata('../data/var_description/GE_Enterprise_ECMA.dta', iterator=True)
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
df_revenue = read_dataframe("../data/var_description/GE_HH-BL_income_revenue.dta", "df")
print(len(df_revenue.loc[df_revenue["selfemp"] == 1.0]))
print(len(df_revenue['selfemp']))



# %%
df_hh, df_fm, df_md, df_td = read_data.create_agent_data()
print(df_hh['p3_totincome_PPP'] -df_hh['p2_consumption_PPP'])

# %%
az.style.use("arviz-doc")

# Generate two example NumPy vectors
data1 = np.random.normal(loc=0, scale=1, size=1000)
data2 = np.random.normal(loc=2, scale=1.5, size=1000)

print(data1)

# Plot the densities on the same figure
fig, ax = plt.subplots()
az.plot_dist(data1, ax=ax, label="Data 1", rug = True, rug_kwargs={'space':0.5})
az.plot_dist(data2, ax=ax, label="Data 2", rug=True, )

# Add labels, title, legend, etc.
ax.set_xlabel("Value")
ax.set_ylabel("Density")
ax.set_title("Kernel Density Plots for Two Vectors")
ax.legend()

# Show the plot
plt.show()

# %%
import torch
a = torch.rand((8,4))
b = a.unsqueeze(1)
print(a.shape)
print(b.shape)

# %%
import numpy as np
a = np.array([[1,2,3]])
a = a.flatten()
print(a.shape)
# %%
a = [[1,2,3],
     [4,5,6]]

b = np.array(a)

result_array = np.concatenate([b] * 2, axis=0)

print(result_array)
# %%
