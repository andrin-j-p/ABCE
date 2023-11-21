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

itr = pd.read_stata('../data/raw_data/GE_HH-ENT_Baseline_Combined.dta', iterator=True)
dct = itr.variable_labels()
f = open("var_des_GE_HH-ENT_Baseline_Combined.txt", "w")
f.write("{\n")
for k in dct.keys():
    f.write("'{}':'{}'\n".format(k, dct[k]))
f.write("}")
f.close()

# %%
df_hh, _, _, df_mk= read_data.create_agent_data()
print(df_mk['location'].head())
#%%
# Define independent (predictor) variables and the dependent (response) variable
X = df_hh[["own_land_acres","h1_11_landvalue","h1_2_agtools", "h1_1_livestock", "h1_12_loans"]]
y = df_hh["p3_totincome"]

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

def plot_dist(data, title):
  az.style.use("arviz-doc")

  fig, ax = plt.subplots()
  az.plot_dist(data, ax=ax, label="Observed Data", rug = True, quantiles=[0.05, 0.5, 0.95], rug_kwargs={'space':0.1})
  # Add labels, title, legend, etc.
  ax.set_ylabel("Density")
  ax.set_xlabel("Value")
  ax.set_title(f"Kernel Density {title}")
  ax.legend()

  # Show the plot
  plt.show()

data = (np.random.lognormal(3.3, 0.5, size=1000) + 1) 
plot_dist(data, 'test')
# %%
list1 = [1,2,3]
list2 = [4,5,6]
l = list2 + list1

print(l)
# %%
import matplotlib.pyplot as plt
import networkx as nx
from netgraph import Graph

# Create a modular graph (dummy data)
partition_sizes = [10, 20, 30, 40]
g = nx.random_partition_graph(partition_sizes, 0.5, 0.1)


# Create a dictionary mapping nodes to their community. 
# This information is used position nodes according to their community 
# when using the `community` node layout in netgraph.
node_to_community = dict()
node = 0
for community_id, size in enumerate(partition_sizes):
    for _ in range(size):
        node_to_community[node] = community_id
        node += 1

# Color nodes according to their community.
community_to_color = {
    0 : 'tab:blue',
    1 : 'tab:orange',
    2 : 'tab:green',
    3 : 'tab:red',
}
node_color = {node: community_to_color[community_id] \
              for node, community_id in node_to_community.items()}


fig, ax = plt.subplots()
Graph(g,
      node_color=node_color, # indicates the community each belongs to  
      node_edge_width=0,     # no black border around nodes 
      edge_width=0.1,        # use thin edges, as they carry no information in this visualisation
      edge_alpha=0.5,        # low edge alpha values accentuates bundles as they appear darker than single edges
      node_layout='community', node_layout_kwargs=dict(node_to_community=node_to_community),
      ax=ax,
)
plt.show()
# %%
fig= plt.figure()
ax= fig.add_subplot(111)
ax.plot(range(10), [i**2 for i in range(10)])
ax.grid(True)
plotly_fig = mpl_to_plotly(fig)
