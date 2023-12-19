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

data = (np.random.lognormal(3.4, 0.7, size=1000) + 1) 
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
#%%
import plotly.graph_objects as go
import numpy as np
import networkx as nx
import ABM
import math

def generate_coordinates_on_sphere(num_villages, radius):
    coordinates = []
    for i in range(num_villages):
        theta = 2 * math.pi * i / num_villages
        phi = math.pi * (i + 1) / (num_villages + 1)  # Distribute vertically
        x = radius * math.sin(phi) * math.cos(theta)
        y = radius * math.sin(phi) * math.sin(theta)
        z = radius * math.cos(phi)
        coordinates.append((x, y, z))
    return coordinates

def create_village_dictionary_on_sphere(num_villages, radius):

    for i in range(num_villages):
        village_name = f"Village_{i + 1}"
        villages[village_name] = coordinates[i]

    return villages

def create_subgraph(G, in_village_nodes):
    """
    Type: Helper function 
    Description: Creates a subgraph of G including all the nodes connected to the list of specified nodes
    Used in: visualize.py callback
    """
    # Iterate through the specified nodes and collect their neighbors
    subgraph_nodes = set(in_village_nodes)
    for n in in_village_nodes:
        subgraph_nodes.update([node for node in G.neighbors(n) if node not in in_village_nodes])        

    return G.subgraph(subgraph_nodes)


steps = 10
model = ABM.Sugarscepe()
model.run_simulation(steps)
df_hh, df_fm, df_md, _= model.datacollector.get_data()

G = nx.Graph()

# create a list of all agents in the model
agents_lst = list(model.all_agents)
villages  = set([agent.village for agent in agents_lst])
coordinates = generate_coordinates_on_sphere(len(villages), 10)
vl_dct = {}

for i, village in enumerate(villages):
    vl_dct[village] = coordinates[i]

print(vl_dct)

# create a node for each agent in the model 
all_nodes = [(agent.unique_id, {'village': agent.village, 'pos': (np.random.uniform(0,1),np.random.uniform(0,1),np.random.uniform(0,1))}) for agent in agents_lst]
G.add_nodes_from(all_nodes)

# for all agents, add an edges for each of its trading partners
for row in df_hh[df_hh['step'] == steps-1].itertuples():
    for dealer in row.best_dealers:
        G.add_edge(row.unique_id, dealer.owner.unique_id)


agents_slctd = [agent.unique_id for agent in agents_lst if agent.village.unique_id == 601010103008]
SG = create_subgraph(G, agents_slctd)

pos = {node: attributes['pos'] for node, attributes in G.nodes(data=True)}

# Create a 3D scatter plot
trace = go.Scatter3d(
    x=[pos[node][0] for node in G.nodes()],
    y=[pos[node][1] for node in G.nodes()],
    z=[pos[node][2] for node in G.nodes()],
    mode='markers',
    marker=dict(size=8, color='blue'),
    text=list(G.nodes())
)


axis=dict(showbackground=False,
          showline=False,
          zeroline=False,
          showgrid=False,
          showticklabels=False,
)

# Create layout
layout = go.Layout(
    title="3D Plotly Graph",
    showlegend=False,
    scene=dict(
        xaxis=dict(axis),
        yaxis=dict(axis),
        zaxis=dict(axis)
    )
)

# Create the figure
fig = go.Figure(data=[trace], layout=layout)

# Show the plo
fig.show()




#%%
import numpy as np
import math
import matplotlib.pyplot as plt

def create_list_partition(input_list):
    output_list = []
    n = math.floor(len(input_list))

    # while the list is larger than 10 split the input list into exponentially decaying sublist
    while n > 10:
        n = math.floor(len(input_list)/7)
        output_list.append(input_list[:n])
        input_list = input_list[n:]

    # split the remaining elements into equally sized sublists and append them
    remainders = np.array_split(input_list, len(input_list)/8)
    output_list.extend([sublist.tolist() for sublist in remainders])
    return output_list


input_list = np.arange(10500)
output_list = create_list_partition(input_list)

data = [len(sublist) for sublist in output_list]
print(data)
# %%
num_bars = len(data) // 4 + (len(data) % 4 > 0)  # Ceiling division

# Calculate the averages for each group of four entries
averages = [np.mean(data[i*4:(i+1)*4]) for i in range(num_bars - 1)]

# For the last bar, calculate the average of the remaining entries
last_average = np.mean(data[(num_bars - 1) * 4:])

# Append the last average to the averages list
averages.append(last_average)

# Create a bar plot
plt.bar(range(1, num_bars + 1), averages)
plt.show()
plt.bar(range(len(data)),data)
plt.show()
# %%

