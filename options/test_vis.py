#%%
import plotly.graph_objects as go
import numpy as np
import networkx as nx
import ABM
import math

def generate_coordinates_on_sphere(num_villages, radius):
    coordinates = []
    for i in range(num_villages):
        x = np.random.uniform(-10, 10)
        y = np.random.uniform(-10, 10)
        z = np.random.uniform(-10, 10)
        coordinates.append((x, y, z))
    return coordinates


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

# create a node for each agent in the model 
all_nodes = [(agent.unique_id, {'village': agent.village, 'pos': (0,0,0)}) for agent in agents_lst]
G.add_nodes_from(all_nodes)

# for all agents, add an edges for each of its trading partners
for row in df_hh[df_hh['step'] == steps-1].itertuples():
    for dealer in row.best_dealers:
        G.add_edge(row.unique_id, dealer.owner.unique_id)

# create subgraph
agents_slctd = [agent.unique_id for agent in agents_lst if agent.village.unique_id == 601010103008]
G = create_subgraph(G, agents_slctd)


villages  = set([G.nodes[n]['village'] for n in G.nodes])
    
coordinates = generate_coordinates_on_sphere(len(villages), 10)
vl_dct = {}

for i, village in enumerate(villages):
    vl_dct[village] = coordinates[i]

for n in G.nodes:
    center = vl_dct[G.nodes[n]['village']] 
    G.nodes[n]['pos'] = tuple([center[i] + np.random.uniform(-1, 1) for i in range(3)])

pos = {node: attributes['pos'] for node, attributes in G.nodes(data=True)}

# Create a 3D scatter plot
trace_nodes = go.Scatter3d(
    x=[pos[node][0] for node in G.nodes()],
    y=[pos[node][1] for node in G.nodes()],
    z=[pos[node][2] for node in G.nodes()],
    mode='markers',
    marker=dict(size=8, color='blue'),
    text=list(G.nodes())
)

edge_x = []
edge_y = []
edge_z = []

for edge in G.edges():
    x0, y0, z0 = pos[edge[0]]
    x1, y1, z1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])
    edge_z.extend([z0, z1, None])

# Create a 3D scatter plot for edges
trace_edges = go.Scatter3d(
    x=edge_x,
    y=edge_y,
    z=edge_z,
    mode='lines',
    line=dict(color='black', width=2)
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
    scene=dict(
        xaxis=dict(axis),
        yaxis=dict(axis),
        zaxis=dict(axis)
    )
)

# Create the figure
fig = go.Figure(data=[trace_nodes, trace_edges], layout=layout)

# Show the plo
fig.show()

