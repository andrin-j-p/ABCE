#%%
import plotly.graph_objects as go
import numpy as np
import networkx as nx
import ABM

def create_subgraph_in_village(G, village):
    """
    Type: Helper function 
    Description: Creates a subgraph of G including all the nodes connected to the list of specified nodes
    Used in: visualize.py callback
    """
    # Iterate over all nodes in the village and collect their neighbors
    subgraph_nodes = [agent.unique_id for agent in agents_lst if agent.village.unique_id == village]

    node_adjacancies = []
    node_text = []
    for n, adjacencies in G.adjacency():
      if G.nodes[n]['village'] == village:
        node_adjacancies.append(len(adjacencies))
        node_text.append('# of connections: '+ str(len(adjacencies)))

    return G.subgraph(subgraph_nodes), node_adjacancies, node_text


def create_figure(G, node_adjacencies, node_text):
    pos = {node: attributes['pos'] for node, attributes in G.nodes(data=True)}

    # Create the 3D node traces
    trace_nodes = go.Scatter3d(
        x=[pos[node][0] for node in G.nodes()],
        y=[pos[node][1] for node in G.nodes()],
        z=[pos[node][2] for node in G.nodes()],
        mode='markers',
        text=0,
        hoverinfo='text',
        marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                reversescale=True,
                color=[],
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2))

    trace_nodes.marker.color = node_adjacencies
    trace_nodes.text = node_text

    edge_x = []
    edge_y = []
    edge_z = []

    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])

    # Create the 3D edge traces
    trace_edges = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode='lines',
        hoverinfo='none',
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
        titlefont_size=16,
        showlegend=False,
        hovermode='closest',
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


steps = 10
model = ABM.Sugarscepe()
model.run_simulation(steps)
df_hh, _, _, _= model.datacollector.get_data()
df_hh = df_hh[df_hh['step'] == steps-1]

G = nx.Graph()

# create a list of all agents in the model
agents_lst = model.all_agents

# create a node for each agent in the model 
all_nodes = [(agent.unique_id, {'village': agent.village.unique_id, 
                                'pos': (np.random.uniform(-1,1),np.random.uniform(-1,1),np.random.uniform(-1,1))}) 
                                for agent in agents_lst]

# create a graph objects with nodes for all agents in the simulation 
G.add_nodes_from(all_nodes)

# for all agents, add an edges for each of its trading partners
for row in df_hh.itertuples():   
  for firm in row.best_dealers:
    G.add_edge(row.unique_id, firm.owner.unique_id)


### G1
# create subgraph containing only nodes and edges in the village selected
village = 601010103008

G1, node_adjacencies, node_text = create_subgraph_in_village(G, village)

create_figure(G1, node_adjacencies, node_text)



