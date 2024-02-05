#%%
import plotly.graph_objects as go
import numpy as np
import networkx as nx
import ABM
import math

def create_circle_coordinates(N, radius):
    """
    Type: Helper function 
    Description: Return N coordinates evenly dispersed around a circle
    Used in: visualize.py create_subgraph 
    """
    assert N > 1

    # calculate angle between two elements
    angle_increment = 2 * math.pi / N
    coordinates = []

    # create N coordinates around the circle
    for i in range(N):
        angle = i * angle_increment
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        coordinates.append((x, y))

    return coordinates


def create_subgraph_G1(G, in_village_nodes):
    """
    Type: Helper function 
    Description: Creates a subgraph of G including all the nodes connected to the list of specified nodes
    Used in: visualize.py callback
    """
    # Iterate through the specified nodes and collect their neighbors
    out_village_nodes = set()
    in_village_coords = create_circle_coordinates(len(in_village_nodes), 10)
    for i, n in enumerate(in_village_nodes):
        G.nodes[n]['pos'] = in_village_coords[i]
        # add neighbouring nodes that are not in the same village to out_village_nodes
        out_village_nodes.update([node for node in G.neighbors(n) if node not in in_village_nodes])

    # create coordinates for the outer village nodes
    out_village_coords = create_circle_coordinates(len(out_village_nodes), 25)
    for i, n in enumerate(list(out_village_nodes)):
        G.nodes[n]['pos'] = out_village_coords[i]

    subgraph = G.subgraph(set(in_village_nodes) | out_village_nodes)
    node_adjacancies = []
    node_text = []
    for n, adjacencies in subgraph.adjacency():
        node_adjacancies.append(len(adjacencies))
        node_text.append('# of connections: '+ str(len(adjacencies)))

    return subgraph, node_adjacancies, node_text

def create_subgraph_G2(G, in_village_nodes):
    """
    Type:        Helper function 
    Description: Creates a subgraph of G including all the nodes connected to the list of specified nodes
    Used in:     visualize.py callback
    """
    # Iterate over all nodes in the village and collect their neighbors

    node_adjacancies = []
    node_text = []
    # Add an edge for each within village connection 
    for n, adjacencies in G.adjacency():
      if G.nodes[n]['village'] == village:
        node_adjacancies.append(len(adjacencies))
        node_text.append('# of connections: '+ str(len(adjacencies)))

    return G.subgraph(in_village_nodes), node_adjacancies, node_text

def create_G1(G, node_adjacancies, node_text):
    pos = {node: attributes['pos'] for node, attributes in G.nodes(data=True)}

    trace_nodes = go.Scatter(
        x=[pos[node][0] for node in G.nodes()],
        y=[pos[node][1] for node in G.nodes()],
        mode='markers',
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
        
        
    trace_nodes.marker.color = node_adjacancies
    trace_nodes.text = node_text


    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
    
    trace_edges = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none', #distance btw. agents
        mode='lines')
    

    # create the graph figure
    fig = go.Figure(data=[trace_edges, trace_nodes],
             layout=go.Layout(
                width=1000,
                height=1000,
                title='Trade Network<br>',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )

    # Show the plo
    fig.show()


def create_G2(G, node_adjacencies, node_text):
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
for agent in model.all_agents:  
    for firm in agent.best_dealers:
        G.add_edge(agent.unique_id, firm.owner.unique_id)

# create subgraph containing only nodes and edges in the village selected
village = 'v_2'
in_village_nodes = [agent.unique_id for agent in agents_lst if agent.village.unique_id == village]

G1, node_adjacencies, node_text = create_subgraph_G1(G, in_village_nodes)
create_G1(G1, node_adjacencies, node_text)




# %%
