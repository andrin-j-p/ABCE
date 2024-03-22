#%%
import plotly.graph_objects as go
import numpy as np
import networkx as nx
import ABM
import math
import random
np.random.seed(0) 
random.seed(0)

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


def create_subgraph_G1(G_village, G_complete, in_village_nodes):
    """
    Type: Helper function 
    Description: Creates a subgraph of G including all the nodes connected to the list of specified nodes
    Used in: visualize.py callback
    """
    # Iterate through the specified nodes and collect their neighbors
    out_village_nodes = set()
    in_village_coords = create_circle_coordinates(len(in_village_nodes), 10)
    for i, n in enumerate(in_village_nodes):
        G_village.nodes[n]['pos'] = in_village_coords[i]
        # add neighbouring nodes that are not in the same village to out_village_nodes
        out_village_nodes.update([node for node in G_village.neighbors(n) if node not in in_village_nodes])

    # create coordinates for the outer village nodes
    out_village_coords = create_circle_coordinates(len(out_village_nodes), 25)
    for i, n in enumerate(list(out_village_nodes)):
        G_village.nodes[n]['pos'] = out_village_coords[i]

    # Create a subgraph of only the outgoing edges edges of in_village nodes
    subgraph = G_village.subgraph(set(in_village_nodes) | out_village_nodes)

    # Number of total edges are infered from the complete graph
    node_adjacancies = []
    node_text = []
    for n, adjacencies in G_complete.adjacency():
        if subgraph.has_node(n):
            node_adjacancies.append(len(adjacencies))
            node_text.append('# of connections: '+ str(len(adjacencies)))

    return subgraph, node_adjacancies, node_text


def create_G1(G, node_adjacancies, node_text, in_village_nodes):
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
                title='',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))
        
    trace_nodes.marker.color = node_adjacancies
    trace_nodes.text = node_text

    # Separate edges by color
    edge_x_blue = [] # For edges within the same village
    edge_y_blue = []
    edge_x_other = [] # For other edges
    edge_y_other = []

    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        if edge[0] in in_village_nodes and edge[1] in in_village_nodes:
            edge_x_blue.extend([x0, x1, None])
            edge_y_blue.extend([y0, y1, None])
        else:
            edge_x_other.extend([x0, x1, None])
            edge_y_other.extend([y0, y1, None])
    
    trace_edges_blue = go.Scatter(
        x=edge_x_blue, 
        y=edge_y_blue,
        line=dict(width=1, color='blue'), # Blue color for intra-village edges
        hoverinfo='none',
        mode='lines')

    trace_edges_other = go.Scatter(
        x=edge_x_other, 
        y=edge_y_other,
        line=dict(width=0.5, color='#455a64'), # Default color for other edges
        hoverinfo='none',
        mode='lines')
    
    # create the graph figure
    fig = go.Figure(data=[trace_edges_other, trace_edges_blue, trace_nodes],
                layout=go.Layout(
                width=1000,
                height=1000,
                title= dict(text='', x=0.5, font=dict(size=24)),
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )

    # Show the plot
    fig.write_image("../data/illustrations/inter_village.pdf")


    fig.show()


steps = 10
model = ABM.Model()
model.run_simulation(steps)
df_hh, _, = model.datacollector.get_data()
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

G_complete = G.copy()
G_village = G.copy()

# for all agents, add an edges for each of its trading partners
for agent in model.all_agents:  
    if agent.village.unique_id == 'v_2':
        for firm in agent.best_dealers:
            G_village.add_edge(agent.unique_id, firm.owner.unique_id)


# for all agents, add an edges for each of its trading partners
for agent in model.all_agents:  
        for firm in agent.best_dealers:
            G_complete.add_edge(agent.unique_id, firm.owner.unique_id)

# create subgraph containing only nodes and edges in the village selected
village = 'v_2'
in_village_nodes = [agent.unique_id for agent in agents_lst if agent.village.unique_id == village]

G1, node_adjacencies, node_text  = create_subgraph_G1(G_village, G_complete, in_village_nodes)

create_G1(G1, node_adjacencies, node_text, in_village_nodes)



# %%
