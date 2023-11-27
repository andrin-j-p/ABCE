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

def create_subgraph(G, in_village_nodes):
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

    return G.subgraph(set(in_village_nodes) | out_village_nodes)

def create_figure(G, node_adjacencies, node_text):
    # create edges to be displayed (see https://plotly.com/python/network-graphs/)
    pos = {node: attributes['pos'] for node, attributes in G.nodes(data=True)}


    edge_trace1 = go.Scatter( x=[], y=[], mode='lines',
        line=dict(width=0.5, color='blue'), hoverinfo='none',
    )

    edge_trace2 = go.Scatter( x=[], y=[], mode='lines',
        line=dict(width=0.5, color='#888'), hoverinfo='none',
    )

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        n1, n2 = edge
        # Arbitrary distinction between the group.
        # Normally you need to have the indices of the groups of edges        
        if G.nodes[n1]['village'] == 601010103008 and G.nodes[n2]['village'] == 601010103008:
            edge_trace1['x'] += tuple([x0,x1,None])
            edge_trace1['y'] += tuple([y0,y1,None])
        else:
            edge_trace2['x'] += tuple([x0,x1,None])
            edge_trace2['y'] += tuple([y0,y1,None])

    edge_trace = [edge_trace1, edge_trace2]

    # create nodes to be displayed 
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
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

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append('# of connections: '+ str(len(adjacencies[1])))

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    # create the graph figure
    fig = go.Figure(data = edge_trace +[node_trace],
             layout=go.Layout(
                width=1000,
                height=1000,
                title='Trade Network<br>',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=True),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=True),
                ))

    # Show the plot
    fig.show()

steps = 10
model = ABM.Sugarscepe()
model.run_simulation(steps)
df_hh, _, _, _= model.datacollector.get_data()
df_hh = df_hh[df_hh['step'] == steps-1]

G = nx.Graph()

# create a list of all agents in the model
village_list = model.all_villages

# create a node for each agent in the model 
all_nodes = [(village.unique_id, {'pos': (village.pos[0], village.pos[1])}) 
              for village in village_list]

# create a graph objects with nodes for all agents in the simulation 
G.add_nodes_from(all_nodes)

# for all agents, add an edges for each of its trading partners
for village in village_list:   
  for vendor in village.market.vendors:
    G.add_edge(village.unique_id, vendor.owner.unique_id)


### G2
village = 601010103008
agents_lst = model.all_agents
agents_slctd = [agent.unique_id for agent in agents_lst if agent.village.unique_id == village]

G2 = create_subgraph(G, agents_slctd)
create_figure(G2)