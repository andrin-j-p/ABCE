import json
import numpy as np
import plotly.express as px
import dash
import math
import networkx as nx
from dash import dcc, html, Input, Output
import ABM
from read_data import read_dataframe, create_geojson
import plotly.graph_objects as go
from Datacollector import Validation_collector2
import plotly.figure_factory as ff

# @TODO
# add selection widget for village in graph plot
# add more info on node hover

def create_line1(x, df, title, ylabel):
    """
    Type:        Helper function 
    Description: Creates line plot objects used in callback
    """
    y_upper = df[f'{ylabel}_Upper'].tolist()
    y_lower = df[f'{ylabel}_Lower'].tolist()
    y = df[f'{ylabel}'].tolist()

    fig = go.Figure([
        go.Scatter(
            x=x,
            y=y,
            line=dict(color='rgb(0,100,80)'),
            mode='lines',
            name='Mean'
        ),
        go.Scatter(
            x=x + x[::-1],  # x, then x reversed
            y=y_upper + y_lower[::-1],  # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name='Confidence Interval'
        )
    ])

    fig.update_layout(
        title=title,
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        width=2000,
        height=1000,
    )
    

    return fig


def create_line2(df_t, df_c, title, ylabel):
    """
    Type:        Helper function 
    Description: Plots two lines on the same plot.
    """
    df_t = df_t.tail(20)
    df_c = df_c.tail(20)
    # Create traces for the two lines
    trace1 = go.Scatter(x=df_t['step'], y=df_t[ylabel], mode='lines', name=f'Treated {ylabel}', line=dict(color='rgba(0,100,80,1)'))
    trace2 = go.Scatter(x=df_c['step'], y=df_c[ylabel], mode='lines', name=f'Control {ylabel}', line=dict(color='rgba(0,40,100,1)'))

    # Create data array with both traces
    data = [trace1, trace2]

    # Create layout
    layout = go.Layout(
        title=title,
        xaxis=dict(title='week'),
        yaxis=dict(title=f'{ylabel}'),
        showlegend=True,
        width=2000,
        height=1000,
    )

    # Create figure
    fig = go.Figure(data=data, layout=layout)

    return fig


def create_dist(df, title, ylabel):
    """
    Type:        Helper function 
    Description: Plots density curve from an array and colors the area under the curve.
    """
    y = df[f'{ylabel}']
    # Create a density plot
    fig = ff.create_distplot([y], ['Density'], show_hist=False, colors=['rgba(0,40,100,0.2)'])

    # Color the area under the curve
    for data in fig.data:
        if 'x' in data:
            data['fill'] = 'tozeroy'  # Color area under curve

    # Update layout
    fig.update_layout(
        title=title, 
        xaxis_title='Value', 
        yaxis_title='Density',
        width=2000,
        height=1000,
        )

    return fig

def create_circle_coordinates(N, radius):
    """
    Type:        Helper function 
    Description: Return N coordinates evenly dispersed around a circle
    Used in:     visualize.py create_subgraph 
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
    Type:        Helper function 
    Description: Creates a subgraph of G including all the nodes connected to the list of specified nodes
    Used in:     visualize.py callback
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

def create_subgraph_G2(G, in_village_nodes, village):
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
                width=2000,
                height=2000,
                title='Trade Network<br>',
                titlefont_size=25,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    return fig


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
        width=2000,    
        height=2000,
        title="3D Plotly Graph",
        titlefont_size=25,
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
    return fig

# update number of agents to be displayed
# Instantiate the model
steps = 500
model_t = ABM.Sugarscepe()
model_t.intervention_handler.control = False
model_t.datacollector = Validation_collector2(model_t)
model_t.run_simulation(steps)

model_c = ABM.Sugarscepe()
model_c.intervention_handler.control = True
model_c.datacollector = Validation_collector2(model_c)
model_c.run_simulation(steps)

#start dash application
app = dash.Dash(__name__)

# Create GeoJSON. Only needs to be executed once to construct the file
create_geojson(exectute=False)

# Load GeoJSON data
file_name = read_dataframe("filtered_ken.json", retval="file")
polygons = json.load(open(file_name, "r"))

# define app layout see https://www.youtube.com/watch?v=hSPmj7mK6ng&t=1187s
app.layout = html.Div([
    # Title
    html.H1("ABCE", style={"textAlign":"center"}),

    html.Div(id='output_container', children=[]),
    html.Br(),

    dcc.Graph(id='ABM_map', figure={}),
    html.Div([
    dcc.Graph(id='ABM_graph1', figure={}),
    dcc.Graph(id='ABM_graph2', figure={}),
    ], style={'display': 'flex', 'flex-direction': 'row'}),

    # Dropdown to select number of steps in the simulation
    dcc.Dropdown(id="slct_range",
                 options=[
                  {"label":'Expenditure', "value":'Expenditure'},
                  {"label":'Money', "value":'Money'},
                  {"label":'Income', "value":'Income'},
                  {"label":'Profit', "value":'Profit'},
                  {"label":'Asstes', "value":'Assets'},
                  {"label":'Revenue', "value":'Revenue'},
                  {"label":'Inventory', "value":'Inventory'}
                  ],
                  multi=False,
                  value='Expenditure',
                  style={'width':'40%'}
                ),

    html.Div([
    dcc.Graph(id='ABM_scatter1', figure={}),
    dcc.Graph(id='ABM_scatter2', figure={}),
    ], style={'display': 'flex', 'flex-direction': 'row'}),

    html.Br(),

    html.Div([
    dcc.Graph(id='ABM_scatter3', figure={}),
    dcc.Graph(id='ABM_scatter4', figure={}),
    ], style={'display': 'flex', 'flex-direction': 'row'}),
        html.Br(),

    html.Div([
    dcc.Graph(id='ABM_scatter5', figure={}),
    dcc.Graph(id='ABM_scatter6', figure={}),
    ], style={'display': 'flex', 'flex-direction': 'row'})

  ]
)


"""
Type:        Callback Function
Description: Updates information displayed if automatically triggered by an event
Comment:     Two outputs: one goes into 'ABM_map' one into 'output' container.
             Component id is the reference to which tage the output goes. the component property is where the output goes within the tag.

            One Input:obtained from html with id slct_range. I.p. value
"""
@app.callback(
    [Output(component_id='output_container', component_property='children')],
    [Output(component_id='ABM_map', component_property='figure'),
     Output(component_id='ABM_graph1', component_property='figure'),
     Output(component_id='ABM_graph2', component_property='figure'),
     Output(component_id='ABM_scatter1', component_property='figure'),
     Output(component_id='ABM_scatter2', component_property='figure'),
     Output(component_id='ABM_scatter3', component_property='figure'),
     Output(component_id='ABM_scatter4', component_property='figure'),
     Output(component_id='ABM_scatter5', component_property='figure'),
     Output(component_id='ABM_scatter6', component_property='figure'),],
     Input(component_id='slct_range', component_property='value')
)

# automatically takes the Input value as argument. If there are two inputs there are two arguments in update_graph
def update_graph(option_slctd):
    # displays the option selected
    container = f"Outcome of Interest: {option_slctd}"
    mapbox_data, sm_data_t = model_t.datacollector.get_data()
    mapbox_data['size'] = 20
    _ , sm_data_c = model_c.datacollector.get_data()

    # Create copies of the dataframes
    # @ change: negative numbers cannot be put as size
    sm_data_t['Income'] = sm_data_t['Income'].apply(lambda x: 0 if x < 0 else x)

    ## Create Scatter Mapbox
    
    # creates the scatter map and superimposes the county borders where the experiment took place
    fig1 = px.scatter_mapbox(mapbox_data, lat="lat", lon="lon", color="Money", size="size", animation_frame="step", animation_group="unique_id", 
                             custom_data=[], color_continuous_scale=px.colors.cyclical.IceFire, height=1000, size_max=20, 
                             hover_data=['Village', 'Income'])

    fig1.update_layout(
        # superimpose the boundries of the study area
        mapbox={
            "style": "carto-positron",
            "zoom": 11,
            "layers": [
                {
                    "source": polygons,
                    "below": "traces",
                    "type": "line",
                    "color": "black",
                    "line": {"width": 1.5},
                }
            ],
        },
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=2000,
        width=4000  # Adjust the height parameter as per your requirement
    )

    fig1.update_traces(visible=True)

    ## Create Graph Plot
        
    # initialize a graph object
    G = nx.Graph()

    # create a list of all agents in the model
    agents_lst = model_t.all_agents
    # create a node for each agent in the model 
    all_nodes = [(agent.unique_id, {'village': agent.village.unique_id, 
                                    'pos': (np.random.uniform(-1,1),np.random.uniform(-1,1),np.random.uniform(-1,1))}) 
                                    for agent in agents_lst]

    # create a graph objects with nodes for all agents in the simulation 
    G.add_nodes_from(all_nodes)

    # for all agents, add an edges for each of its trading partners
    for agent in model_t.all_agents:  
        for firm in agent.best_dealers:
            G.add_edge(agent.unique_id, firm.owner.unique_id)

    # create subgraph containing only nodes and edges in the village selected
    village = 'v_2'
    in_village_nodes = [agent.unique_id for agent in agents_lst if agent.village.unique_id == village]
    G2, node_adjacencies, node_text = create_subgraph_G2(G, in_village_nodes, village)
    fig3 = create_G2(G2, node_adjacencies, node_text)

    G1, node_adjacencies, node_text = create_subgraph_G1(G, in_village_nodes)
    fig2 = create_G1(G1, node_adjacencies, node_text)

    ## Create Scatter Plots
    x = [i for i in range(len(sm_data_t))]

    # Average price
    fig4 = create_line1(x, sm_data_t, f'Average Daily {option_slctd} Treatment', option_slctd)
    fig5 = create_line1(x, sm_data_c, f'Average Daily {option_slctd} Control', option_slctd)
    fig6 = create_dist(model_t.datacollector.hh_df, f'{option_slctd} Distribution Treatment', option_slctd)
    fig7 = create_dist(model_c.datacollector.hh_df, f'{option_slctd} Distribution Control', option_slctd)
    fig8 = create_line2(sm_data_t, sm_data_c, 'Average Daily Price', f'{option_slctd}_Recipient')
    fig9 = create_line2(sm_data_t, sm_data_c, 'Average Daily Price', f'{option_slctd}_Nonrecipient')

    return container, fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9


if __name__ =='__main__':
    app.run_server(debug=False)

