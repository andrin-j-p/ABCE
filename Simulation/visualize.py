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
from datacollector import Datacollector
import plotly.figure_factory as ff
import random

def create_line1(x, df, title, ylabel):
    """
    Type:        Helper function 
    Description: Creates line plot objects used in callback
    """
    y_upper = df[f'{ylabel} Upper'].tolist()
    y_lower = df[f'{ylabel} Lower'].tolist()
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
        xaxis_title='Week',
        yaxis_title=f'{ylabel}',
        width=2200,
        height=1000,
        title_font_size=40,
        xaxis=dict(title_font=dict(size=30), tickfont=dict(size=30)),  
        yaxis=dict(title_font=dict(size=30), tickfont=dict(size=30)),
        legend=dict(font=dict(size=30)),  
        margin=dict(r=100), 
        hoverlabel=dict(font=dict(size=30)),
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
    trace1 = go.Scatter(x=df_t['step'], y=df_t[ylabel], mode='lines', name=f'Treatment', line=dict(color='rgba(0,100,80,1)'))
    trace2 = go.Scatter(x=df_c['step'], y=df_c[ylabel], mode='lines', name=f'Counterfactual', line=dict(color='rgba(0,40,100,1)'))

    # Create data array with both traces
    data = [trace1, trace2]

    # Create layout
    layout = go.Layout(
        title=f'Average Daily {title}',
        showlegend=True,
        width=2150,
        height=1000,
        title_font_size=40,
        xaxis=dict(title='Week', title_font=dict(size=30), tickfont=dict(size=30)),  
        yaxis=dict(title=f'{title}', title_font=dict(size=30), tickfont=dict(size=30)),
        legend=dict(font=dict(size=30)), 
        hoverlabel=dict(font=dict(size=30)),
 
    )

    # Create figure
    fig = go.Figure(data=data, layout=layout)

    return fig


def create_dist(df_hh, df_fm, title, var, padding_right=0, padding_left=0):
    """
    Type:        Helper function 
    Description: Plots density curve from an array and colors the area under the curve.
    """
    if var in ['Profit', 'Revenue', 'Assets']:
        df = df_fm

    else: df = df_hh

    y = df[df[f'{var}'] > 0][f'{var}']
    # Create a density plot
    fig = ff.create_distplot([y], ['Density'], show_hist=False, colors=['rgba(0,40,100,0.2)'])

    # Color the area under the curve
    for data in fig.data:
        if 'x' in data:
            data['fill'] = 'tozeroy'  # Color area under curve

    # Update layout
    fig.update_layout(
        title=dict(text = title, pad=dict(l=padding_left-50)), 
        width=2000,
        height=1000,
        title_font_size=40,
        xaxis=dict(title=f'{var}', title_font=dict(size=30), tickfont=dict(size=30)),  
        yaxis=dict(title=f'Density', title_font=dict(size=30), tickfont=dict(size=30)),
        showlegend=False,
        margin=dict(r=padding_right, l=padding_left), 
        hoverlabel=dict(font=dict(size=30)),

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


def create_subgraph_G1(G_complete, G_village, in_village_nodes):
    """
    Type:        Helper function 
    Description: Creates a subgraph of G including all the nodes connected to the list of specified nodes
    Used in:     visualize.py callback
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

    subgraph = G_village.subgraph(set(in_village_nodes) | out_village_nodes)
    node_adjacancies = []
    node_text = []
    for n, adjacencies in G_complete.adjacency():
        if subgraph.has_node(n):
            node_adjacancies.append(len(adjacencies))
            node_text.append('# of connections: '+ str(len(adjacencies)))

    return subgraph, node_adjacancies, node_text

def create_subgraph_G2(G, in_village_nodes, village):
    """
    Type:        Helper function 
    Description: Creates a subgraph of G including all the nodes connected to the list of specified nodes
    Used in:     visualize.py callback
    """
    subgraph = G.subgraph(in_village_nodes)

    # Iterate over all nodes in the village and collect their neighbors

    node_adjacancies = []
    node_text = []
    
    # Add an edge for each within village connection 
    for n, adjacencies in subgraph.adjacency():
      if G.nodes[n]['village'] == village:
        node_adjacancies.append(len(adjacencies))
        node_text.append('# of connections: '+ str(len(adjacencies)))

    return subgraph, node_adjacancies, node_text

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
            size=20,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right',
                tickfont = dict(size=30),

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
                height=2200,
                title='Inter Village Trade Network<br>',
                titlefont_size=40,
                showlegend=False,
                hovermode='closest',
                hoverlabel=dict(font=dict(size=30)),
                margin=dict(b=200,l=5,r=5,t=70),
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
                size=15,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right',
                    tickfont = dict(size=30),
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
        height=2200,
        title="Intra Village Trade Network",
        titlefont_size=40,
        margin=dict(b=200,l=5,r=5,t=70),
        showlegend=False,
        hovermode='closest',
        hoverlabel=dict(font=dict(size=30)),
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
np.random.seed(0) 
random.seed(0)
model_t = ABM.Model()
model_t.intervention_handler.control = False
model_t.datacollector = Datacollector(model_t)
model_t.run_simulation(steps)

np.random.seed(0) 
random.seed(0)
model_c = ABM.Model()
model_c.intervention_handler.control = True
model_c.datacollector = Datacollector(model_c)
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
    html.H1("Additional Simulation Outcomes", style={"textAlign":"left"}),

    html.Div(id='output_container', children=[]),
    html.Br(),

    dcc.Graph(id='ABM_map', figure={}, style={'margin-bottom': '150px'}),

    html.Div([
    dcc.Graph(id='ABM_graph1', figure={}),
    dcc.Graph(id='ABM_graph2', figure={}),
    ], style={'display': 'flex', 'flex-direction': 'row', 'margin-bottom': '150px'}),

    # Dropdown to select number of steps in the simulation
    dcc.Dropdown(id="slct_range",
                 options=[
                  {"label":'Expenditure', "value":'Expenditure'},
                  {"label":'Money', "value":'Money'},
                  {"label":'Income', "value":'Income'},
                  {"label":'Profit', "value":'Profit'},
                  {"label":'Asstes', "value":'Assets'},
                  {"label":'Revenue', "value":'Revenue'},
                  ],
                  multi=False,
                  value='Expenditure',
                  style={'width':'50%', 'font-size': '20px'}
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
    mapbox_data, sm_data_t = model_t.datacollector.get_data()
    mapbox_data['size'] = 20
    _ , sm_data_c = model_c.datacollector.get_data()

    # Create copies of the dataframes
    # @ change: negative numbers cannot be put as size
    sm_data_t['Income'] = sm_data_t['Income'].apply(lambda x: 0 if x < 0 else x)

    ## Create Scatter Mapbox
    
    # creates the scatter map and superimposes the county borders where the experiment took place
    fig1 = px.scatter_mapbox(mapbox_data, lat="lat", lon="lon", color="Money", size="size", animation_frame="step", animation_group="unique_id", 
                             color_continuous_scale=px.colors.cyclical.IceFire, size_max=20, hover_data=['Income'])

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
        width=4000 ,
        hoverlabel=dict(font=dict(size=30)),
        coloraxis_colorbar=dict(
                    title='Houshold Assets',
                    xanchor='left',
                    titleside='right',
                    titlefont=dict(size=30),
                    tickfont=dict(size=30)  # Set the font size here (adjust 18 to your desired size)
                )
                
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

    G_complete = G.copy()
    G_village = G.copy()
    village = 'v_2'


    # for all agents in 'village', add an edges for each of its trading partners
    for agent in model_t.all_agents:  
        if agent.village.unique_id == village:
            for firm in agent.best_dealers:
                G_village.add_edge(agent.unique_id, firm.owner.unique_id)


    # for all agents, add an edges for each of its trading partners
    for agent in model_t.all_agents:  
            for firm in agent.best_dealers:
                G_complete.add_edge(agent.unique_id, firm.owner.unique_id)
    
    in_village_nodes = [agent.unique_id for agent in agents_lst if agent.village.unique_id == village]

    G2, node_adjacencies, node_text = create_subgraph_G2(G_village, in_village_nodes, village)
    fig3 = create_G2(G2, node_adjacencies, node_text)


    G1, node_adjacencies, node_text = create_subgraph_G1(G_complete, G_village, in_village_nodes)
    fig2 = create_G1(G1, node_adjacencies, node_text)

    fig2.add_annotation(
    dict(
        font=dict(color='black', size=30),
        x=0,
        y=-0.1,
        showarrow=False,
        text=("This graph illustrates the trade network of agents in a randomly chosen village v with agents in other villages. <br>"
              "Inhabitants of the village are represented by nodes in the inner circle where nodes on the outer circle represent <br>"
              "trading partners from other villages. Each agent in the simulation maintains a network of up to four vendors with whom <br>"
              "they have a trade relation. Grey edges capture such relations between agents in village v and agents in other villages. <br>" 
              "The total size of an agent's trade network (incoming and outgoing) is encoded by the node color."),
        textangle=0,
        align='left',
        xanchor='left',
        xref="paper",
        yref="paper"
    )
)
    
    
    fig3.add_annotation(
    dict(
        font=dict(color='black', size=30),
        x=0,
        y=-0.1,
        showarrow=False,
        text=("This graph illustrates the intra-village trade network of agents in a randomly chosen village v. <br>"
              "Each inhabitants of the village is represented by a node and every edge represents a trade relation <br>"
              "with another agent in the same village v. The node color encodes the number of trade relations <br>"
              "each agents maintains with agents from the same village."),
        textangle=0,
        align='left',   
        xanchor='left',
        xref="paper",
        yref="paper"
    )
)
    

    ## Create Scatter Plots
    x = [i for i in range(len(sm_data_t))]

    # Average price
    fig4 = create_line1(x, sm_data_t, f'{option_slctd} Treatment', option_slctd)
    fig5 = create_line1(x, sm_data_c, f'{option_slctd} Counterfactual', option_slctd)
    fig6 = create_dist(model_t.datacollector.hh_df, model_t.datacollector.fm_df, f'Daily {option_slctd} Treatment', option_slctd, padding_right=200)
    fig7 = create_dist(model_c.datacollector.hh_df, model_t.datacollector.fm_df,f'Daily {option_slctd} Counterfactual', option_slctd, padding_left=250)
    fig8 = create_line2(sm_data_t, sm_data_c, f' {option_slctd} Recipients', f'{option_slctd} Recipient')
    fig9 = create_line2(sm_data_t, sm_data_c, f'Average Daily {option_slctd} Nonrecipients', f'{option_slctd} Nonrecipient')

    return fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9


if __name__ =='__main__':
    app.run_server(debug=False)

