import json
import pandas as pd
import numpy as np
import plotly.express as px
import dash
import networkx as nx
from dash import dcc, html, Input, Output
import ABM
from read_data import read_dataframe, create_geojson
import plotly.graph_objects as go


#start dash application
app = dash.Dash(__name__)

# Create GeoJSON. Only needs to be executed once to construct the file
create_geojson(exectute=False)

# Load GeoJSON data
file_name = read_dataframe("filtered_ken.json", retval="file")
polygons = json.load(open(file_name, "r"))

# Get the agent information from the simulation
#model = ABM.run_simulation()
#df = model.get_data()

# define app layout
app.layout = html.Div([
    # Title
    html.H1("ABCE", style={"textAlign":"center"}),

    # Dropdown to select number of steps in the simulation
    dcc.Dropdown(id="slct_range",
                 options=[
                  {"label":10, "value":10},
                  {"label":100, "value":100},
                  {"label":1000, "value":1000}
                  ],
                  multi=False,
                  value=10,
                  style={'width':'40%'}
                ),

    html.Div(id='output_container', children=[]),
    html.Br(),

    dcc.Graph(id='ABM_map', figure={}),
    dcc.Graph(id='ABM_graph', figure={})
  ]
)

"""
Type: Callback Function
Description: 
Comment:    Two outputs: one goes into 'ABM_map' one into 'output' container.
            component id is the reference to which tage the output goes. the component property is where the output goes within the tag.

            One Input:obtained from html with id slct_range. I.p. value
"""
@app.callback(
    [Output(component_id='output_container', component_property='children'),
     Output(component_id='ABM_map', component_property='figure'),
     Output(component_id='ABM_graph', component_property='figure')],
     [Input(component_id='slct_range', component_property='value')]
)

# automatically takes the Input value as argument. If there are two inputs there are two arguments in update_graph
def update_graph(option_slctd):
    # displays the option selected
    container = f"Number of steps: {option_slctd}"

    # update number of agents to be displayed
    model = ABM.run_simulation(option_slctd)
    dff = model.get_data()
    dff = dff[:]

    """
    Create Scatter Mapbox
    """
    # creates the scatter map and superimposes the county borders where the experiment took place
    fig1 = px.scatter_mapbox(dff, lat="lat", lon="lon", color="income", size="output", animation_frame="step",
                  animation_group="unique_id", color_continuous_scale=px.colors.cyclical.IceFire, height=1000, size_max=20).update_layout(
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
    )

    fig1.update_traces(visible=True)

    """
    Create Graph Plot
    """
    # @TODO make a widget to specify how the graph should be displayed (possibly by village)
    #       outsource createtion such that graph object is only created once then create subgraph in callback
    #       how to make out of village nodes red
    #       scale coordinates
    #       add comments
    
    # initialize a graph object
    G = nx.Graph()

    # get the list of agents to be displayed in the graph plot
    agents_slctd = list(model.schedule.agents_by_type[ABM.Agent].values())

    # create a dict with agent's unique_id as key an postion as value
    node_positions = {agent.unique_id: agent.pos for agent in agents_slctd}
    G.add_nodes_from(node_positions.keys())

    # 
    for n, p in node_positions.items():
        G.nodes[n]['pos'] = p
    
    for row in dff[dff['step'] == 1].itertuples():
        if len(row.best_dealers) > 0:
            for dealer in row.best_dealers:
                G.add_edge(row.unique_id, dealer.unique_id)

    G = nx.subgraph(G, [agent.unique_id for agent in agents_slctd[:]])

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
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

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

    fig2 = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                width=2000,
                height=1000,
                title='Trade Network<br>',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )

    return container, fig1, fig2


if __name__ =='__main__':
    app.run_server(debug=True)