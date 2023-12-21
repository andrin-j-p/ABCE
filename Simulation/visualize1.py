import json
import pandas as pd
import numpy as np
import plotly.express as px
import dash
import math
import networkx as nx
from dash import dcc, html, Input, Output
import ABM
from read_data import read_dataframe, create_geojson
import plotly.graph_objects as go
import pickle

#start dash application
app = dash.Dash(__name__)

# Create GeoJSON. Only needs to be executed once to construct the file
create_geojson(exectute=False)

# Load GeoJSON data
file_name = read_dataframe("filtered_ken.json", retval="file")
polygons = json.load(open(file_name, "r"))

# Instantiate the model
model = ABM.Sugarscepe()

# define app layout see https://www.youtube.com/watch?v=hSPmj7mK6ng&t=1187s
app.layout = html.Div([
    # Title
    html.H1("ABCE", style={"textAlign":"center"}),

    # Dropdown to select number of steps in the simulation
    dcc.Dropdown(id="slct_range",
                 options=[
                  {"label":100, "value":100},
                  {"label":200, "value":200},
                  {"label":400, "value":400}
                  ],
                  multi=False,
                  value=200,
                  style={'width':'40%'}
                ),

    html.Div(id='output_container', children=[]),
    html.Br(),

    dcc.Graph(id='ABM_map', figure={}),

    html.Div([
    #scatters
    dcc.Graph(id='ABM_scatter1', figure={}),
    dcc.Graph(id='ABM_scatter2', figure={}),
    dcc.Graph(id='ABM_scatter3', figure={}),
    dcc.Graph(id='ABM_scatter4', figure={}),
    dcc.Graph(id='ABM_scatter5', figure={}),

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
    [Output(component_id='output_container', component_property='children'),
     Output(component_id='ABM_map', component_property='figure')],
     Output(component_id='ABM_scatter1', component_property='figure'),
     Output(component_id='ABM_scatter2', component_property='figure'),
     Output(component_id='ABM_scatter3', component_property='figure'),
     Output(component_id='ABM_scatter4', component_property='figure'),
     Output(component_id='ABM_scatter5', component_property='figure'),
     [Input(component_id='slct_range', component_property='value')]
)

# automatically takes the Input value as argument. If there are two inputs there are two arguments in update_graph
def update_graph(option_slctd):
    # displays the option selected
    container = f"Number of steps: {option_slctd}"

    # update number of agents to be displayed
    # Open the pickled file and load the DataFrame
    with open('../data/output_data/model_output.pkl', 'rb') as file:
      df_hh, df_fm, df_md = pickle.load(file)
    # Create copies of the dataframes
    dff_hh, dff_fm, dff_md = df_hh[:], df_fm[:], df_md[:]

    mapbox_data= dff_hh[dff_hh['step'] % 10 == 0]

    # creates the scatter map and superimposes the county borders where the experiment took place
    fig1 = px.scatter_mapbox(mapbox_data, lat="lat", lon="lon", color="money", size="income", animation_frame="step", animation_group="unique_id", 
                             custom_data=[], color_continuous_scale=px.colors.cyclical.IceFire, height=1000, size_max=20, 
                             hover_data=['village_id', 'income'])
    fig1.update_traces(marker=dict(size=15))

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
    )

    fig1.add_trace(
        go.Scattermapbox(
            name='flow1',
            lon = [134.340916, -3.704239],
            lat = [-25.039402, 40.415887],
            mode = 'lines',
            line = dict(width = 8,color = 'green')
        )
    )

    fig1.update_traces(visible=True)

### Create Scatter Plots

    x = [i for i in range(option_slctd)]

    # Get average daily price
    y = dff_md['average_price']

    # Create figure object
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=x, y=y,mode='markers'))
    fig3.update_layout(title='Average Daily Price', xaxis_title = 'Step', yaxis_title = 'Price', 
                       width=1000,    
                       height=500,
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=True),
                       yaxis=dict(showgrid=False, zeroline=True, showticklabels=True))
    
    # get average inventory
    y = dff_md['average_stock']

    # create figure object
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=x, y=y,mode='markers'))
    fig4.update_layout(title='Average Daily Inventory', xaxis_title = 'Step', yaxis_title = 'Inventory', 
                       width=1000,    
                       height=500,
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=True),
                       yaxis=dict(showgrid=False, zeroline=True, showticklabels=True))

    # get average employment
    y = dff_md['average_employees']
    
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=x, y=y,mode='markers'))
    fig5.update_layout(title='Average Daily employees', xaxis_title = 'Step', yaxis_title = 'employees', 
                       width=1000,    
                       height=500,
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=True),
                       yaxis=dict(showgrid=False, zeroline=True, showticklabels=True))
    
    # get sales 
    y = dff_md['trade_volume']
    
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=x, y=y,mode='markers'))
    fig6.update_layout(title='Daily Trade Volume', xaxis_title = 'Step', yaxis_title = 'Amount', 
                       width=1000,    
                       height=500,
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=True),
                       yaxis=dict(showgrid=False, zeroline=True, showticklabels=True))

    # get sales 
    y = dff_md['average_income']
    
    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(x=x, y=y,mode='markers'))
    fig7.update_layout(title='Average Income', xaxis_title = 'Step', yaxis_title = 'Income', 
                       width=1000,    
                       height=500,
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=True),
                       yaxis=dict(showgrid=False, zeroline=True, showticklabels=True))

    return container, fig1, fig3, fig4, fig5, fig6, fig7


if __name__ =='__main__':
    app.run_server(debug=True)