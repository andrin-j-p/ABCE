import json
import pandas as pd
import numpy as np
import plotly.express as px
import dash
from dash import dcc, html, Input, Output
import ABM 
from read_data import read_dataframe, create_geojson

#start dash application
app = dash.Dash(__name__)

# Create GeoJSON. Only needs to be executed once to construct the file
create_geojson(exectute=False)

# Load GeoJSON data
file_name = read_dataframe("filtered_ken.json", retval="file")
polygons = json.load(open(file_name, "r"))

# Get the agent information from the simulation
model = ABM.run_simulation()
df = model.get_data()

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

    dcc.Graph(id='ABM_map', figure={})
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
     Output(component_id='ABM_map', component_property='figure')],
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

    # creates the scatter map and superimposes the county borders where the experiment took place
    fig = px.scatter_mapbox(dff, lat="lat", lon="lon", color="income", size="output", animation_frame="step",
                  animation_group="id", color_continuous_scale=px.colors.cyclical.IceFire, height=1500, size_max=20).update_layout(
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

    fig.update_traces(visible=True)

    return container, fig


if __name__ =='__main__':
    app.run_server(debug=True)