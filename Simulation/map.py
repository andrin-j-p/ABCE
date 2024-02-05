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

# define app layout see https://www.youtube.com/watch?v=hSPmj7mK6ng&t=1187s
app.layout = html.Div([
    # Title
    html.H1("ABCE", style={"textAlign":"center"}),

    html.Div(id='output_container', children=[]),
    html.Br(),

    dcc.Graph(id='ABM_map', figure={}),
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
    [Input(component_id='slct_range', component_property='value')]

)

# automatically takes the Input value as argument. If there are two inputs there are two arguments in update_graph
def update_graph(option_slctd):
    container = f"Number of steps: {option_slctd}"

    model = ABM.Sugarscepe()
    model.run_simulation(1)

    village_location = [vl.pos for vl in model.all_villages]
    df_vl = pd.DataFrame(village_location, columns=['lat', 'lon'])
    df_vl['trated'] = [vl.treated for vl in model.all_villages]
    df_vl['size'] = 1
    print(df_vl.head(5))

    # creates the scatter map and superimposes the county borders where the experiment took place
    fig1 = px.scatter_mapbox(df_vl, lat="lat", lon="lon", color="treated", size="size", animation_frame="step", animation_group="unique_id", 
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
    
    return container, fig1


if __name__ =='__main__':
    app.run_server(debug=True)