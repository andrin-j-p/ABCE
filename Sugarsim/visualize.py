import json
import pandas as pd
import numpy as np
import plotly.express as px
import dash
from dash import dcc, html, Input, Output
from ABM import run_simulation
from read_data import read_dataframe
import geopandas as gpd

#start dash application
app = dash.Dash(__name__)

# Load the GeoJSON data 
file_name = read_dataframe("ken.json", retval="file")
gdf = gpd.read_file(file_name)

# Extract the three subcounties (Alego, Ugunja and Ukwala) where the experiment took place
county_list = ['CentralAlego', 'NorthAlego', 'SouthEastAlego', 'WestAlego', 'SiayaTownship', 'Ugunja', 'Ukwala' ]
filtered_gdf = gdf[gdf['NAME_3'].isin(county_list)]

# Save the filtered GeoJSON to a new file
filtered_gdf.to_file('../data/filtered_ken.json', driver='GeoJSON')
file_name = read_dataframe("filtered_ken.json", retval="file")
polygons = json.load(open(file_name, "r"))

# Get the agent information from the simulation
model = run_simulation()
df = model.df

# define app layout
app.layout = html.Div([
    
    html.H1("ABM", style={"text-align":"center"}),

    dcc.Dropdown(id="slct_range",
                 options=[
                  {"label":100, "value":100},
                  {"label":1000, "value":1000},
                  {"label":5000, "value":5000}
                  ],
                  multi=False,
                  value=100,
                  style={'width':'40%'}
                ),

    html.Div(id='output_container', children=[]),
    html.Br(),

    dcc.Graph(id='ABM_map', figure={})
  ]
)

"""
Two outputs: one goes into 'ABM_map' one into 'output' container.
component id is the reference to which tage the output goes. the component property is where the output goes within the tag.

One Input: obtained from html with id slct_range. I.p. value
"""
@app.callback(
    [Output(component_id='output_container', component_property='children'),
     Output(component_id='ABM_map', component_property='figure')],
     [Input(component_id='slct_range', component_property='value')]
)

# automatically takes the Input value as argument. If there are two inputs there are two arguments in update_graph
def update_graph(option_slctd):
    # displays the option selected
    container = f"Number of agents chosen is: {option_slctd}"

    # update [number of agents to be displayed] 
    dff = df.copy()
    dff = dff[:option_slctd]

    # creates the scatter map and superimposes the county borders where the experiment took place
    fig = px.scatter_mapbox(dff, lat="lat", lon="lon", color="p3_totincome", size="own_land_acres",
                  color_continuous_scale=px.colors.cyclical.IceFire,  height=1000).update_layout(
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

    # return the outputs
    return container, fig


if __name__ =='__main__':
    app.run_server(debug=True)