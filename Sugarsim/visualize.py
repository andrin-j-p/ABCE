# %%
import json
import pandas as pd
import numpy as np
import plotly.express as px
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
from ABM import run_simulation
from read_data import read_dataframe
import geopandas as gpd

app = dash.Dash(__name__)

# Load the GeoJSON data 

# Read the GeoJSON file
file_name = read_dataframe("ken.json", retval="file")
gdf = gpd.read_file(file_name)
gdf['NAME_3']

#%%
#Alego, Ugunja and Ukwala
county_list = ['CentralAlego', 'NorthAlego', 'SouthEastAlego', 'WestAlego', 'SiayaTownship', 'Ugunja', 'Ukwala' ]
filtered_gdf = gdf[gdf['NAME_3'].isin(county_list)]


# Save the filtered GeoJSON to a new file
filtered_gdf.to_file('../data/filtered_ken.json', driver='GeoJSON')


file_name = read_dataframe("filtered_ken.json", retval="file")
polygons = json.load(open(file_name, "r"))

#%%
# Ensure that the "fips" values in the DataFrame correspond to the "id" values in GeoJSON
l = []
for feature in polygons['features']:
    l.append(feature["properties"]["GID_3"])
length = len(l)

# Generate some data for each region defined in GeoJSON. 
# @TODO use real population data 
df = pd.DataFrame({"fips": l, "unemp": np.random.uniform(0.4, 10.4, length)})

# get the agent positions from the simulation
agent_pos = run_simulation()
agent_lon = [pos[1] for pos in agent_pos]
agent_lat = [pos[0] for pos in agent_pos]

# define app layout
app.layout = html.Div([
    
    html.H1("ABM", style={"text-align":"center"}),

    dcc.Dropdown(id="slct_range",
                 options=[
                  {"label":10, "value":10},
                  {"label":20, "value":20},
                  {"label":30, "value":30}
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

    container = f"Range chosen is: {option_slctd}"

    # update information on unemployment 
    dff = df.copy()
    dff['unemp'] = np.random.uniform(0, option_slctd, length)
    
    # https://plotly.com/python/scattermapbox/
    # https://plotly.com/python/reference/choropleth/
    # @TODO add hover
    fig = go.Figure()
    fig.add_trace(go.Choropleth(
            geojson=polygons,
            locations=dff["fips"],
            featureidkey="properties.GID_3",
            z=dff["unemp"],
            colorscale="Viridis", 
            zmin=0, 
            zmax=option_slctd,
        )
    )

    # Add agent as dot
    fig.add_trace(go.Scattergeo(
      lon = agent_lon,
      lat = agent_lat,
      mode = 'markers',
      marker_color = "red",
      ))

    # set focus of the map
    fig.update_geos(fitbounds="locations", visible=True)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0} )

    #outputs
    return container, fig


if __name__ =='__main__':
    app.run_server(debug=True)