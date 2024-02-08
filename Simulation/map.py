import json
import pandas as pd
import plotly.express as px
import ABM
from read_data import read_dataframe
import plotly.graph_objects as go

# Load GeoJSON data
file_name = read_dataframe("filtered_ken.json", retval="file")
polygons = json.load(open(file_name, "r"))

# automatically takes the Input value as argument. If there are two inputs there are two arguments in update_graph
def create_map():
    # run the simulation until treatment status is asigned
    model = ABM.Sugarscepe()
    model.run_simulation(365)

    # get village data and treatment status
    data_vl = [(vl.unique_id, vl.pos[0], vl.pos[1], vl.treated, 25)  for vl in model.all_villages]
    df_vl = pd.DataFrame(data_vl, columns=['village_id','lat', 'lon', 'treated', 'size'])

    # get market data
    data_mk = [(mk.unique_id, mk.pos[0], mk.pos[1], 'rgb(100,0,0,1)', 25)  for mk in model.all_markets]
    df_mk = pd.DataFrame(data_mk, columns=['market_id','lat', 'lon', 'color', 'size'])

    # creates the scatter map and superimposes the county borders where the experiment took place
    fig = px.scatter_mapbox(df_mk, lat="lat", lon="lon", color="color", size='size', height=5000, width=7000, size_max=0.0001)

    # Adding trace for treatment 
    fig.add_trace(go.Scattermapbox(
        lat=df_vl[df_vl['treated'] == 1]['lat'],
        lon=df_vl[df_vl['treated'] == 1]['lon'],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=30,  
            color='rgba(0,0,80, 1)', 
            opacity=1  
        ),
        name='Red Dots' 
    ))

    # Adding trace for control villages
    fig.add_trace(go.Scattermapbox(
        lat=df_vl[df_vl['treated'] == 0]['lat'],
        lon=df_vl[df_vl['treated'] == 0]['lon'],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=30,  
            color='rgba(0,0,80, 0.2)', 
            opacity=1  
        ),
        name='Red Dots' 
    ))

    # Adding trace for market
    fig.add_trace(go.Scattermapbox(
        lat=df_mk['lat'],
        lon=df_mk['lon'],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=50,  
            color='red', 
            opacity=1  
        ),
        name='Red Dots' 
    ))


    # Adding trace for study area boundary
    fig.update_layout(
        # superimpose the boundries of the study area
        mapbox={
            "style": "carto-positron",
            "zoom": 12,
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
    
    fig.show()

create_map()