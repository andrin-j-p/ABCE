import json
import numpy as np
import plotly.express as px
from read_data import read_dataframe, create_geojson
import pandas as pd
# Load GeoJSON data
file_name = read_dataframe("filtered_ken.json", retval="file")
polygons = json.load(open(file_name, "r"))


data = {
    'lat': [34, 32, 33, 34],
    'lon': [0, 0, 1, 1],
    'color': ['blue', 'red', 'blue', 'red'],
    'size': [100,100,100, 100]
}

df = pd.DataFrame(data)

# creates the scatter map and superimposes the county borders where the experiment took place
fig1 = px.scatter_mapbox(data, lat="lat", lon="lon", color="color", size="size" ,
                         custom_data=[], color_continuous_scale=px.colors.cyclical.IceFire, height=1000, size_max=200,
                         hover_data=['color'])

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
fig1.show()