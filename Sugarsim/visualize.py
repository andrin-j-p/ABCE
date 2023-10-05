# %%
import json
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
pio.renderers = "chrome"

# Load the GeoJSON data
polygons = json.load(open("C:\\Users\\andri\\OneDrive\\UNI\\Semester 13\\Thesis\\Sugarsim\\Sugarsim\\ken.json", "r"))

polygons["features"][500]
# Ensure that the "fips" values in the DataFrame correspond to the "id" values in GeoJSON
l = []
for feature in polygons['features']:
    l.append(feature["properties"]["GID_3"])
length = len(l)

# Generate some data for each region defined in GeoJSON
df = pd.DataFrame({"fips": l, "unemp": np.random.uniform(0.4, 10.4, length)})

# %%
# Create the choropleth map
fig = px.choropleth(
    df,
    geojson=polygons,
    locations="fips",
    featureidkey="properties.GID_3",
    color="unemp",
    color_continuous_scale="Viridis",
    range_color=(0, 12),
    scope="africa",
    labels={"unemp": "unemployment rate"},
)

# Update layout and show the map
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.update_geos(fitbounds="locations", visible=False)
fig.show()

# %%
