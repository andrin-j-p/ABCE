#%%
import holoviews as hv
from holoviews import dim
import ABM
import pandas as pd

hv.extension('matplotlib')
hv.output(fig='svg', size=1000)

steps = 2
model = ABM.Sugarscepe()
model.run_simulation(steps)
_, _, _, df_td = model.datacollector.get_data()
df_td = df_td[(df_td['step'] == steps - 1) & (df_td['market'] == 'm_116')]
edges = df_td.groupby(['from', 'to'], as_index=False)['volume'].sum()
edges = edges.rename(columns={'from': 'source', 'to': 'target', 'volume': 'value'})


# Create a new DataFrame with the unique values and sort it
nodes = pd.DataFrame({'villages': list(set(edges['source'].unique()) | set(edges['target'].unique()))})

nodes = hv.Dataset(pd.DataFrame(nodes['villages']))

# Create Chord plot directly from Pandas DataFrames
chord_plot = hv.Chord((edges, nodes)).options(
    node_cmap='Category20', 
    edge_cmap='Category20',    
    node_color=dim('villages').astype(str),
    edge_color=dim('source').astype(str),
    # You can customize the edge color as needed
)
chord_plot



# %%
