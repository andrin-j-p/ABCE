import plotly.graph_objects as go

def create_line(x, y, y_upper, y_lower, title, ylabel):
    fig = go.Figure([
        go.Scatter(
            x=x,
            y=y,
            line=dict(color='rgb(0,100,80)'),
            mode='lines',
            name='Mean'
        ),
        go.Scatter(
            x=x + x[::-1],
            y=y_upper + y_lower[::-1],
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name='Confidence Interval'
        )
    ])

    fig.update_layout(
        title=title,
        xaxis_title='X Axis Label',
        yaxis_title=ylabel
    )

    return fig

# Sample data
x_values = [i for i in range(20)]
y_values = [5.970943175136798, 11.990342482821424, 17.874574683925264, 23.731390915298103, 29.766729740268197, 35.69750533184561, 41.68693834703355, 40.53737317644368, 39.41993525858561, 38.165110801377075, 36.92496818366155, 35.70056623120372, 34.34344510647001, 32.87515951919279, 32.65036323271969, 32.13633183672821, 31.58093251850681, 30.779419913574515, 29.989814477647073, 29.299739885506956]
y_upper_values = [41.68693834703355, 41.68693834703355, 41.68693834703355, 41.68693834703355, 41.68693834703355, 41.68693834703355, 41.68693834703355, 258.5337981678784, 258.5337981678784, 258.5337981678784, 258.5337981678784, 258.5337981678784, 258.5337981678784, 258.5337981678784, 340.60425125751163, 340.60425125751163, 340.60425125751163, 340.60425125751163, 340.60425125751163, 340.60425125751163]
y_lower_values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 41.68693834703355, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Create the Plotly plot using the create_line function
plot = create_line(x_values, y_values, y_upper_values, y_lower_values, 'Title', 'Y Axis Label')

# Show the plot
plot.show()
