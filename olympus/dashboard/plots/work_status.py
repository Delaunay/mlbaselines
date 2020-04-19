

def work_status(data):
    """Return a pie chart"""
    import plotly.graph_objects as go

    fig = go.Figure(data=[
        go.Pie(labels=tuple(data.keys()), values=tuple(data.values()))])
    fig.update_traces(hoverinfo='label+percent', textinfo='value')
    fig.update_layout(template='plotly_dark')

    return fig
