

def work_status(data):
    """Return a pie chart

    Examples
    --------

    .. image:: ../../../docs/_static/plots/work_status.png

    """
    import plotly.graph_objects as go

    fig = go.Figure(data=[
        go.Pie(labels=tuple(data.keys()), values=tuple(data.values()))])
    fig.update_traces(hoverinfo='label+percent', textinfo='value')
    fig.update_layout(template='plotly_dark')

    return fig


def prepare_overview_altair(data):
    altair_data = []
    agents = []
    for namespace, statuses in data.items():
        altair_data.append(dict(experiment=namespace, message='pending', count=statuses['unread']))
        altair_data.append(dict(experiment=namespace, message='in-progress', count=statuses['unactioned']))
        altair_data.append(dict(experiment=namespace, message='finished', count=statuses['actioned']))
        altair_data.append(dict(experiment=namespace, message='lost', count=statuses['lost']))
        altair_data.append(dict(experiment=namespace, message='failed', count=statuses['failed']))
        agents.append(dict(experiment=namespace, message='agents', count=statuses['agent']))
    return altair_data, agents


def aggregate_overview_altair(status, agents):
    """
    Examples
    --------

    .. image:: ../../../docs/_static/plots/aggregate_overview.png
    """
    import altair as alt
    alt.themes.enable('dark')

    data = alt.Data(values=status)
    chart = alt.Chart(data, title='Message status per experiment').mark_bar().encode(
        x=alt.X('count:Q', stack='normalize'),
        y='experiment:N',
        color='message:N'
    )

    data = alt.Data(values=agents)
    agent_chart = alt.Chart(data, title='Agent per experiment').mark_bar().encode(
        x=alt.X('count:Q'),
        y='experiment:N',
        color='message:N')

    # return chart
    return alt.vconcat(chart, agent_chart)