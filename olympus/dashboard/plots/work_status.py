

def work_status(status):
    """Return a pie chart for a given dictionary

    Parameters
    ----------
    status: Dict[Key, int]
        A simple dictionary with integer or floats as value and string as keys

    Examples
    --------

    .. code-block:: python

        status = {
            'pending': 10,
            'in-progress': 20,
            'finished': 40,
            'lost': 0,
            'failed': 0
        }

        fig = work_status(status)


    .. image:: ../../../docs/_static/plots/work_status.png

    """
    import plotly.graph_objects as go

    fig = go.Figure(data=[
        go.Pie(labels=tuple(status.keys()), values=tuple(status.values()))])
    fig.update_traces(hoverinfo='label+percent', textinfo='value')
    fig.update_layout(template='plotly_dark')

    return fig


def prepare_overview_altair(data):
    altair_data = []
    agents = []
    for namespace, statuses in data.items():
        altair_data.append(dict(experiment=namespace, status='pending', count=statuses['unread']))
        altair_data.append(dict(experiment=namespace, status='in-progress', count=statuses['unactioned']))
        altair_data.append(dict(experiment=namespace, status='finished', count=statuses['actioned']))
        altair_data.append(dict(experiment=namespace, status='lost', count=statuses['lost']))
        altair_data.append(dict(experiment=namespace, status='failed', count=statuses['failed']))
        agents.append(dict(experiment=namespace, agents='agents', count=statuses['agent']))
    return altair_data, agents


def aggregate_overview_altair(status, agents):
    """
    Parameters
    ----------
    status: List[dict(experiment=str, status=str, count=int)]]
        List of the count of each message status for different experiment

    agents: List[dict(experiment=str, agents=str, count=int)]

    Examples
    --------

    .. code-block:: python

        status = [
            dict(experiment='classification', status='pending', count=10),
            dict(experiment='classification', status='in-progress', count=11),
            dict(experiment='classification', status='finished', count=12)
        ]

        agents = [
            dict(experiment='classification', agents='agents', count=10)
        ]

        chart = aggregate_overview_altair(status, agents)

    .. image:: ../../../docs/_static/plots/aggregate_overview.png

    """
    import altair as alt
    alt.themes.enable('dark')

    data = alt.Data(values=status)
    chart = alt.Chart(data, title='Message status per experiment').mark_bar().encode(
        x=alt.X('count:Q', stack='normalize'),
        y='experiment:N',
        color='status:N'
    )

    data = alt.Data(values=agents)
    agent_chart = alt.Chart(data, title='Agent per experiment').mark_bar().encode(
        x=alt.X('count:Q'),
        y='experiment:N',
        color='agents:N')

    # return chart
    return alt.vconcat(chart, agent_chart)