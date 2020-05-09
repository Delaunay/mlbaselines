

def importance_heatmap_plotly(fanova, columns):
    import plotly.graph_objects as go

    fig = go.Figure(data=go.Heatmap(
        z=fanova.importance, x=columns, y=list(reversed(columns))))

    fig_std = go.Figure(data=go.Heatmap(
        z=fanova.importance_std, x=columns, y=list(reversed(columns))))

    fig.update_layout(template='plotly_dark')
    fig_std.update_layout(template='plotly_dark')

    return fig, fig_std


def importance_heatmap_altair(fanova):
    """Outputs the importance of each hyper-parameter according to FANOVA

    Parameters
    ----------
    fanova: FANOVA
        instance of FANOVA class

    Examples
    --------

    >>> from olympus.dashboard.analysis.hpfanova import FANOVA
    >>> import pandas as pd
    >>> data = [
    ...     dict(objective=0.12 / 0.08, uid=0, epoch=32, hp1=0.12, hp2=0.08),
    ...     dict(objective=0.14 / 0.09, uid=0, epoch=32, hp1=0.14, hp2=0.09),
    ...     dict(objective=0.15 / 0.10, uid=0, epoch=32, hp1=0.15, hp2=0.10),
    ...     dict(objective=0.16 / 0.11, uid=0, epoch=32, hp1=0.16, hp2=0.11),
    ...     dict(objective=0.17 / 0.12, uid=0, epoch=32, hp1=0.17, hp2=0.12)
    ... ]
    >>> space = {
    ...     'hp1': 'uniform(0, 1)',
    ...     'hp2': 'uniform(0, 1)'
    ... }
    >>> data = pd.DataFrame(data)
    >>> fanova = FANOVA(
    ...    data,
    ...    hp_names=list(space.keys()),
    ...    objective='objective',
    ...    hp_space=space)
    >>> chart = importance_heatmap_altair(fanova)

    .. image:: ../../../docs/_static/plots/importance.png

    """
    import altair as alt
    alt.themes.enable('dark')

    data = alt.Data(values=fanova.importance_long)

    base = alt.Chart(data).mark_rect().encode(
        x='row:O',
        y='col:O'
    ).properties(
        width=200,
        height=200
    )

    chart = alt.concat(
        base.encode(color='importance:Q'),
        base.encode(color='std:Q')
    ).resolve_scale(
        color='independent'
    )

    return chart


def marginals_altair(fanova):
    """Outputs the marginal effect of each hyper-parameter according to FANOVA

    Parameters
    ----------
    fanova: FANOVA
        instance of FANOVA class

    Examples
    --------

    .. image:: ../../../docs/_static/plots/marginals.png

    """
    import altair as alt
    alt.themes.enable('dark')

    data = fanova.compute_marginals()
    marginals = alt.Data(values=data)

    base = alt.Chart(marginals).encode(
        alt.X('value', type='quantitative'),
        alt.Y('objective', type='quantitative'),
        yError='std:Q',
        color='name:N'
    ).properties(
        width=200,
        height=200
    )

    chart = (base.mark_errorband() + base.mark_line())\
        .facet(column='name:N').interactive()

    return chart


plots = {
    'importance': {
        'altair': importance_heatmap_altair,
        'plotly': importance_heatmap_plotly
    },
    'marginal': {
        'altair': marginals_altair
    }
}

