

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

