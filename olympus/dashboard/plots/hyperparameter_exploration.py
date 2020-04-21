def scatter_matrix_plotly(data, columns):
    # Looks ugly
    import plotly.graph_objects as go
    import pandas as pd

    df = pd.DataFrame(data)
    index_vals = df['epoch'].astype('category').cat.codes

    fig = go.Figure(data=go.Splom(
        showlowerhalf=False,
        diagonal_visible=False,
        text=df['epoch'],
        dimensions=[
            dict(label=col, values=df[col]) for col in columns],
        marker=dict(
            color=index_vals,
            showscale=False,
            line_color='white',
            line_width=0.5)))

    fig.update_layout(template='plotly_dark')
    fig.update_layout(
        showlegend=True,
        width=600,
        height=600)

    return fig


def scatter_matrix_altair(configs, columns, color='epoch'):
    """Plots hyper-parameter space exploration

    Parameters
    ----------
    configs: List[dict]
        A list of configuration tried by the hyper-parameter

    columns: List[str]
        A list of the hyper-parameters

    color: str
        Dimension to use to color each points

    Examples
    --------

    .. code-block:: python

        columns = ['a', 'b', 'c']
        data = [
            dict(a=1, b=2, c=3, epoch=1),
            dict(a=2, b=1, c=1, epoch=2),
            dict(a=3, b=3, c=2, epoch=3),
        ]

        chart = scatter_matrix_altair(data, columns, color)

    .. image:: ../../../docs/_static/plots/space_exploration.png

    """
    import altair as alt
    alt.themes.enable('dark')

    from olympus.dashboard.plots.utilities import AltairMatrix

    space = alt.Data(values=configs)
    base = alt.Chart().properties(
        width=120,
        height=120
    )

    def scatter_plot(row, col):
        """Standard Scatter plot"""
        return base.mark_circle(size=5).encode(
            alt.X(row, type='quantitative'),
            alt.Y(col, type='quantitative'),
            color=f'{color}:N'
        ).interactive()

    def density_plot(row):
        """Estimate the density function using KDE"""
        return base.transform_density(
            row,
            as_=[row, 'density']
        ).mark_line().encode(
            x=f'{row}:Q',
            y='density:Q'
        )

    def histogram_plot(row):
        """Show density as an histogram"""
        return base.mark_bar().encode(
            alt.X(row, type='quantitative', bin=True),
            y='count()'
        )

    return (AltairMatrix(space)
            .fields(*columns)
            # .upper(scatter_plot)
            .diag(histogram_plot)
            .lower(scatter_plot)).render()


plots = {
    'exploration': {
        'altair': scatter_matrix_altair,
        'plotly': scatter_matrix_plotly
    }
}
