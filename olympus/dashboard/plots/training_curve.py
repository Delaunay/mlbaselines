

def plot_mean_objective_altair(results, fidelity='epoch'):
    """Plot the evolution of the objective averaged over all trials, and show the

    Parameters
    ----------
    results: List[dict(fidelity=str, objective=float)]

    fidelity: str
        name of the x-axis

    Examples
    --------

    .. code-block: python

        results = [
            dict(epoch=1, objective=0.229),
            dict(epoch=1, objective=0.239),
            dict(epoch=1, objective=0.249),
            dict(epoch=2, objective=0.312),
            dict(epoch=2, objective=0.333),
            dict(epoch=2, objective=0.346),
        ]

        plot_results_altair(results, fidelity=epoch)

    .. image:: ../../../docs/_static/plots/objective.png

    """
    import altair as alt
    alt.themes.enable('dark')

    data = alt.Data(values=list(results))

    # data = pd.DataFrame(results)
    line = alt.Chart(data).mark_line().encode(
        x=alt.X(f'{fidelity}:Q'),
        y='mean(objective):Q'
    )
    band = alt.Chart(data).mark_errorband(extent='ci').encode(
        x=alt.X(f'{fidelity}:Q'),
        y=alt.Y('objective:Q', title='objective'),
    )

    graph = (band + line).configure_view(height=500, width=1000)
    return graph


def plot_training_curves_altair(results, fidelity='epoch'):
    """Plot each objective curve independently

    Parameters
    ----------
    results: List[dict(fidelity=str, objective=float, uid=str)]

    fidelity: str
        name of the x-axis

    Examples
    --------

    .. code-block: python

        results = [
            dict(epoch=1, objective=0.229, uid=0),
            dict(epoch=1, objective=0.239, uid=1),
            dict(epoch=1, objective=0.249, uid=2),
            dict(epoch=2, objective=0.312, uid=0),
            dict(epoch=2, objective=0.333, uid=1),
            dict(epoch=2, objective=0.346, uid=2),
        ]

        plot_training_curves_altair(results, fidelity=epoch)

    """
    import altair as alt
    alt.themes.enable('dark')

    data = alt.Data(values=list(results))

    # data = pd.DataFrame(results)
    line = alt.Chart(data).mark_line().encode(
        x=alt.X(f'{fidelity}:Q'),
        y='mean(objective):Q',
        colors='uid:N'
    )

    graph = line.configure_view(height=500, width=1000)
    return graph


plots = {
    'objective': {
        'altair': plot_mean_objective_altair
    }
}
