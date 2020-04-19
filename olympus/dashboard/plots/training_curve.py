

def plot_results_altair(results, fidelity='epoch'):
    """
    Parameters
    ----------
    results: [{'fidelity': ..., 'objective'...}]
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


plots = {
    'objective': {
        'altair': plot_results_altair
    }
}
