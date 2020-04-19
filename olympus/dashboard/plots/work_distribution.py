from olympus.hpo.parallel import WORK_ITEM, HPO_ITEM, WORKER_LEFT, WORKER_JOIN, RESULT_ITEM, SHUTDOWN

mappings = {
    WORK_ITEM: 'trial',
    HPO_ITEM: 'hpo',
    WORKER_LEFT: 'worker_left',
    WORKER_JOIN: 'worker_join',
    RESULT_ITEM: 'result',
    SHUTDOWN: 'shutdown'
}


def prepare_gantt_array(work_items, worker_count):
    """

    Returns
    -------
    jobs: [{Task, Start, Finish, Resource}]
    annotations: [{x, y, text}]

    Examples
    --------

    .. code-block:: python

        messages = monitor.messages(queue, namespace)
        _, _ = prepare_gantt_array(*extract_work_messages(messages))

    .. image:: ../../../docs/_static/hpo/hyperband.png
        :width: 45 %

    """
    workers = {i: None for i in range(worker_count + 1)}

    def find_free_worker(start_time, end_time):
        """Find a free worked or create a new worker"""
        for i, worker in enumerate(workers.values()):
            if worker is None or worker < start_time:
                workers[i] = end_time
                return i

        workers[len(workers)] = end_time
        return len(workers)

    jobs = []
    annotations = []
    unique_resources = set()

    for w in work_items:
        worker_id = find_free_worker(w.read_time, w.actioned_time)

        resource = mappings.get(w.mtype)
        epoch = dict(w.message.get('kwargs', [])).get('epoch', None)
        if epoch is not None:
            resource = f'{resource} ({epoch})'

        task = f'worker-{worker_id}'
        unique_resources.add(resource)
        jobs.append(dict(
            Task=task,
            Start=w.read_time,
            Finish=w.actioned_time,
            epoch=epoch,
            Resource=resource))

        # annotations.append(dict(
        #     x=w.read_time + (w.actioned_time - w.read_time) / 2,
        #     y=worker_id,
        #     text=str(w.uid)[:4],
        #     showarrow=True))

    return jobs, annotations, unique_resources


def plot_gantt_plotly(jobs, annotations=None, resources=None):
    import plotly.colors
    import plotly.figure_factory as ff
    from olympus.dashboard.plots.utilities import colors_1024

    colors = plotly.colors.DEFAULT_PLOTLY_COLORS

    # We need more colors
    if len(resources) > len(colors):
        missing = len(resources) - len(colors)
        colors.extend(colors_1024[-missing - 1:-1])

    fig = ff.create_gantt(
        jobs, title='Work Schedule', index_col='Resource', colors=colors, showgrid_x=True, showgrid_y=True,
        show_colorbar=True, group_tasks=True, bar_width=0.4, reverse_colors=True)

    if annotations:
        fig['layout']['annotations'] = annotations

    fig.update_layout(template='plotly_dark')

    # make the separation between items more obvious
    for data in fig['data']:
        data['marker']['symbol'] = 'line-ns'
        data['marker']['size'] = 20
        data['marker']['opacity'] = 1
        data['marker']['line'] = {
            'width': 1
        }

    return fig


plots = {
    'gantt': {
        'plotly': plot_gantt_plotly
    }
}
