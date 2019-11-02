

class ReportMeta(type):
    tasks = []

    def __call__(cls, *args, **kwargs):
        """Catch the task object being created to generate a report afterwards"""
        task = super(ReportMeta, cls).__call__(*args, **kwargs)
        ReportMeta.tasks.append(task)
        return task
