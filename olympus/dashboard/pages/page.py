from olympus.dashboard.base import DOMComponent


class Page(DOMComponent):
    @staticmethod
    def route():
        """URL pattern used to access this page

        Examples
        --------
        >>> '/project/?(?P<project_id>[a-zA-Z0-9]*)'
        """
        raise NotImplementedError()

    def events(self, app):
        """Events triggered by this page

        Examples
        --------
        >>> app.callback(
        >>>     Output('a', 'value'), [Input('b', 'value')])(self.on_event)
        """
        return None

    def render(self, app, *args, **kwargs):
        """Render a webpage given its arguments

        Parameters
        ----------
        app:
            Dashboard app rendering the page

        kwargs:
            Route parameters defined in `self.route()`
        """
        raise NotImplementedError
