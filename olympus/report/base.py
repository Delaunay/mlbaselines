class DOMComponent:
    def events(self, app):
        """Events triggered by this page

        Examples
        --------
        >>> app.callback(
        >>>     Output('a', 'value'), [Input('b', 'value')])(self.on_event)
        """
        return None

    def render(self, app, **kwargs):
        """Render an html component given its arguments

        Parameters
        ----------
        app:
            Dashboard app rendering the page
        """
        raise NotImplementedError
