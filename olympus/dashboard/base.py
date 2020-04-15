from flask import session


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


_dash_config = {}


def insert_kv(k, v):
    global _dash_config
    _dash_config[k] = v


def get_kv(k, v=None):
    """Get value from session, fallback to global state"""
    global _dash_config
    return session.get(k, _dash_config.get(k, v))

