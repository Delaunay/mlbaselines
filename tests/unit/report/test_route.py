from olympus.dashboard.dashboard import router, add_regex_route, add_route, DuplicateRoute
import pytest


def add(app, a, b):
    return int(a) + int(b)


def show(app):
    return 1


def test_add_regex_route():
    add_regex_route('/add/(?P<a>[a-zA-Z0-9]*)/(?P<b>[a-zA-Z0-9]*)', add)
    assert router(None, '/add/1/2') == 3


def test_simple_route():
    add_route('/show', show)
    assert router(None, '/show') == 1


def test_non_existent_route():
    assert router(None, '/whatever')


def test_duplicate_regex_route():
    with pytest.raises(DuplicateRoute):
        add_regex_route('/add/(?P<a>[a-zA-Z0-9]*)/(?P<b>[a-zA-Z0-9]*)', add)


def test_duplicate_simple_route():
    with pytest.raises(DuplicateRoute):
        add_route('/show', show)

