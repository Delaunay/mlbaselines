import copy

from sspace.space import compute_identity

from olympus.studies.repro.main import generate

import pytest


def test_generate():
    defaults = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    num_experiments = 2
    num_repro = 2
    objective = 'obj'
    variables = list('abc')
    resumable = False
    configs = generate(num_experiments, num_repro, objective, variables, defaults, resumable)

    assert list(configs.keys()) == variables

    for name in 'abc':
        assert len(configs[name]) == num_experiments * num_repro

    def test_doc(name, i, j):
        a_doc = copy.copy(defaults)
        a_doc[name] = int(i)
        a_doc['variable'] = name
        a_doc['repetition'] = j
        a_doc['uid'] = compute_identity(a_doc, 16)
        a_doc.pop('repetition')
        a_doc.pop('variable')
        return a_doc

    for name in 'abc':
        for i in range(num_experiments):
            for j in range(num_repro):
                k = i * num_repro + j
                assert configs[name][k] == test_doc(name, i + 1, j + 1)

    assert k == (num_experiments * num_repro) - 1


def test_generate_with_interupts():
    defaults = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    num_experiments = 10
    num_repro = 10
    objective = 'obj'
    variables = list('abc')
    resumable = True
    configs = generate(num_experiments, num_repro, objective, variables, defaults, resumable)

    assert list(configs.keys()) == variables

    for name in 'abc':
        assert len(configs[name]) == num_experiments * num_repro * 2

    def test_doc(name, i, j, interupt):
        a_doc = copy.copy(defaults)
        a_doc[name] = i
        if interupt:
            a_doc['_interrupt'] = True
        a_doc['variable'] = name
        a_doc['repetition'] = j
        a_doc.pop('uid', None)
        a_doc['uid'] = compute_identity(a_doc, 16)
        a_doc.pop('repetition')
        a_doc.pop('variable')
        return a_doc

    for name in 'abc':
        for i in range(num_experiments):
            for j in range(num_repro):
                k = (i * num_repro + j) * 2
                assert configs[name][k] == test_doc(name, i + 1, j + 1, interupt=False)
                assert configs[name][k + 1] == test_doc(name, i + 1, j + 1, interupt=True)

    assert k == (num_experiments * num_repro * 2) - 2
