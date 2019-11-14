from olympus.baselines.classification import classification_baseline
from olympus.utils import fetch_device
from olympus.utils.storage import NoStorage


def test_model_init():
    params = {
        'optimizer': {
            'lr': 0.011113680070144951,
            'momentum': 0.04081791544572477,
            'weight_decay': 6.2091793568732874e-06
        },
    }

    device = fetch_device()

    model2 = classification_baseline(
        'logreg', 'glorot_uniform', 'sgd', 'none', 'test-mnist', 32, device, storage=NoStorage())

    model1 = classification_baseline(
        'logreg', 'glorot_uniform', 'sgd', 'none', 'test-mnist', 32, device, storage=NoStorage())

    model1.init(**params)
    model2.init(**params)

    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        diff = (p1 - p2).abs().sum()
        assert diff == 0
