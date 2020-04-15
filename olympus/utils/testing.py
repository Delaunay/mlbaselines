from olympus.hpo.utility import hyperparameters


@hyperparameters(epoch='fidelity(1, 30, 4)', lr='uniform(0, 1)', b='uniform(0, 1)', c='uniform(0, 1)')
def my_trial(epoch, lr, a, b, c, **kwargs):
    import time
    time.sleep(epoch / 10)
    return lr * a - b * c
