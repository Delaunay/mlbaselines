from argparse import ArgumentParser, RawDescriptionHelpFormatter
import torch

from olympus.datasets import SplitDataset, Dataset, DataLoader
from olympus.datasets.decorator.window import WindowedDataset
from olympus.metrics import Loss
from olympus.models import Model

from olympus.optimizers import Optimizer, known_optimizers
from olympus.tasks.finance import Finance, SharpeRatioCriterion
from olympus.utils import option, fetch_device, show_hyperparameter_space, parse_args
from olympus.observers import metric_logger, CheckPointer

from olympus.utils.storage import StateStorage


DEFAULT_EXP_NAME = 'finance'
base = option('base_path', '/media/setepenre/local/')


def arguments():
    parser = ArgumentParser(
        prog='finance',
        description='Finance Baseline',
        epilog=show_hyperparameter_space(),
        formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument(
        '--uri', type=str, default=None,
        help='Resource to use to store metrics')

    parser.add_argument(
        '--database', type=str, default='olympus',
        help='which database to use')

    parser.add_argument(
        '--arg-file', type=str, default=None, metavar='ARGS',
        help='Json File containing the arguments to use')

    parser.add_argument(
        '--optimizer', type=str, default='adam',
        metavar='OPTIMIZER_NAME', choices=known_optimizers(),
        help='Name of the optimiser (default: adam)')

    parser.add_argument(
        '--batch-size', type=int, default=32, metavar='B',
        help='input batch size for training (default: 32)')

    parser.add_argument(
        '--epochs', type=int, default=100, metavar='N',
        help='maximum number of epochs to train (default: 100)')

    parser.add_argument(
        '--window', type=int, default=70, metavar='W',
        help='Windown timeframe in days (default: 70)')

    parser.add_argument(
        '--storage', action='store_true', default=False,
        help='Enable storage')

    return parse_args(parser)


def oracle(x):
    x = x[:, :, 1:] - x[:, :, :-1]
    u = x.mean(axis=2).unsqueeze(2)
    x = x - u
    cov = torch.matmul(x, x.transpose(2, 1)) / x.shape[2]

    return cov, u


def finance_baseline(tickers, start, end, optimizer, batch_size, device, window=70, sampler_seed=0, hpo_done=False):
    dataset = Dataset(
        'stockmarket',
        path=f'{base}/data',
        tickers=tickers,
        start_date=start,
        end_date=end)

    dataset = WindowedDataset(
        dataset,
        window=window,
        transforms=lambda x: x.transpose(1, 0),
        overlaps=True
    )

    dataset = SplitDataset(
        dataset,
        split_method='original'
    )

    loader = DataLoader(
        dataset,
        sampler_seed=sampler_seed,
        batch_size=batch_size
    )

    model = Model(
        'MinVarianceReturnMomentEstimator',
        weight_init='noinit',
        input_size=(len(tickers), window),
        lags=2).to(device=device)

    optimizer = Optimizer(optimizer)

    train, valid, test = loader.get_loaders(hpo_done=hpo_done)

    main_task = Finance(
        model=model,
        optimizer=optimizer,
        oracle=oracle,
        dataset=train,
        device=device,
        criterion=SharpeRatioCriterion())

    name = 'validation'
    if hpo_done:
        name = 'test'

    main_task.metrics.append(Loss(name=name, loader=test))
    main_task.metrics.append(Loss(name='train', loader=train))

    return main_task


def main():
    from sspace.space import compute_identity

    args = arguments()
    tickers = [
        # 1     2      3     4      5       6     7     8      9    10
        'MO', 'AEP', 'BA', 'BMY', 'CPB', 'CAT', 'CVX', 'KO', 'CL', 'COP',  # 1
        'ED', 'CVS', 'DHI', 'DHR', 'DRI', 'DE', 'D', 'DTE', 'ETN', 'EBAY',  # 2
        'F', 'BEN', 'HSY', 'HBAN', 'IBM', 'K', 'GIS', 'MSI', 'NSC', 'TXN'
    ]
    start, end = '2000-01-01', '2019-05-10'

    device = fetch_device()

    task = finance_baseline(
        tickers, start, end,
        args.optimizer,
        args.batch_size,
        device,
        args.window)

    lr = 1e-8
    uid = compute_identity(dict(
        tickers=tickers,
        start=start,
        end=end,
        window=args.window,
        lr=lr,
        epochs=args.epochs
    ), 16)

    if args.uri is not None:
        logger = metric_logger(args.uri, args.database, f'{DEFAULT_EXP_NAME}_{uid}')
        task.metrics.append(logger)

    if args.storage is not None:
        storage = StateStorage(folder=option('state.storage', '/home/setepenre/zshare/tmp'))
        task.metrics.append(CheckPointer(storage=storage, time_buffer=5, keep_best='validation_loss', save_init=True))

    optimizer = task.optimizer.defaults
    optimizer['lr'] = lr

    task.init(optimizer=optimizer, uid=uid)
    task.fit(args.epochs)

    stats = task.metrics.value()
    print(stats)
    return float(stats['validation_loss'])


if __name__ == '__main__':
    main()
