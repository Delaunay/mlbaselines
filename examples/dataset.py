from olympus.datasets import Dataset, SplitDataset, DataLoader
from olympus.utils import option, new_seed, get_seeds
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('dataset', type=str, help='name of the dataset to load')
args = parser.parse_args()

# can be customized using OLYMPUS_BASE_PATH or Olympus configuration file if found
base = option('base_path', '/tmp/olympus')

# get the dataset
dataset = Dataset(args.dataset, path=f'{base}/data')

# How to split the dataset
splits = SplitDataset(dataset, split_method='original')

# DataLoader builder
loader = DataLoader(
    splits,
    sampler_seed=new_seed(sampler=1),
    batch_size=32
)

# Train my model
for step, batch in enumerate(loader.train()):
    print('\rTrain:', step, len(batch), end='')
print()

# Using a bigger batch size when gradient is not computed
for step, batch in enumerate(loader.valid(batch_size=1024)):
    print('\rValid:', step, len(batch), end='')
print()

for step, batch in enumerate(loader.test(batch_size=1024)):
    print('\rTest:', step, len(batch), end='')
print()

# Show all seeds that were used for this
print(get_seeds())
