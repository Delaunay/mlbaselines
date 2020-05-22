import torch
from collections import defaultdict
from itertools import zip_longest
import numpy as np


def dict_key(k1, k2):
    if k1 is None:
        return k2
    return f'{k1}.{k2}'


def compare_states(d1, d2, depth=0, key=None):
    if type(d1) != type(d2):
        return f'{key} Type mismatch ({type(d1)} != {type(d2)})', False

    elif isinstance(d1, (dict, defaultdict, )):
        keys = set(d1.keys())
        keys.update(set(d2.keys()))
        keys = list(keys)

        diffs = []
        all_match = True

        for k in keys:
            v1 = d1.get(k)
            v2 = d2.get(k)

            d, m = compare(v1, v2, depth + 1, dict_key(key, k))

            all_match &= m
            if not m:
                diffs.append(d)

        return ('\n' + '  ' * depth).join(diffs), all_match

    elif isinstance(d1, (list, tuple)):
        diffs = []
        all_match = True

        for i, (v1, v2) in enumerate(zip_longest(d1, d2)):
            d, m = compare(v1, v2, depth + 1, dict_key(key, i))
            all_match &= m

            if not m:
                diffs.append(d)

        return ('\n' + '  ' * depth).join(diffs), all_match

    elif isinstance(d1, str):
        return f'{key} {d1} {d2}', d1 == d2

    elif isinstance(d1, (float, int)):
        d = float(d1 - d2)
        return f'{key}: {d:15f}', d < 1e-5

    elif isinstance(d1, torch.Tensor):
        d = float((d1 - d2).abs().sum().item())
        return f'{key}: {d:15f}', d < 1e-5

    elif isinstance(d1, np.ndarray):
        d = float(np.sum(np.abs(d1 - d2)))
        return f'{key}: {d:15f}', d < 1e-5
    else:
        return f'{type(d1)}, {d1}', d1 == d2


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('files', type=str, nargs=2, help='State files to compare')
    args = parser.parse_args()

    f1, f2 = args.files

    d1 = torch.load(f1)
    d2 = torch.load(f2)

    diffs, m = compare(d1, d2)

    print(diffs)


if __name__ == '__main__':
    p1 = '/home/setepenre/Downloads/segmentation/segstates/8a2ee39d8aec0e350bb7581e28796e5a8c6b0ae9b24c6c4c66ba7232ceba9bbf-1-1.state'
    p2 = '/home/setepenre/Downloads/segmentation/segstates/8a2ee39d8aec0e350bb7581e28796e5a8c6b0ae9b24c6c4c66ba7232ceba9bbf-1-2.state'

    d1 = torch.load(p1)
    d2 = torch.load(p2)

    diffs, m = compare(d1, d2)

    print(diffs)

