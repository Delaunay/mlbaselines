from olympus.datasets.sampling.group import GroupedBatchSampler
from torch.utils.data.sampler import RandomSampler


def test_group_samples():
    samples = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

    rnd_sampler = RandomSampler(samples)
    group_sampler = GroupedBatchSampler(rnd_sampler, samples, batch_size=4)

    for sample_ids in group_sampler:

        group = samples[sample_ids[0]]
        for id in sample_ids[1:]:
            assert group == samples[id]




