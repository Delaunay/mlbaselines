from olympus.reinforcement.dataloader import RLDataloader

import gym
import os


def file_descriptor_count():
    return len(os.listdir(os.path.join('/proc', str( os.getpid()), 'fd')))


def scoped_init():
    def to_nchw(states):
        return states.permute(0, 3, 1, 2)

    loader = RLDataloader(
        12,       # Number of parallel simulations
        24,          # Max number of steps in a simulation
        to_nchw,            # transform state
        gym.make,
        'SpaceInvaders-v0'
    )
    after_init = file_descriptor_count()
    loader.close()

    return after_init


def test_dataloader_does_clean_up(num=10):
    fd_before = file_descriptor_count()

    fd_while = 0
    for _ in range(num):
        fd_while += scoped_init()

    fd_after = file_descriptor_count()

    # print('before', fd_before)
    # print('while', fd_while)
    # print('after', fd_after)
    # print('remaining', (fd_after - fd_before))

    assert fd_after <= fd_before + 1


if __name__ == '__main__':
    test_dataloader_does_clean_up()



