#!/usr/bin/env python3

import argparse
import pickle
import collections


def analyze_pretraining(data_paths: list[str]):
    episodes = []

    for path in data_paths:
        with open(path, 'rb') as f:
            episodes.extend(pickle.load(f))

    print('Loaded', len(episodes), 'pretraining trajectories.')

    goals = [e.goal for e in episodes]
    N = 100
    top_goals = collections.Counter(goals).most_common(n=N)

    print(N, 'most common goals:')
    for g in top_goals:
        print(f'{g[0]} - {g[1]} ({100 * g[1] / len(goals):.3f}%)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Analyze generated data.')

    parser.add_argument('--pretraining', action='store_true',
                        help='Analyze generated pretraining trajectories.')
    parser.add_argument('--data', type=str, nargs='+', required=False,
                        help='Path to the data files.')

    opt = parser.parse_args()

    if opt.pretraining:
        analyze_pretraining(opt.data)
