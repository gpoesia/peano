#!/usr/bin/env python3

import argparse
import pickle
import collections
import os

import hydra
from omegaconf import DictConfig

from util import plot_vegalite


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


def plot_learning_curves(cfg: DictConfig):
    plot_data = []

    khan_academy_order = {
        'subst-eval': 0,
        'comb-like': 1,
        'one-step-add-eq': 2,
    }

    khan_academy_name = ['Substituting and Evaluating Expressions',
                         'Combining Like Terms',
                         'One-Step Addition and Subtraction Equations']

    for agent in cfg.agents:
        for domain in cfg.domains:
            for i in range(cfg.iterations):
                with open(os.path.join(agent.root,
                          f'eval-episodes-{domain}-{i}.pkl'), 'rb') as f:
                    episodes = pickle.load(f)
                success_rate = sum(1 for e in episodes if e.success) / len(episodes)

                plot_data.append({
                    'Agent': agent.name,
                    'Domain': khan_academy_name[khan_academy_order[domain]],
                    'domain_order': khan_academy_order[domain],
                    'Iteration': i + 1,
                    'Success Rate': success_rate,
                    })

    print(len(plot_data), 'data points.')

    plot_vegalite('learning-curves', plot_data, cfg.output)

@hydra.main(version_base="1.2", config_path="config", config_name="analysis")
def main(cfg: DictConfig):
    if cfg.task == 'learning-curves':
        plot_learning_curves(cfg.learning_curves)


if __name__ == '__main__':
    main()
