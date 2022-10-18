#!/usr/bin/env python3

"""
Algorithm for reconstructing a curriculum from an agent's training episodes.
"""

import collections
import pickle
import json
import random

import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from tactics import Tactic
from policy import Episode
from util import count_inversions, bootstrap_mean_ci, plot_vegalite


N_OUTPUT_SECTIONS = 3


def episode_dependencies(e: Episode,
                         tactic_dependencies: dict[str, set[str]]) -> tuple[str]:
    deps = set()
    for a in e.actions[::2]:
        deps.update(tactic_dependencies.get(a, ()))
    return tuple(sorted(deps))


def compare_dependencies(d1: tuple[str], d2: tuple[str]):
    d1 = set(d1)
    d2 = set(d2)

    if d1 != d2:
        if d1.issubset(d2):
            return -1
        if d2.issubset(d1):
            return 1

    return 0

def topologically_sort_dependencies(l: list[tuple[str]]) -> list[tuple[str]]:
    random.shuffle(l)

    edges = collections.defaultdict(list)
    indegree = collections.defaultdict(int)

    for i, l_i in enumerate(l):
        for j, l_j in enumerate(l[i + 1:], i + 1):
            cmp = compare_dependencies(l_i, l_j)

            if cmp == -1:
                # i should come first.
                edges[i].append(j)
                indegree[j] += 1
            elif cmp == 1:
                edges[j].append(i)
                indegree[i] += 1

    queue = []
    layer = collections.defaultdict(int)

    for i in range(len(l)):
        if indegree[i] == 0:
            queue.append(i)
            layer[l[i]] = 0

    result = []

    while queue:
        idx = queue.pop()
        result.append(l[idx])

        for e in edges[idx]:
            indegree[e] -= 1
            layer[l[e]] = max(layer[l[e]], layer[l[idx]] + 1)
            if indegree[e] == 0:
                queue.append(e)

    return result, layer


def reconstruct_curriculum(cfg: DictConfig):
    with open(cfg.episodes, 'rb') as f:
        episodes = pickle.load(f)

    with open(cfg.tactics, 'rb') as f:
        tactics = pickle.load(f)

    sections = []
    successful_episodes = []

    for e in episodes:
        if e.success:
            sections.append(e.domain)
            successful_episodes.append(e)

    tactics_deps = compute_tactics_dependencies(tactics)
    for t in tactics:
        print(f'{t.name}: depends on {str(tactics_deps[t.name])}')

    ep_deps = [episode_dependencies(e, tactics_deps)
               for e in successful_episodes]

    episodes_by_ep_deps = collections.defaultdict(list)

    for e, deps in zip(successful_episodes, ep_deps):
        episodes_by_ep_deps[deps].append(e)

    distinct_ep_deps = list(set(ep_deps))
    print(len(distinct_ep_deps), 'distinct episode dependencies.')

    sorted_ep_deps, layers = topologically_sort_dependencies(distinct_ep_deps)

    induced_curriculum = []

    for deps in sorted_ep_deps:
        induced_curriculum.extend(episodes_by_ep_deps[deps])

    khan_academy_order = {
        'subst-eval': 0,
        'comb-like': 1,
        'one-step-add-eq': 2,
    }

    khan_academy_name = ['Substituting and Evaluating Expressions',
                         'Combining Like Terms',
                         'One-Step Addition and Subtraction Equations']

    n_induced_sections = max(layers.values()) + 1

    order = [khan_academy_order[e.domain] for e in induced_curriculum]
    inversions_mean, inversions_ci = estimate_induced_curriculum_inversions(
        distinct_ep_deps, episodes_by_ep_deps,
        lambda e: khan_academy_order[e.domain])

    baseline_mean, baseline_ci = compute_inversions_baseline(order)
    max_inversions = count_inversions(sorted(order, reverse=True))

    print('The induced curriculum has', len(order), 'elements.')
    print('The induced curriculum has', n_induced_sections, 'sections')
    print('The maximum number of inversions would be', max_inversions)
    print(f'99% CI for induced: {inversions_mean} @ {inversions_ci}')
    print(f'99% CI for baseline: {baseline_mean} @ {baseline_ci}')

    plot_data = []
    for i, o in enumerate(order):
        plot_data.append({'order': i,
                          'Curriculum': 'Induced',
                          'Section': khan_academy_name[o]})
    for i, o in enumerate(sorted(order)):
        plot_data.append({'order': i,
                          'Curriculum': 'Khan Academy',
                          'Section': khan_academy_name[o]})

    # plot_vegalite('curriculum', plot_data, cfg.curriculum_plot)

    plot_vegalite('inversions', [
        {'Curriculum': 'Random',
         'mean': baseline_mean / max_inversions,
         'lo': baseline_ci[0] / max_inversions,
         'hi': baseline_ci[1] / max_inversions},
        {'Curriculum': 'Inferred',
         'mean': inversions_mean / max_inversions,
         'lo': inversions_ci[0] / max_inversions,
         'hi': inversions_ci[1] / max_inversions},
    ], cfg.inversions_plot)

    begin = 0

    for induced_section_idx in range(N_OUTPUT_SECTIONS):
        end = begin
        problems = []

        while end < len(sorted_ep_deps):
            for episode in episodes_by_ep_deps[sorted_ep_deps[end]]:
                problems.append({'problem': episode.problem,
                                 'goal': episode.goal,
                                 'domain': episode.domain})
            end += 1
            if len(problems) > len(order) // N_OUTPUT_SECTIONS:
                break

        begin = end

        with open(cfg.output_sections.format(induced_section_idx), 'wb') as f:
            pickle.dump(problems, f)
            print('Wrote', f.name, 'with', len(problems), 'problems.')



def compute_inversions_baseline(l):
    trials = []

    for i in tqdm(range(100)):
        l2 = l[:]
        random.shuffle(l2)
        trials.append(count_inversions(l2))

    return bootstrap_mean_ci(trials, .99)


def estimate_induced_curriculum_inversions(distinct_ep_deps: list[tuple[str]],
                                           episodes_by_ep_deps: dict[tuple[str], list[Episode]],
                                           order_fn):
    trials = []

    for i in tqdm(range(100)):
        sorted_ep_deps, _ = topologically_sort_dependencies(distinct_ep_deps)

        order = []
        for deps in sorted_ep_deps:
            l = list(map(order_fn, episodes_by_ep_deps[deps]))
            random.shuffle(l)
            order.extend(l)

        trials.append(count_inversions(order))

    return bootstrap_mean_ci(trials, .99)


def compute_tactics_dependencies(tactics: list[Tactic]) -> dict[str, set[str]]:
    deps = {}
    names = {t.name for t in tactics}

    for t in tactics:
        deps_t = {t.name}
        for s in t.steps:
            arrow = s.arrow
            if arrow in names:
                deps_t.add(arrow)
                deps_t.update(deps.get(arrow, []))
        deps[t.name] = deps_t

    return deps


@hydra.main(version_base="1.2", config_path="config", config_name="curriculum")
def main(cfg: DictConfig):
    if cfg.task == 'reconstruct':
        reconstruct_curriculum(cfg)


if __name__ == '__main__':
    main()
