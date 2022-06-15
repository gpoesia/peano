#!/usr/bin/env python3

import random
import heapq
import collections
from dataclasses import dataclass, field
import pickle
from typing import Any

from tqdm import tqdm

from domain import Domain, Problem, make_domain
from utility import SearchHeuristic, GRUUtilityFunction, TwoStageUtilityFunction, LengthUtilityFunction
import torch

MAX_NEGATIVES = 10000


@dataclass(order=True)
class PrioritizedDefinition:
    utility: float
    value: str = field(compare=False)
    definition: Any = field(compare=False)
    depth: int = field(compare=False)

@dataclass
class ProofSearchEpisode:
    success: bool
    iterations: int
    steps_added: int
    steps_created: int
    problem: str
    solution: list[str]
    visited_negatives: list[str]
    discovered_negatives: list[str]


def print_solution(steps, derivation):
    for i, s in enumerate(steps):
        print(f'{i+1:02d}.', 'show' if derivation.is_prop(s) else 'construct',
              derivation.value_of(s),
              'by',
              s.generating_action())


def recover_solution(steps, goal, order):
    if not order:
        return []

    reached = {goal}
    stack = [goal]

    while stack:
        step = stack.pop()

        for d in steps[step].dependencies():
            if d in steps and d not in reached:
                reached.add(d)
                stack.append(d)

    solution = []

    for s in order:
        if s in reached:
            solution.append(steps[s])

    return solution


def batched_forward_search(domain: Domain,
                           problem: Problem,
                           heuristic: SearchHeuristic,
                           batch_size: int,
                           max_batches: int=100):
    seen_vals = set()
    pqs = collections.defaultdict(list)
    cnts = collections.defaultdict(int)
    depth = collections.defaultdict(int)
    steps = {}
    order = []

    idx = 0

    for action in domain.derivation_actions(problem.universe):
        outcomes = problem.universe.apply(action)

        for o in outcomes:
            val = problem.universe.value_of(o)
            if val not in seen_vals:
                seen_vals.add(val)
                pd = PrioritizedDefinition(-heuristic.utility(problem.description, [val])[0],
                                           value=val,
                                           definition=o,
                                           depth=0)
                heapq.heappush(pqs[heuristic.group(o, 0)], pd)

    visited_negatives = set()

    for i in range(batch_size):
        if domain.derivation_done(problem.universe):
            break

        nonempty_pqs = [(k, v) for k, v in pqs.items() if v]

        if not nonempty_pqs:
            break

        k, pq = random.choices(nonempty_pqs, weights=[1 / (1 + cnts[k]) for k, _ in nonempty_pqs])[0]
        d = heapq.heappop(pq)
        name = f'!step{idx}'
        idx += 1
        sub_defs = problem.universe.define(name, d.definition)
        steps[name] = d.definition
        depth[name] = d.depth
        visited_negatives.add(d.value)
        order.append(name)

        new_defs = [d
                    for dname in [name] + sub_defs
                    for d in problem.universe.apply_all_with(dname)]

        unseen_vals, unseen_defs = [], []

        for d in new_defs:
            val = problem.universe.value_of(d)

            if val not in seen_vals:
                unseen_defs.append(d)
                unseen_vals.append(val)

        if unseen_vals:
            utilities = heuristic.utility(problem.description, unseen_vals)

            for v, d, u in zip(unseen_vals, unseen_defs, utilities):
                next_depth = max([0] + [depth[parent] + 1 for parent in d.dependencies()])
                seen_vals.add(val)
                pd = PrioritizedDefinition(-u,
                                           value=v,
                                           definition=d,
                                           depth=next_depth)
                heapq.heappush(pqs[heuristic.group(d, next_depth)], pd)

    goal = domain.derivation_done(problem.universe)

    discovered_negatives = [neg.value for _, vals in pqs.items() for neg in vals]
    discovered_negatives = random.sample(discovered_negatives,
                                         k=min(len(discovered_negatives), MAX_NEGATIVES))

    if goal:
        solution_defs = recover_solution(steps, goal, order)
        solution = [problem.universe.value_of(s) for s in solution_defs]
        visited_negatives -= set(solution)
        # print_solution(solution_defs, problem.universe)
    else:
        solution = None

    return ProofSearchEpisode(
        success=domain.derivation_done(problem.universe),
        iterations=i,
        steps_added=idx,
        steps_created=len(seen_vals),
        problem=problem.description,
        solution=solution,
        visited_negatives=visited_negatives,
        discovered_negatives=discovered_negatives,
    )


def test_search_heuristic_hyperparams(name, group_fn, depth_weight, max_depth=400):
    domain = make_domain('equations')

    problems = [
        '(= (+ (- 3 3) x) (* 2 4))',
        '(= (+ 0 x) 3)',
        '(= (+ 0 (+ 0 x)) 3)',
        '(= (+ 0 (+ 0 (+ 0 x))) 3)',
        '(= (+ 0 (+ 0 (+ 0 (+ 0 x)))) 3)',
        '(= (+ x 0) 3)',
        '(= (+ (+ (+ x 0) 0) 0) 3)',
        '(= (+ x (* 0 x)) (- x x))',
        '(= x (+ 1 (+ 2 (+ 3 4))))',
        '(= (+ x 1) 3)',
    ]

    successes = []
    nodes = []

    print(f'{name:30}:', end=' ', flush=True)

    for eq in problems:
        p = domain.start_derivation(eq, '(= x ?)')
        episode = batched_forward_search(
            domain,
            p,
            group_fn,
            utility=lambda val, depth: (depth_weight * depth - len(val)),
            batch_size=max_depth,
        )

        successes.append(episode.success)
        if episode.success:
            nodes.append(episode.steps_added)

        print('.' if episode.success else 'X', end='', flush=True)

    print('    Mean/Max', f'{sum(nodes) // len(problems):4} / {max(nodes):4}')


def run_search_on_batch(domain, seeds, heuristic, max_depth, output_path):
    episodes = []
    successes = 0
    steps = 0

    for seed in tqdm(seeds):
        problem = domain.generate_derivation(seed)
        episode = batched_forward_search(
            domain,
            problem,
            heuristic,
            batch_size=max_depth,
        )

        if episode.success:
            print('Solved', episode.problem)
            successes += 1

        episodes.append(episode)

        steps += 1
        if steps % 100 == 0:
            with open(output_path, 'wb') as f:
                pickle.dump(episodes, f)

    with open(output_path, 'wb') as f:
        pickle.dump(episodes, f)

    print(f'Solved {successes}/{len(seeds)}, wrote {output_path}')


def run_bootstrap_step(domain, seeds, output_path):
    run_search_on_batch(domain,
                        seeds,
                        LengthUtilityFunction(),
                        400,
                        output_path)

def run_trained_utility_function(domain, seeds, model_path, device, output_path):
    m = torch.load(model_path, map_location=device)
    m.to(device)
    m.eval()

    h = TwoStageUtilityFunction(LengthUtilityFunction(), m, k=200)

    run_search_on_batch(domain,
                        seeds,
                        h,
                        400,
                        output_path)


if __name__ == '__main__':
    # run_bootstrap_step(make_domain('equations'), range(10000), 'bootstrap_episodes.pkl')
    run_trained_utility_function(make_domain('equations'),
                                 range(1),
                                 '1.pt',
                                 torch.device('cpu'),
                                 'episodes-it1.pkl')
    # test_search_heuristic_hyperparams('Bare', lambda action, depth: '', 0)
    # test_search_heuristic_hyperparams('Group by action', lambda action, depth: action, 0)
    # test_search_heuristic_hyperparams('Group by action + depth', lambda action, depth: (action, depth), 0)
    # test_search_heuristic_hyperparams('Group by action, beta=1', lambda action, depth: action, 1)
    # test_search_heuristic_hyperparams('Group by action, beta=2', lambda action, depth: action, 2)
