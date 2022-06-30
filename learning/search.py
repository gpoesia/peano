#!/usr/bin/env python3

import random
import heapq
import collections
from dataclasses import dataclass, field
import pickle
from typing import Any
import logging

from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

from domain import Domain, Problem, make_domain
from utility import SearchHeuristic, GRUUtilityFunction, TwoStageUtilityFunction, LengthUtilityFunction
from episode import ProofSearchEpisode
from util import get_device

import torch

MAX_NEGATIVES = 10000
MIN_UTILITY = -70


logger = logging.getLogger(__name__)


@dataclass(order=True)
class PrioritizedDefinition:
    utility: float
    value: str = field(compare=False)
    definition: Any = field(compare=False)
    depth: int = field(compare=False)


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
                logger.debug('[%s] Utility of %s: %f', pd.definition.generating_action(), val, -pd.utility)
                if pd.utility >= MIN_UTILITY:
                    heapq.heappush(pqs[heuristic.group(o, 0)], pd)

    visited_negatives = set()

    for i in range(batch_size):
        if domain.derivation_done(problem.universe):
            break

        nonempty_pqs = [(k, v) for k, v in pqs.items() if v]

        if not nonempty_pqs:
            break

        k, pq = random.choices(nonempty_pqs, weights=[1 / (1 + cnts[k]) for k, v in nonempty_pqs])[0]
        d = heapq.heappop(pq)
        name = f'!step{idx}'
        logger.debug('Adding %s = %s from queue %s, utility %f', name, d.value, k, -d.utility)

        idx += 1
        sub_defs = problem.universe.define(name, d.definition)
        steps[name] = d.definition
        depth[name] = d.depth
        visited_negatives.add(d.value)
        order.append(name)

        new_defs = [d
                    for dname in [name] + sub_defs
                    for d in problem.universe.apply_all_with(
                            domain.derivation_actions(problem.universe), dname)]

        unseen_vals, unseen_defs = [], []

        for d in new_defs:
            val = problem.universe.value_of(d)

            if val not in seen_vals:
                seen_vals.add(val)
                unseen_defs.append(d)
                unseen_vals.append(val)

        if unseen_vals:
            utilities = heuristic.utility(problem.description, unseen_vals)

            for v, d, u in zip(unseen_vals, unseen_defs, utilities):
                if u < MIN_UTILITY:
                    continue
                next_depth = max([0] + [depth[parent] + 1 for parent in d.dependencies()])
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

        if logger.isEnabledFor(logging.DEBUG):
            print_solution(solution_defs, problem.universe)
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


@dataclass
class SearcherResults:
    episodes: list[ProofSearchEpisode]

    def successes(self):
        return sum(1 for e in self.episodes if e.success)

    def success_rate(self):
        return self.successes() / len(self.episodes)


class SearcherAgent:
    '''Agent that runs proof search on batches of problems.

    When we have a distributed setup, this will run in parallel finding
    solutions to problems, and occasionally receiving an updated trained model
    from the a TrainerAgent.
    '''

    def __init__(self, domain, utility_fn, max_depth, debug=False):
        self.domain = domain
        self.utility_fn = utility_fn
        self.max_depth = max_depth
        self.debug = debug

    def run_batch(self, seeds) -> SearcherResults:
        episodes = []
        successes = 0

        for seed in tqdm(seeds):
            problem = self.domain.generate_derivation(seed)

            if self.debug:
                r = input(f'Problem #{seed}: {problem.description} - skip? (y/n/t)')

                if r == 'y':
                    continue
                elif r == 't':
                    problem = self.domain.start_derivation(input('Problem: '), input('Goal: '))

                breakpoint()

            with torch.no_grad():
                episode = batched_forward_search(
                    self.domain,
                    problem,
                    self.utility_fn,
                    batch_size=self.max_depth,
                )

            if episode.success:
                logger.info('Solved %s', episode.problem)
                successes += 1

            episodes.append(episode)

        return SearcherResults(episodes)


def run_search_on_batch(domain, seeds, utility_fn, max_depth, output_path, debug):
    searcher = SearcherAgent(domain, utility_fn, max_depth, debug)

    result = searcher.run_batch(seeds)
    print(f'Solved {result.successes()}/{len(seeds)}')

    if output_path is not None:
        with open(output_path, 'wb') as f:
            pickle.dump(result.episodes, f)
        print('Wrote', output_path)

    return result


def run_utility_function(domain, seeds, model_path, device, output_path,
                         debug, max_depth=400, rerank_top_k=500):
    if model_path is not None:
        m = torch.load(model_path, map_location=device)
        m.to(device)
        m.eval()

        h = TwoStageUtilityFunction(LengthUtilityFunction(), m, k=rerank_top_k)
    else:
        h = LengthUtilityFunction()

    return run_search_on_batch(domain,
                        seeds,
                        h,
                        max_depth,
                        output_path,
                        debug)


@hydra.main(version_base="1.2", config_path="config", config_name="search")
def main(cfg: DictConfig):
    if cfg.task == 'solve':
        run_utility_function(make_domain(cfg.domain),
                             range(cfg.range[0], cfg.range[1]),
                             to_absolute_path(cfg.model_path) if 'model_path' in cfg else None,
                             get_device(cfg),
                             to_absolute_path(cfg.output),
                             cfg.get('debug'),
                             max_depth=400)
    else:
        raise ValueError(f'Unknown command {cfg.task}')


if __name__ == '__main__':
    main()
