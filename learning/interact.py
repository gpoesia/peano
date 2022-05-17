'Tools for interacting with models, mostly for debugging or intuition.'

#!/usr/bin/env python3

import argparse

import torch
from tqdm import tqdm
import numpy as np

from environment import *
import util
from domain import EquationsDomain
import logging


def _choose_from_list(prompt, l, to_str=str):
    print(prompt)
    for i, e in enumerate(l):
        print(f'{i:2d} - ', to_str(e))

    return l[int(input('> '))]


def _input_problem(domain):
    opt = input('a) Type problem, b) select one from list, or Enter for debug mode: ')

    if opt == 'a':
        p = input('Problem: ')
        return domain.make_problem(p)
    elif opt == 'b':
        return _choose_from_list('Pick a problem:',
                                 [domain.generate(i) for i in range(40)],
                                 lambda p: p.description)


def run_beam_search(agent_path, domain, device):
    agent = torch.load(agent_path, map_location=device)['agent']
    print('Loaded', agent_path, ':', util.format_parameter_count(agent.policy), 'parameters.')

    p = _input_problem(domain)

    if p is not None:
        episode = agent.policy.rollout(domain, p, 8)
        
    breakpoint()
    'Debug mode'


def run_best_first_search(agent_path, domain, device):
    agent = torch.load(agent_path, map_location=device)['agent']
    print('Loaded', agent_path, ':', util.format_parameter_count(agent.policy), 'parameters.')

    p = _input_problem(domain)

    if p is not None:
        episode = agent.policy.best_first_search(domain, p, 100)

    breakpoint()
    'Debug mode'


def evaluate_agent(agent_path, d, device, rollout_type):
    agent = torch.load(agent_path, map_location=device)['agent']
    print('Loaded', agent_path, ':', util.format_parameter_count(agent.policy), 'parameters.')

    succ = []
    agent.policy.eval()

    for i in tqdm(range(100)):
        problem = d.generate(seed=10**7 + i)
        if rollout_type == 'greedy':
            rollout = agent.policy.rollout(d, problem, depth=8, temperature=0.01, beam_size=1)
            print(rollout.actions)
        elif rollout_type == 'beam-search':
            rollout = agent.policy.rollout(d, problem, depth=8, temperature=0.01, beam_size=10)
            print(rollout.actions)
        elif rollout_type == 'best-first-search':
            rollout = agent.policy.best_first_search(d, problem, 100)
        succ.append(rollout.success)

        print(f'#{i} - {problem.description} - {rollout.success}')

#        if rollout.success:

    print('Success rate:', np.mean(succ))


def interact_with_environment(domain):
    i, p = 0, _input_problem(domain)
    prob = 1

    while not domain.reward(p.universe):
        actions = domain.actions(p.universe)

        print('State:', domain.state(p.universe))
        a = _choose_from_list('Arrow to apply:', actions)

        prob *= 1 / len(actions)

        outcomes = p.universe.apply(a)
        o = _choose_from_list('Result to use:', outcomes)

        prob *= 1 / len(outcomes)

        p.universe.define(f'!subd{i}', o)
        i += 1

    print('Solved in', i, 'steps!')
    print('Probability of this trajectory for a random policy:', prob)


def try_random_rollouts(env, n_problems=10**3, n_steps=30):
    solved = []

    print('Actions:', env.sample_problem(0).actions())
    actions = input('Allowed actions (default: all): ')

    actions = actions.split(',') or p.actions()

    print('Using actions', actions)

    for i in tqdm(range(n_problems)):
        p = env.sample_problem(3)
        if p.random_rollout(actions, n_steps, i):
            solved.append(p.starting_state())

    print(len(solved), 'solved:')

    for p in solved:
        print(p)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Interact with pre-trained models or the environment.')
    parser.add_argument('--eval', help='Evaluate the given agent on the test set.', action='store_true')
    parser.add_argument('--beam-search', help='Run beam search with the given agent', action='store_true')
    parser.add_argument('--best-first-search', help='Run best-first search with the given agent', action='store_true')
    parser.add_argument('--agent', help='Path to a pre-trained agent', type=str)
    parser.add_argument('--environment', help='Solve a problem manually', action='store_true')
    parser.add_argument('--random-rollouts', help='Try to solve problems using random rollouts', action='store_true')
    parser.add_argument('--gpu', help='GPU device to use.', type=int)
    parser.add_argument('--verbose', help='Use debug-level .', action='store_true')

    opt = parser.parse_args()

    env = SingleDomainEnvironment('equations')
    domain = EquationsDomain()

    device = torch.device('cpu') if not opt.gpu else torch.device(opt.gpu)

    logging.basicConfig()

    if opt.verbose:
        logging.root.setLevel(logging.DEBUG)

    if opt.eval:
        evaluate_agent(opt.agent, domain, device,
                       'beam-search' if opt.beam_search
                       else 'best-first-search' if opt.best_first_search
                       else 'greedy')
    elif opt.beam_search:
        run_beam_search(opt.agent, domain, device)
    elif opt.best_first_search:
        run_best_first_search(opt.agent, domain, device)
    elif opt.environment:
        interact_with_environment(domain)
    elif opt.random_rollouts:
        try_random_rollouts(env)
