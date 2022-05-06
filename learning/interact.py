'Tools for interacting with models, mostly for debugging or intuition.'

#!/usr/bin/env python3

import argparse

import torch
from tqdm import tqdm
import numpy as np

from environment import *
import util
from domain import EquationsDomain


def _choose_from_list(prompt, l, to_str=str):
    print(prompt)
    for i, e in enumerate(l):
        print(f'{i:2d} - ', to_str(e))

    return l[int(input('> '))]


def _input_problem(env):
    return _choose_from_list('Pick a problem:',
                             [env.sample_problem(i) for i in range(40)],
                             lambda p: p.starting_state())


def run_agent(agent_path, env, device):
    agent = torch.load(agent_path, map_location=device)['agent']
    print('Loaded', agent_path, ':', util.format_parameter_count(agent.policy), 'parameters.')

    p = _input_problem(env)
    breakpoint()

    agent.policy.rollout(p, 5)

    print('Done.')


def evaluate_agent(agent_path, d, device):
    agent = torch.load(agent_path, map_location=device)['agent']
    print('Loaded', agent_path, ':', util.format_parameter_count(agent.policy), 'parameters.')

    succ = []
    agent.policy.eval()

    for i in tqdm(range(100)):
        problem = d.generate(seed=10**7 + i)
        rollout = agent.policy.rollout(d, problem, depth=8, temperature=0.01)
        succ.append(rollout.success)

        print(f'#{i} - {problem.description} - {rollout.success}')

#        if rollout.success:
        print(rollout.actions)

    print('Success rate:', np.mean(succ))


def interact_with_environment(env):
    i, p = 0, _input_problem(env)
    prob = 1

    breakpoint()

    while not p.reward():
        actions = p.actions()
        print('State:', env.format_state(p, set(actions + ['real'])))
        a = _choose_from_list('Arrow to apply:', actions)

        prob *= 1 / len(actions)

        outcomes = p.apply(a)
        o = _choose_from_list('Result to use:', outcomes)

        prob *= 1 / len(outcomes)

        p.define(f'!sub{i}', o)
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
    parser.add_argument('--eval', help='Evaluate the given agent.', action='store_true')
    parser.add_argument('--agent', help='Path to a pre-trained agent', type=str)
    parser.add_argument('--environment', help='Solve a problem manually', action='store_true')
    parser.add_argument('--random-rollouts', help='Try to solve problems using random rollouts', action='store_true')
    parser.add_argument('--gpu', help='GPU device to use.', type=int)

    opt = parser.parse_args()

    env = SingleDomainEnvironment('equations')

    device = torch.device('cpu') if not opt.gpu else torch.device(opt.gpu)

    if opt.eval:
        evaluate_agent(opt.agent, EquationsDomain(), device)
    elif opt.agent is not None:
        run_agent(opt.agent, env, device)
    elif opt.environment:
        interact_with_environment(env)
    elif opt.random_rollouts:
        try_random_rollouts(env)
