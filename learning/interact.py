'Tools for interacting with models, mostly for debugging or intuition.'

#!/usr/bin/env python3

import argparse

import torch
from environment import *

import util


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


def interact_with_environment(env):
    i, p = 0, _input_problem(env)
    prob = 1

    while not p.reward():
        actions = p.actions() + ['eval']
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Interact with pre-trained models or the environment.')
    parser.add_argument('--agent', help='Path to a pre-trained agent', type=str)
    parser.add_argument('--environment', help='Solve a problem manually', action='store_true')
    parser.add_argument('--gpu', help='GPU device to use.', type=int)

    opt = parser.parse_args()

    env = SingleDomainEnvironment('equations-easy')

    device = torch.device('cpu') if not opt.gpu else torch.device(opt.gpu)

    if opt.agent is not None:
        run_agent(opt.agent, env, device)
    elif opt.environment:
        interact_with_environment(env)
