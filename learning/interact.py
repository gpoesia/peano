'Tools for interacting with models, mostly for debugging or intuition.'

#!/usr/bin/env python3

import argparse

import torch
from environment import *

import util


def run_agent(agent_path, env, device):
    agent = torch.load(agent_path, map_location=device)['agent']
    print('Loaded', agent_path, ':', util.format_parameter_count(agent.policy), 'parameters.')

    problems = [env.sample_problem(i) for i in range(20)]

    print('Choose a problem:')
    for i, p in enumerate(problems):
        print(f'{i:2d} - ', p.starting_state())

    p = problems[int(input('> '))]

    breakpoint()

    agent.policy.rollout(p, 5)

    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Interact with pre-trained models or the environment.')
    parser.add_argument('--agent', help='Path to a pre-trained agent', type=str)
    parser.add_argument('--gpu', help='GPU device to use.', type=int)

    opt = parser.parse_args()

    env = SingleDomainEnvironment('equations-easy')

    device = torch.device('cpu') if not opt.gpu else torch.device(opt.gpu)

    if opt.agent is not None:
        run_agent(opt.agent, env, device)
