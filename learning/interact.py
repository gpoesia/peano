'Tools for interacting with models, mostly for debugging or intuition.'

#!/usr/bin/env python3

import argparse
import logging

import torch
from tqdm import tqdm
import numpy as np

from environment import *
import util
from domain import EquationsDomain, make_domain
from policy import encode_batch, decode_batch, EOS
from search import batched_forward_search


def _choose_from_list(prompt, l, to_str=str):
    print(prompt)
    for i, e in enumerate(l):
        print(f'{i:2d} - ', to_str(e))

    return l[int(input('> '))]


def _input_problem(domain, derivation=False):
    opt = input('a) Type problem, b) select one from list, or Enter for debug mode: ')

    if opt == 'a':
        p = input('Problem: ')
        g = input('Goal: ')
        return (domain.make_problem
                if not derivation
                else domain.start_derivation)(p, g)

    generate = domain.generate if not derivation else domain.generate_derivation
    return _choose_from_list('Pick a problem:',
                             [generate(i) for i in range(40)],
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


def make_derivation(domain):
    i, p = 0, _input_problem(domain, derivation=True)

    while not domain.derivation_done(p.universe):
        actions = domain.derivation_actions(p.universe)

        print('Derivation so far:', domain.derivation_state(p.universe))
        a = _choose_from_list('Arrow to apply:', actions)

        outcomes = p.universe.apply(a)
        o = _choose_from_list('Result to use:', outcomes)

        p.universe.define(f'!subd{i}', o)
        i += 1


def run_proof_search(domain):
    i, p = 0, _input_problem(domain, derivation=True)

    episode = batched_forward_search(
        domain,
        p,
        utility=lambda val: 1 / len(val) + random.random() * 0.001,
        batch_size=10000,
        max_per_type=1
    )

    print('Success?', episode.success)
    print('Depth:', episode.iterations)
    print('Nodes discovered:', episode.steps_created)
    print('Nodes added:', episode.steps_added)


def interact_with_policy(policy_path, domain, device):
    if policy_path is not None:
        policy = torch.load(policy_path, map_location=device)
        policy.eval()
    else:
        from policy import RandomPolicy
        policy = RandomPolicy()

    try:
        with torch.no_grad():
            while True:
                i, p = 0, _input_problem(domain)

                while not domain.reward(p.universe):
                    actions = domain.actions(p.universe)

                    state = domain.state(p.universe)
                    print('State:', state, '\tGoal:', p.goal)

                    scores = policy.score_arrows(actions, state, p.goal).tolist()
                    action_scores = list(zip(actions, scores))
                    action_scores.sort(key=lambda a_s: -a_s[1])

                    a = _choose_from_list('Arrow to apply:',
                                          action_scores,
                                          to_str=lambda a_s: f'[{a_s[1]:.3f}]  {a_s[0]}')[0]

                    outcomes = p.universe.apply(a)
                    scores = policy.score_outcomes(list(map(lambda o: o.clean_str(p.universe), outcomes)),
                                                   a,
                                                   state,
                                                   p.goal).tolist()
                    outcome_scores = list(zip(outcomes, scores))
                    outcome_scores.sort(key=lambda o_s: -o_s[1])

                    o = _choose_from_list('Result to use:',
                                          outcome_scores,
                                          to_str=lambda o_s: f'[{o_s[1]:.3f}]  {o_s[0].clean_str(p.universe)}')[0]

                    p.universe.define(f'!subd{i}', o)
                    i += 1
    except KeyboardInterrupt:
        pass


def generate_from_policy(policy_path, domain, device):
    policy = torch.load(policy_path, map_location=device)
    policy.eval()

    while True:
        p = input('Prefix: ')

        result = policy.lm.generate(encode_batch([p] * 5, device,
                                                 bos=True, eos=False),
                                                 max_length=500,
                                                 eos_token_id=EOS)

        Y = decode_batch(result)

        for s in Y:
            print(s)

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
    parser.add_argument('--generate', help='Query the language model to generate.', action='store_true')
    parser.add_argument('--beam-search', help='Run beam search with the given agent', action='store_true')
    parser.add_argument('--best-first-search', help='Run best-first search with the given agent', action='store_true')
    parser.add_argument('--agent', help='Path to a pre-trained agent', type=str)
    parser.add_argument('--environment', help='Solve a problem manually', action='store_true')
    parser.add_argument('--derivation', help='Solve a problem manually', action='store_true')
    parser.add_argument('--proof-search', help='Run proof seearch on a problem', action='store_true')
    parser.add_argument('--domain', help='Which domain to use.', type=str, default='equations')
    parser.add_argument('--policy', help='Interact with a pre-trained policy', action='store_true')
    parser.add_argument('--random-rollouts', help='Try to solve problems using random rollouts', action='store_true')
    parser.add_argument('--gpu', help='GPU device to use.', type=int)
    parser.add_argument('--verbose', help='Use debug-level .', action='store_true')

    opt = parser.parse_args()

    domain = make_domain(opt.domain)

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
    elif opt.derivation:
        make_derivation(domain)
    elif opt.proof_search:
        run_proof_search(domain)
    elif opt.random_rollouts:
        env = SingleDomainEnvironment('equations')
        try_random_rollouts(env)
    elif opt.policy:
        interact_with_policy(opt.agent, domain, device)
    elif opt.generate:
        generate_from_policy(opt.agent, domain, device)
