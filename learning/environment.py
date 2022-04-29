#!/usr/bin/env python3

import random

from peano import PyUniverse as Universe, PyDefinition as Definition, get_domain


class Environment:
    'Generic environment back-end'
    def sample_problem(self, seed: int = None) -> Universe:
        raise NotImplementedError()

    @staticmethod
    def from_config(config: dict):
        'Returns the appropriate environment given the experiment configuration options.'
        return SingleDomainEnvironment(config.get('domain'))

    def format_state(self, universe, ignore):
        return '; '.join(f'{{{"=".join(set(vals))}}} : {dtype}'
                         for vals, dtype in universe.state(ignore))


class SingleDomainEnvironment(Environment):
    def __init__(self, domain: str):
        self.domain = get_domain(domain)

    def sample_problem(self, seed: int = None):
        if seed is None:
            seed = random.randint(10**7, 10**8)
        return self.domain.generate(seed)


def interact(domain):
    env = SingleDomainEnvironment(domain)
    p = env.sample_problem()

    print('Problem:', p.starting_state())

    while not p.reward():
        actions = p.actions() + ['eval']
        print('Actions:', list(enumerate(actions)))

        a = int(input('Choose action: '))
        defs = p.apply(actions[a])

        print('Outcomes:', list(enumerate(map(str, defs))))

        o = int(input('Outcome: '))
        p.define('r', defs[o])

        print('Reward?', p.reward())

    print('Solved!')


if __name__ == '__main__':
    interact('equations-easy')
