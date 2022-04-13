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


class SingleDomainEnvironment(Environment):
    def __init__(self, domain: str):
        self.domain = get_domain(domain)

    def sample_problem(self, seed: int = None):
        seed = seed or random.randint(10**7, 10**8)
        return self.domain.generate(seed)
