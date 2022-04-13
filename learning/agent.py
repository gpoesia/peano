#!/usr/bin/env python3


import itertools
import logging
from dataclasses import dataclass

import numpy as np
import torch

from environment import Environment
from policy import Policy, GRUPolicy


class LearningAgent:
    '''Algorithm that guides learning via interaction with the enviroment.
    Gets to decide when to start a new problem, what states to expand, when to take
    random actions, etc.

    Any learning algorithm can be combined with any Recurrent Policy.
    '''

    subtypes: dict = {}

    def learn_from_environment(self, e: Environment):
        "Lets the agent learn by interaction using any algorithm."
        raise NotImplementedError()

    def learn_from_experience(self):
        "Lets the agent optionally learn from its past interactions one last time before eval."

    def stats(self):
        "Returns a string with learning statistics for this agent, for debugging."
        return ""

    def get_policy(self) -> Policy:
        "Returns a Policy that encodes the current learned model."
        raise NotImplementedError()

    @staticmethod
    def new(p: Policy, config: dict):
        return LearningAgent.subtypes[config['type']](p, config)

    @staticmethod
    def register(subclass):
        LearningAgent.subtypes[subclass.__name__] = subclass
        return subclass


@LearningAgent.register
class RecurrentContrastivePolicyLearning(LearningAgent):
    "Agent that learns to discriminate positive/negative examples of actions to take."

    def __init__(self, policy, config):
        self.policy = policy
        self.config = config

        self.depth = 10
        self.training_successes = []
        self.examples = []
        self.training_problems_solved = 0
        self.optimize_every = 16

    def name(self) -> str:
        return 'ReConPoLe'

    def learn_from_environment(self, environment):
        for i in itertools.count():
            print(i)
            problem = environment.sample_problem()
            rollout = self.policy.rollout(problem, depth=self.depth)

            if rollout.success:
                print('Problem solved!!!', problem.starting_state())
                self.training_problems_solved += 1

                if self.training_problems_solved % self.optimize_every == 0:
                    self.optimize()

                # FIXME(gpoesia) add examples to training buffer

            self.training_successes.append(rollout.success)

    def get_policy(self):
        return self.policy

    def stats(self):
        return "{} solutions found, {:.2f} training acc".format(
            np.sum(self.training_successes),
            np.mean(self.training_successes[-100:]))

    def optimize(self):
        if not self.examples:
            return

        # FIXME(gpoesia) Run gradient steps

        raise NotImplementedError()

if __name__ == '__main__':
    import environment
    e = environment.SingleDomainEnvironment('equations')
    arrows = e.sample_problem(0).actions()

    pi = GRUPolicy({}, arrows)

    agent = RecurrentContrastivePolicyLearning(pi, {})

    agent.learn_from_environment(e)
