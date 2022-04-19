#!/usr/bin/env python3


import collections
import itertools
import logging
from dataclasses import dataclass
import random

import numpy as np
import torch
from tqdm import tqdm

from environment import Environment
from policy import Policy, DecisionTransformer, RandomPolicy, encode_batch, PAD


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

        self.depth = 1
        self.training_successes = []
        self.examples = []
        self.training_problems_solved = 0
        self.optimize_every = 16

    def name(self) -> str:
        return 'ReConPoLe'

    def learn_from_environment(self, environment):
        for i in tqdm(range(10000)):
            problem = environment.sample_problem(seed=i)
            rollout = self.policy.rollout(problem, depth=self.depth)

            if rollout.success:
                print(i, problem.starting_state(), 'solved!')
                self.training_problems_solved += 1

                if self.training_problems_solved % self.optimize_every == 0:
                    self.optimize()

                # FIXME(gpoesia) add examples to training buffer

            self.training_successes.append(rollout.success)

        print(self.training_problems_solved)

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


@LearningAgent.register
class LMPolicyLearning(LearningAgent):
    "Agent that learns to discriminate positive/negative examples of actions to take."

    def __init__(self, policy, config):
        self.policy = policy

        assert hasattr(policy, 'lm'), \
            'LMPolicyLearning only supports LM-based policies.'

        self.config = config

        self.depth = config['depth']
        self.training_successes = []
        self.examples = collections.deque(maxlen=config['max_examples'])
        self.training_problems_solved = 0
        self.optimize_every = config['optimize_every']
        self.optimizer = torch.optim.Adam(policy.lm.parameters())
        self.batch_size = config['batch_size']
        self.eval_every = config['eval_every']

        self.n_evals = 0

    def name(self) -> str:
        return 'LMPolicy'

    def learn_from_environment(self, env: Environment):
        for i in tqdm(range(self.config['episodes'])):
            if i % self.eval_every == 0:
                self.eval(env)

            self.policy.eval()
            self.policy.lm.eval()

            problem = env.sample_problem(seed=i)
            rollout = self.policy.rollout(problem, depth=self.depth)

            if rollout.success:
                print(i, problem.starting_state(), 'solved!')
                self.training_problems_solved += 1

                self.examples.append(rollout)

                if self.training_problems_solved % self.optimize_every == 0:
                    self.optimize()

            self.training_successes.append(rollout.success)

        print(self.training_problems_solved)

    def get_policy(self):
        return self.policy

    def stats(self):
        return "{} solutions found, {:.2f} training acc".format(
            np.sum(self.training_successes),
            np.mean(self.training_successes[-100:] or [0]))

    def optimize(self):
        if not self.examples:
            return

        self.policy.train()

        for _ in range(self.config['gradient_steps']):
            batch = random.choices(self.examples, k=self.batch_size)
            t = encode_batch([e.format() for e in batch],
                             self.policy.lm.device)

            X = t[:, :-1]
            y = t[:, 1:]

            X = self.policy.pad_train_batch(X)
            y = self.policy.pad_train_batch(y)

            # Do not count PAD tokens in the loss
            # (-100 is the mask ID from the huggingface API).
            y[y == PAD] = -100

            output = self.policy.lm(X, labels=y)
            self.optimizer.zero_grad()
            output.loss.backward()
            self.optimizer.step()

    def eval(self, env: Environment):
        succ = []

        print('Evaluating...')
        self.policy.eval()

        for i in tqdm(range(self.config['eval_problems'])):
            problem = env.sample_problem(seed=10**7 + i)
            rollout = self.policy.rollout(problem, depth=self.depth)
            succ.append(rollout.success)

        acc = np.mean(succ)
        print('Accuracy:', acc)

        torch.save({
            'agent': self,
            'accuracy': acc,
            }, f'lm.{self.n_evals}.pt')

        self.n_evals += 1


if __name__ == '__main__':
    import environment
    e = environment.SingleDomainEnvironment('equations-easy')
    arrows = e.sample_problem(0).actions() + ['eval']

    pi = DecisionTransformer({}, arrows)

    agent = LMPolicyLearning(pi, {})

    agent.learn_from_environment(e)
