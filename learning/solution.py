#!/usr/bin/env python3

import unittest

from typing import Optional
from dataclasses import dataclass, field
import itertools

import peano
from domain import Domain, make_domain

@dataclass
class Action:
    kind: str  # arrow or result
    value: str

    arrow: Optional[str] = None
    arguments: Optional[list[str]] = None

    definitions: Optional[peano.PyDefinition] = None

    def __str__(self):
        return self.value

@dataclass
class Solution:
    'Represents a step-by-step solution'

    problem: str
    goal: str
    success: bool = False
    results: list[str] = field(default_factory=list)
    arguments: list[str] = field(default_factory=list)
    subdefinitions: list[str] = field(default_factory=list)
    actions: list[(str, list[str])] = field(default_factory=list)
    derivation: peano.PyDerivation = None
    score: float = None

    @staticmethod
    def from_problem(problem):
        return Solution(problem.description, problem.goal, derivation=problem.universe)

    @staticmethod
    def states_from_episode(problem: str, goal: str, actions: list[str],
                            max_len: int = 200):
        states = []
        solution = Solution(problem, goal)

        states.append(solution.format(max_len))

        for i in range(len(actions)):
            if i % 2 == 0:
                solution.actions.append((actions[i], None))
            else:
                solution.results.append(actions[i])

            states.append(solution.format(max_len))

        return states

    def push_action(self, action, domain):
        if action.kind == 'arrow':
            return Solution(self.problem,
                            self.goal,
                            self.success,
                            results=self.results,
                            subdefinitions=self.subdefinitions,
                            actions=self.actions + [(action.arrow, None)],
                            derivation=self.derivation)
        elif action.kind == 'result':
            if action.definitions is None:
                derivation = self.derivation
                subdefs = []
            else:
                derivation = self.derivation.clone()
                subdefs = domain.define(derivation, f'!step{len(self.results)}', action)

            return Solution(self.problem,
                            self.goal,
                            self.success,
                            results=self.results + [action.value],
                            subdefinitions=self.subdefinitions + [subdefs],
                            arguments=self.arguments + [action.arguments],
                            actions=self.actions,
                            derivation=derivation)
        else:
            raise ValueError(f'Invalid action kind {action.kind}')

    def __lt__(self, rhs):
        return self.score < rhs.score

    def format(self, max_len: int = 1000):
        lines = [self.problem]

        for d, (a, _) in itertools.zip_longest(self.results, self.actions):
            lines.append(f'{a}:-{d or "###"}')

        s = '\n'.join(lines)

        if max_len is not None and len(s) > max_len:
            s = f'...{s[-max_len:]}'

        return f'G:{self.goal}\n' + s

    def _is_action_chosen(self):
        '''Returns True if the action (arrow) to be applied has been chosen at this state.

        If that's the case, then the next step is to choose an outcome. Otherwise, it
        returns False and the policy has to next choose the action.'''
        return len(self.actions) > len(self.results)

    def successors(self, domain: Domain):
        tactics = domain.tactic_actions()
        actions = domain.derivation_actions(self.derivation) + tactics

        if not self._is_action_chosen():
            return [Action(kind='arrow', arrow=a, value=a) for a in actions]
        elif self.actions[-1][0] in tactics:
            tactic = domain.get_tactic(self.actions[-1][0])
            traces = tactic.execute(self.derivation, domain)
            if not traces:
                return [Action(kind='result', definitions=None, value='_')]
            return [Action(kind='result',
                           definitions=t.definitions,
                           value=domain.value_of(t.universe, t),
                           arguments=t.generating_arguments())
                    for t in traces]
        else:
            results = self.derivation.apply(self.actions[-1][0])
            if not results:
                return [Action(kind='result', definitions=None, value='_')]
            return [Action(kind='result',
                           definitions=[('!result', d)],
                           value=self.derivation.value_of(d),
                           arguments=d.generating_arguments())
                    for d in results]

class SolutionTest(unittest.TestCase):
    def test_successors(self):
        d = make_domain('comb-like')
        p = d.generate_derivation(0)
        sol = Solution.from_problem(p)

        # To debug, uncomment / insert more of the following:
        # print('Solution so far:')
        # print(sol.format())

        # Choose */_assoc_l
        a = [a for a in sol.successors(d) if a.arrow == '*/_assoc_r'][0]
        sol = sol.push_action(a, d)

        sol = sol.push_action(sol.successors(d)[0], d)
