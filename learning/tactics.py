#!/usr/bin/env python3

from typing import Optional
from dataclasses import dataclass

import peano


@dataclass
class Step:
    'Represents one step in a tactic.'
    arrow: str
    arguments: list[str]
    result: str

    def rewrite(self, before: str, after: str):
        'Replace all occurrences of an argument by another.'
        return Step(self.arrow,
                    [after if x == before else x for x in self.arguments],
                    self.result)


class Tactic:
    '''Represents a high-level derivation sketch that operates on proof terms.
    What makes a tactic a "sketch" is that it can have holes in arguments,
    which is represented by arguments with a name starting with '?'.
    '''

    def __init__(self, steps: list[Step]):
        self.steps = steps

    def generalize(self, t: 'Tactic') -> Optional['Tactic']:
        'Returns a tactic that generalizes self and t, if possible.'

        # Sketch:
        # 1- If any step applies a different arrow, return None
        # 2- For each step, try to unify their arguments.
        #    - Two equal arguments => identity
        #    - Two different concrete arguments ==> make a new parameter
        #    - Two different parameters: return None

        raise NotImplementedError()

    def execute(self, u: peano.PyUniverse) -> list[peano.PyDefinition]:
        'Executes the tactic on the given universe and returns all results it is able to generate.'

        traces = [[]]

        for s in self.steps:
            for t in traces:
                # 1- Execute the step.
                # 2- For each valid result, make a new trace.
                pass

        return filter(None, traces)

    def _execute_step(self, u: peano.PyUniverse, trace: list[peano.PyDefinition]):
        raise NotImplementedError()
