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


class Tactic:
    '''Represents a high-level derivation sketch that operates on proof terms.
    What makes a tactic a "sketch" is that it can have holes in arguments,
    which is represented by arguments with a name starting with '?'.
    '''

    def __init__(self, steps: list[Step]):
        self.steps = steps

    def unify(self, t: 'Tactic') -> Optional['Tactic']:
        'Returns a tactic that generalizes self and t, if possible.'
        raise NotImplementedError()

    def execute(self, u: peano.PyUniverse) -> list[peano.PyDefinition]:
        'Executes the tactic on the given universe and returns all results it is able to generate.'
        raise NotImplementedError()
