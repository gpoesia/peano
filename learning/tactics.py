#!/usr/bin/env python3

from typing import Optional
from dataclasses import dataclass
import unittest

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

@dataclass
class Trace:
    assignments: dict[str, str]
    universe: peano.PyDerivation
    definitions: list[tuple[str, peano.PyDefinition]]
    subdefs: list[str]


class Tactic:
    '''Represents a high-level derivation sketch that operates on proof terms.
    What makes a tactic a "sketch" is that it can have holes in arguments,
    which is represented by arguments with a name starting with '?'.
    '''

    def __init__(self, name: str, steps: list[Step]):
        self.steps = steps
        self.name = name

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

        # A trace is a tuple[universe, list[definitions]]
        traces = [Trace({}, u, [], [])]

        for s in self.steps:
            new_traces = []

            for trace in traces:
                # 1- Execute the step.
                # NOTE: apply is fully non-deterministic. We should replace this with a new
                # API for specifying all known parameters, and only being non-deterministic on the holes.
                new_defs = trace.universe.apply(s.arrow)

                # 2- For each valid result, make a new trace.
                for d in new_defs:
                    args = d.generating_arguments()

                    new_assignments = self._unify_args(args, s.arguments, trace)

                    if new_assignments is not None:
                        u = trace.universe.clone()
                        subdef_name = f'!{self.name}{u.next_id()}'
                        new_subdefs = u.define(subdef_name, d)
                        new_assignments[s.result] = subdef_name
                        new_traces.append(Trace(
                            new_assignments,
                            u,
                            trace.definitions + [(subdef_name, d)],
                            subdefs=trace.subdefs + new_subdefs))

            traces = new_traces

        return traces

    def _unify_args(self,
                    concrete_args: list[str],
                    abstract_args: list[str],
                    trace: list[dict]) -> Optional[dict]:

        assignments = trace.assignments.copy()

        for concrete, abstract in zip(concrete_args, abstract_args):
            if concrete == assignments.get(abstract, abstract):
                continue

            # They are different. If abstract is not a parameter, then this is a mismatch.
            if not abstract.startswith('?'):
                return None

            # Abstract is a parameter. If it's already assigned (and we know it's to
            # a different value) then this is a mismatch.
            if abstract in assignments:
                return None

            # Otherwise, it's an unassigned parameter: unify.
            assignments[abstract] = concrete

        return assignments

    def _execute_step(self, u: peano.PyUniverse, trace: list[peano.PyDefinition]):
        raise NotImplementedError()


class TacticsTest(unittest.TestCase):
    def test_eval_rewrite_tactic(self):
        import domain

        tactic = Tactic(
            'eval_rewrite_x2',
            [
                Step('eval', ['?a'], '?0'),
                Step('rewrite', ['?0', '?b'], '?1'),
                Step('eval', ['?c'], '?2'),
                Step('rewrite', ['?2', '?1'], '?3'),
            ]
        )

        d = domain.make_domain('subst-eval')
        problem = d.start_derivation('(= x (+ (+ 1 2) 3))', '(= x ?)')

        traces = tactic.execute(problem.universe)

        # This tactic should produce only one result here.
        assert len(traces) == 1
        # And its last definition should be a proof that (= x 6).
        assert (traces[0].definitions[-1][1]
                .clean_dtype(traces[0].universe) == '(= x 6)')
