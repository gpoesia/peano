#!/usr/bin/env python3

import collections
from typing import Optional
from dataclasses import dataclass
import unittest
import pickle

import hydra
from omegaconf import DictConfig

import peano
from domain import Domain, make_domain
from policy import Episode


def next_parameter_name(n: int):
    'Returns the n-th parameter name of a tactic.'
    name = []

    while n or not name:
        name.append(chr(ord('a') + (n % 26)))
        n = n // 26

    return f'?{"".join(name)}'


def is_result_name(name: str):
    return name.startswith('?') and name[1:].isdigit()

def is_parameter_name(name: str):
    return name.startswith('?') and name[1:].isalpha()


@dataclass(eq=True, frozen=True)
class Step:
    'Represents one step in a tactic.'
    arrow: str
    arguments: tuple[str]
    result: str

    def __init__(self, arrow: str, arguments: list[str], result: str):
        object.__setattr__(self, 'arrow', arrow)
        object.__setattr__(self, 'arguments', tuple(arguments))
        object.__setattr__(self, 'result', result)

    def rewrite(self, before: str, after: str):
        'Replace all occurrences of an argument by another.'
        return Step(self.arrow,
                    [after if x == before else x for x in self.arguments],
                    after if self.result == before else self.result)

    def __str__(self):
        return f'{self.result} <- {self.arrow} {", ".join(self.arguments)}'


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
        self.steps = tuple(steps)
        self.name = name

    def __str__(self):
        return f'{self.name}:\n' + '\n'.join(map(str, self.steps))

    def __hash__(self):
        return hash(self.steps)

    def __eq__(self, rhs: 'Tactic'):
        '''Returns whether the two tactics are identical modulo their names.
        Note that this is not testing for alpha-equivalence.'''
        return self.steps == rhs.steps

    @staticmethod
    def from_solution_slice(name: str,
                            start_index: int,
                            arrows: list[str],
                            arguments: list[list[str]]) -> 'Tactic':
        'Constructs a tactic from a slice of a solution found in a search episode.'

        steps = []
        rewrites = {}

        for i, (arrow, args) in enumerate(zip(arrows, arguments)):
            result = f'?{i}'
            rewrites[f'!step{start_index + i}'] = result
            steps.append(Step(arrow, [rewrites.get(a, a) for a in args], result))

        return Tactic(name, steps)

    def is_generalization_of(self, rhs: 'Tactic') -> bool:
        '''Returns whether self is equal to or a more general tactic than rhs.

        This happens when, essentially, every result produced by running rhs would
        also be produced by running self. This defines a partial order on tactics.

        Two tactics are alpha-equivalent iff this returns True for both (a, b) and (b, a).'''

        if len(self.steps) != len(rhs.steps):
            return False

        # Strategy: try to find an assignment of parameters to self that would
        # rewrite self into a2. A parameter of self could rewrite into either
        # a concrete value used by rhs or to one of rhs's parameters.
        assignment = {}

        for s1, s2 in zip(self.steps, rhs.steps):
            if s1.arrow != s2.arrow:
                return False

            for a1, a2 in zip(s1.arguments, s2.arguments):
                if is_parameter_name(a1):
                    if a1 in assignment:
                        if assignment[a1] != a2:
                            return False
                    else:
                        assignment[a1] = a2
                elif a1 != a2:
                    return False

        return True


    def generalize(self, t: 'Tactic', lgg_name: str) -> Optional['Tactic']:
        'Returns a tactic that generalizes self and t, if possible.'

        if len(self.steps) != len(t.steps):
            return None

        params_to_lgg = {}
        lgg_steps = []

        for s1, s2 in zip(self.steps, t.steps):
            if s1.arrow != s2.arrow:
                return None

            assert s1.result == s2.result, \
                "Results should be consistently named after their indices."

            unified_args = []

            for a1, a2 in zip(s1.arguments, s2.arguments):
                # If they're both equal arguments that are not parameter names, no need
                # to generalize, just reuse the same value. Does not hold for parameters
                # since we want to make sure we'll always use parameters introduced in the lgg,
                # and even if there's a parameter with the same role and same name in both
                # tactics, we might have already used that name for something else in the lgg.
                # The cases below will then make a fresh name that will functionally map
                # to this common parameter in params_to_lgg.
                # For example, we could have params_to_lgg[('?b', '?b')] = '?d'
                # if ?d is the lgg parameter that corresponds to '?b' in both tactics.
                if a1 == a2 and not is_parameter_name(a1):
                    unified_args.append(a1)
                elif (a1, a2) in params_to_lgg:
                    # If we already have a parameter that is instantiated to a1 in self
                    # and to a2 in t, reuse it.
                    unified_args.append(params_to_lgg[(a1, a2)])
                else:
                    # Otherwise, need to make a new parameter.
                    name = next_parameter_name(len(params_to_lgg))
                    params_to_lgg[(a1, a2)] = name
                    unified_args.append(name)

            lgg_steps.append(Step(s1.arrow, unified_args, s1.result))

        return Tactic(lgg_name, lgg_steps)

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


def induce_tactics(episodes: list[Episode]):
    episodes = [e for e in episodes if e.success]
    tactics = []

    for i, e in enumerate(episodes):
        arrows, arguments = e.actions[::2], e.arguments[1::2]

        for start in range(len(arrows) - 1):
            for length in range(2, len(arrows) - 1 - start):
                t = Tactic.from_solution_slice(f't_{i}_{start}_{length}', start,
                                               arrows[start:start+length],
                                               arguments[start:start+length])
                tactics.append(t)

    print(len(tactics), 'tactics from slices.')
    lggs = []

    for i, t1 in enumerate(tactics):
        for t2 in tactics[i+1:]:
            lgg = t1.generalize(t2, t1.name)

            if lgg is not None:
                lggs.append(lgg)

    print(len(lggs), 'lggs.')
    lggs = list(set(lggs))
    print(len(lggs), 'unique lggs.')

    for t in lggs:
        print(t)


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

    def test_generalize_tactic(self):
        t1 = Tactic(
            't1',
            [
                Step('eval', ['!sub1'], '?0'),
                Step('rewrite', ['?0', '!sub2'], '?1'),
                Step('eval', ['!sub48'], '?2'),
                Step('rewrite', ['?2', '?1'], '?3'),
            ]
        )

        t2 = Tactic(
            't1',
            [
                Step('eval', ['!sub19'], '?0'),
                Step('rewrite', ['?0', '!sub42'], '?1'),
                Step('eval', ['!sub25'], '?2'),
                Step('rewrite', ['?2', '?1'], '?3'),
            ]
        )

        lgg = t1.generalize(t2, 't1+t2')

        assert lgg is not None
        assert lgg.is_generalization_of(t1)
        assert lgg.is_generalization_of(t2)
        assert lgg.is_generalization_of(lgg)

        assert not t1.is_generalization_of(lgg)
        assert not t2.is_generalization_of(lgg)

        assert not t1.is_generalization_of(t2)
        assert not t2.is_generalization_of(t1)

        assert t1.is_generalization_of(t1)
        assert t2.is_generalization_of(t2)


# HACK: This becomes obsolete for runs that now track arguments, but we
# need this if they don't.
def recover_arguments(episode: Episode, domain: Domain):
    problem = domain.start_derivation(episode.problem, episode.goal)
    arguments = []

    for i, (arrow, outcome) in enumerate(zip(episode.actions[::2], episode.actions[1::2])):
        if outcome == '_':
            arguments.append([])
            arguments.append([])
            continue

        choices = problem.universe.apply(arrow)
        arguments.append([])
        definitions = [d for d in choices if problem.universe.value_of(d) == outcome]

        assert len(definitions) > 0, "Failed to replay the solution."

        arguments.append(definitions[0].generating_arguments())
        problem.universe.define(f'!step{i}', definitions[0])

    episode.arguments = arguments


def induce(cfg: DictConfig):
    domain = make_domain(cfg.domain)

    with open(cfg.episodes, 'rb') as f:
        episodes = pickle.load(f)

        for e in episodes:
            recover_arguments(e, domain)

    induce_tactics(episodes)


@hydra.main(version_base="1.2", config_path="config", config_name="tactics")
def main(cfg: DictConfig):
    if cfg.task == 'induce':
        induce(cfg)


if __name__ == '__main__':
    main()
