#!/usr/bin/env python3

import collections
from typing import Optional
from dataclasses import dataclass
import unittest
import pickle
from functools import cached_property
import random

import hydra
from omegaconf import DictConfig

import peano
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

    def generating_arguments(self):
        'Returns the concrete values passed as each argument.'
        return [v for k, v in sorted(list(self.assignments.items()))]


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

    @cached_property
    def number_of_steps(self):
        return len(self.steps)

    @cached_property
    def number_of_parameters(self):
        return len(set(p for s in self.steps for p in s.arguments
                       if is_parameter_name(p)))

    def __eq__(self, rhs: 'Tactic'):
        '''Returns whether the two tactics are identical modulo their names.
        Note that this is not testing for alpha-equivalence.'''
        return self.steps == rhs.steps

    def rename(self, new_name: str) -> 'Tactic':
        return Tactic(new_name, self.steps)

    @staticmethod
    def from_solution_slice(name: str,
                            start_index: int,
                            arrows: list[str],
                            arguments: list[list[str]]) -> 'Tactic':
        'Constructs a tactic from a slice of a solution found in a search episode.'

        steps = []
        rewrites = {}

        for i, (arrow, args) in enumerate([(arr, args)
                                           for arr, args in zip(arrows, arguments)
                                           if args is not None]):
            result = f'?{i}'
            rewrites[f'!step{start_index + i}'] = result
            steps.append(Step(arrow, [rewrites.get(a, a) for a in args], result))

        return Tactic(name, steps).abstract_concrete_arguments()

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

    def is_comparable_to(self, rhs: 'Tactic') -> bool:
        '''Returns whether self and rhs belong to the same lattice with
        generalize() being the meet operator.'''
        return  self.is_generalization_of(rhs) or rhs.is_generalization_of(self)


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

    def abstract_concrete_arguments(self) -> 'Tactic':
        '''Abstracts away concrete arguments in steps of the tactic.

        This tries to create the least number of required formal parameters by
        reusing parameters, creating at most one parameter for each distinct
        concrete argument.

        Returns a new, abstracted tactic.
        '''

        new_steps = []

        parameter_values = {}

        for s in self.steps:
            new_args = []

            for a in s.arguments:
                if is_result_name(a):
                    new_args.append(a)
                elif a in parameter_values:
                    new_args.append(parameter_values[a])
                else:
                    new_param_name = next_parameter_name(len(parameter_values))
                    parameter_values[new_param_name] = a
                    new_args.append(new_param_name)

            new_steps.append(Step(s.arrow, new_args, s.result))

        return Tactic(self.name, new_steps)

    def execute(self, u: peano.PyUniverse) -> list[Trace]:
        'Executes the tactic on the given universe and returns all results it is able to generate.'

        traces = [Trace({}, u, [])]

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
                        try:
                          new_subdefs = u.define(subdef_name, d)
                        except:
                            import pdb; pdb.set_trace()
                        new_assignments[s.result] = subdef_name
                        new_traces.append(Trace(
                            new_assignments,
                            u,
                            trace.definitions + [(subdef_name, d)]))

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


# Maximum number of solution slices to use for inducing tactics.
MAX_SLICES = 10**4

def induce_tactics(episodes: list[Episode], max_n: int, min_score: float,
                   filter_comparable_to: list[Tactic] = []):
    episodes = [e for e in episodes if e.success]
    tactics_from_slices = []

    for i, e in enumerate(episodes):
        arrows, arguments = e.actions[::2], e.arguments[1::2]

        for start in range(len(arrows) - 1):
            for length in range(2, len(arrows) - start + 1):
                t = Tactic.from_solution_slice(f't_{i}_{start}_{length}', start,
                                               arrows[start:start+length],
                                               arguments[start:start+length])
                tactics_from_slices.append(t)

    print(len(tactics_from_slices), 'tactics from slices.')

    if len(tactics_from_slices) > MAX_SLICES:
        tactics_from_slices = random.sample(tactics_from_slices, MAX_SLICES)

    lggs = []

    for i, t1 in enumerate(tactics_from_slices):
        for t2 in tactics_from_slices[i+1:]:
            lgg = t1.generalize(t2, t1.name)

            if lgg is not None:
                lggs.append(lgg)

    print(len(lggs), 'lggs.')
    lggs = list(set(lggs))
    print(len(lggs), 'unique lggs.')

    scored_lggs = []

    for t in lggs:
        # Make sure tactic is independent from all existing ones.
        if any(map(lambda e_t: e_t.is_comparable_to(t), filter_comparable_to)):
            continue

        occurrences = 0
        for s in tactics_from_slices:
            if t.is_generalization_of(s):
                occurrences += 1
        scored_lggs.append((t, occurrences))

    scored_lggs.sort(key=(lambda ts:
                          # Occurrences.
                          ts[1] *
                          # Number of reduced steps in rewritten solutions.
                          ((ts[0].number_of_steps) - 1) /
                          # Number of parameters.
                          max(1, ts[0].number_of_parameters)),
                     reverse=True)

    candidates = []
    total_score = 0

    for i in range(len(scored_lggs)):
        is_independent = True

        for t, _s in candidates:
            if t.is_comparable_to(scored_lggs[i][0]):
                is_independent = False
                break

        if is_independent:
            candidates.append(scored_lggs[i])
            total_score += scored_lggs[i][1]

    print(f'Induced {len(candidates)} independent tactics:')

    induced_tactics = []

    for t, s in candidates:
        if len(induced_tactics) == max_n:
            break

        print(f'=== Score {s} / {s / total_score}\n', t, '\n', sep='')

        if s >= min_score:
            induced_tactics.append(t)

    print('Selected the top', len(induced_tactics))

    return induced_tactics


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

    def test_tactic_beam_search(self):
        import domain
        import policy

        tactic = Tactic(
            'eval_rewrite_x2',
            [
                Step('eval', ['?a'], '?0'),
                Step('rewrite', ['?0', '?b'], '?1'),
                Step('eval', ['?c'], '?2'),
                Step('rewrite', ['?2', '?1'], '?3'),
            ]
        )

        d = domain.make_domain('subst-eval', [tactic])
        problem = d.start_derivation('(= x (* (+ (+ 1 2) 3) (* 2 2)))', '(= x ?)')

        # Also works with the random policy, but requires large beam size (e.g. 10^4).
        pi = policy.ConstantPolicy('eval_rewrite_x2')
        episode = pi.beam_search(problem, depth=4, beam_size=1000)

        # Only way to solve the problem within this depth is with the tactic twice.
        assert episode.success
        assert episode.actions[0] == 'eval_rewrite_x2'
        assert episode.actions[2] == 'eval_rewrite_x2'

    def test_beam_search_without_tactics(self):
        # The policy execution code had to be generalize to accomodate for tactics.
        # Here we test that doesn't break execution without tactics. This test should
        # likely be in policy though.
        import domain
        import policy

        d = domain.make_domain('subst-eval', [])
        problem = d.start_derivation('(= x (* 2 2))', '(= x ?)')
        pi = policy.RandomPolicy()
        episode = pi.beam_search(problem, depth=4, beam_size=1000)

        # Only way to solve the problem within this depth is with the tactic twice.
        assert episode.success
        assert episode.actions[0] == 'eval'
        assert episode.actions[2] == 'rewrite'


def induce(cfg: DictConfig):
    from domain import make_domain


    with open(cfg.episodes, 'rb') as f:
        episodes = pickle.load(f)

        if 'domain' in cfg.domain:
            domain = make_domain(cfg.domain)
            for e in episodes:
                if cfg.get('cleanup'):
                    e.cleanup(domain)
                else:
                    e.recover_arguments(domain)

    induce_tactics(episodes, 20, 200)


@hydra.main(version_base="1.2", config_path="config", config_name="tactics")
def main(cfg: DictConfig):
    if cfg.task == 'induce':
        induce(cfg)


if __name__ == '__main__':
    main()
