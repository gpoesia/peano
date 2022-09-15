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
from solution import Solution


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
        return [v
                for k, v in sorted(list(self.assignments.items()))
                if is_parameter_name(k)]

    def return_name(self):
        name, d = self.definitions[-1]
        if isinstance(d, Trace):
            return d.return_name()
        return name


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

    def is_connected(self):
        'Checks if all intermediate steps are used to compute the tactics output.'
        steps_used = {len(self.steps) - 1}
        queue = [len(self.steps) - 1]

        while queue:
            i = queue.pop()

            for a in self.steps[i].arguments:
                if a not in steps_used and is_result_name(a):
                    idx = int(a[1:])
                    steps_used.add(a)
                    queue.append(idx)

        return len(steps_used) == len(self.steps)


    @staticmethod
    def from_solution_slice(name: str,
                            start_index: int,
                            arrows: list[str],
                            arguments: list[list[str]],
                            abstract_constants: bool = True) -> 'Tactic':
        'Constructs a tactic from a slice of a solution found in a search episode.'

        steps = []
        rewrites = {}

        for i, (arrow, args) in enumerate([(arr, args)
                                           for arr, args in zip(arrows, arguments)
                                           if args is not None]):
            result = f'?{i}'
            rewrites[f'!step{start_index + i}'] = result
            steps.append(Step(arrow, [rewrites.get(a, a) for a in args], result))

        t = Tactic(name, steps)
        return t.abstract_concrete_arguments() if abstract_constants else t

    def is_generalization_of(self, rhs: 'Tactic') -> bool:
        '''Returns whether self is equal to or a more general tactic than rhs.

        This happens when, essentially, every result produced by running rhs would
        also be produced by running self. This defines a partial order on tactics.

        Two tactics are alpha-equivalent iff this returns True for both (a, b) and (b, a).'''

        if len(self.steps) != len(rhs.steps):
            return False, None

        # Strategy: try to find an assignment of parameters to self that would
        # rewrite self into a2. A parameter of self could rewrite into either
        # a concrete value used by rhs or to one of rhs's parameters.
        assignment = {}

        for s1, s2 in zip(self.steps, rhs.steps):
            if s1.arrow != s2.arrow:
                return False, None

            for a1, a2 in zip(s1.arguments, s2.arguments):
                if is_parameter_name(a1):
                    if a1 in assignment:
                        if assignment[a1] != a2:
                            return False, None
                    else:
                        assignment[a1] = a2
                elif a1 != a2:
                    return False, None

        return True, assignment

    def is_comparable_to(self, rhs: 'Tactic') -> bool:
        '''Returns whether self and rhs belong to the same lattice with
        generalize() being the meet operator.'''
        return self.is_generalization_of(rhs)[0] or rhs.is_generalization_of(self)[0]


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

    def execute(self, u: peano.PyUniverse, d: 'Domain', toplevel = True, scope: list[str] = []) -> list[Trace]:
        'Executes the tactic on the given universe and returns all results it is able to generate.'

        traces = [Trace({}, u, [])]

        for s in self.steps:
            new_traces = []

            for trace in traces:
                # 1- Execute the step.
                # NOTE: apply is fully non-deterministic. We should replace this with a new
                # API for specifying all known parameters, and only being non-deterministic on the holes.
                new_defs = d.apply(s.arrow, trace.universe, False,
                                   (scope or []) + [v
                                    for k, v in trace.assignments.items()
                                    if is_result_name(k)])

                # 2- For each valid result, make a new trace.
                for definition in new_defs:
                    args = definition.generating_arguments()

                    new_assignments = self._unify_args(args, s.arguments, trace)

                    if new_assignments is not None:
                        u = trace.universe.clone()
                        # (definition
                        #     if isinstance(definition, Trace)
                        #     else trace).universe.clone()

                        if isinstance(definition, Trace):
                            subdef_name = definition.return_name()
                        else:
                            subdef_name = f'!tac{u.next_id()}'

                        d.define(u, subdef_name, definition)

                        new_assignments[s.result] = subdef_name
                        new_traces.append(Trace(
                            new_assignments,
                            u,
                            trace.definitions + [(subdef_name, definition)]))

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

    def rewrite_episode(self, e: Episode, d: 'Domain') -> Episode:
        if not e.success:
            return e

        e_arrows = e.actions[::2]
        e_arguments = e.arguments[1::2]

        for i in range(len(e_arrows) - len(self.steps) + 1):
            arrows = e_arrows[i:i + len(self.steps)]
            arguments = e_arguments[i:i + len(self.steps)]

            t_i = Tactic.from_solution_slice('slice', i, arrows, arguments, False)

            is_g, _ = self.is_generalization_of(t_i)

            if is_g:
                # Check if any of the intermediate results are used later in the episode.
                # If they are, then this rewrite cannot be done, since it would hide those
                # results inside the tactic's local scope.
                intermediate_results = {f'!step{j}' for j in range(i, i + len(self.steps) - 1)}
                is_scope_barrier_violated = False

                for a in e_arguments[i + len(self.steps):]:
                    if set(a).intersection(intermediate_results):
                        is_scope_barrier_violated = True
                        break

                if not is_scope_barrier_violated:
                    actions = (e.actions[:2*i] +
                               [self.name] +
                               e.actions[2*(i + len(self.steps)) - 1:])
                    states = Solution.states_from_episode(e.problem,
                                                          e.goal,
                                                          actions)
                    e_rw = Episode(e.problem, e.goal, e.domain, e.success,
                                   actions, None, states, None)

                    e_rw.recover_arguments(d)

                    return self.rewrite_episode(e_rw, d)

        return e


def rewrite_episode_using_tactics(episode: Episode, d: 'Domain',
                                  tactics: list[Tactic]) -> Episode:
    'Rewrite episode using the given set of tactics until reaching a fixpoint.'
    changed = True

    while changed:
        changed = False

        for t in tactics:
            e_rw = t.rewrite_episode(episode, d)
            if e_rw is not episode:
                episode = e_rw
                changed = True

    return episode


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
                if t.is_connected():
                    tactics_from_slices.append(t)

    print(len(tactics_from_slices), 'tactics from slices.')

    if len(tactics_from_slices) > MAX_SLICES:
        tactics_from_slices = tactics_from_slices[-MAX_SLICES:]

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
            if t.is_generalization_of(s)[0]:
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

        traces = tactic.execute(problem.universe, d)

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
        assert lgg.is_generalization_of(t1)[0]
        assert lgg.is_generalization_of(t2)[0]
        assert lgg.is_generalization_of(lgg)[0]

        assert not t1.is_generalization_of(lgg)[0]
        assert not t2.is_generalization_of(lgg)[0]

        assert not t1.is_generalization_of(t2)[0]
        assert not t2.is_generalization_of(t1)[0]

        assert t1.is_generalization_of(t1)[0]
        assert t2.is_generalization_of(t2)[0]

    def test_tactic_composition(self):
        import domain
        import policy

        t1 = Tactic(
            't1',
            [
                Step('eval', ['?a'], '?0'),
                Step('rewrite', ['?0', '?b'], '?1'),
            ]
        )

        t2 = Tactic(
            't2',
            [
                Step('t1', ['?a', '?b'], '?0'),
                Step('t1', ['?c', '?0'], '?1'),
            ]
        )

        d = domain.make_domain('subst-eval', [t1, t2])
        problem = d.start_derivation('(= x (+ (+ 1 2) 3))', '(= x ?)')

        pi = policy.ConstantPolicy('t2')
        episode = pi.beam_search(problem, depth=2, beam_size=10)

        assert episode.success
        assert episode.actions[0] == 't2'

        episode.recover_arguments(d)

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

    def test_is_connected(self):
        assert Tactic(
            'eval_rewrite_x2',
            [
                Step('eval', ['?a'], '?0'),
                Step('rewrite', ['?0', '?b'], '?1'),
                Step('eval', ['?c'], '?2'),
                Step('rewrite', ['?2', '?1'], '?3'),
            ]
        ).is_connected()

        assert not Tactic(
            'eval_rewrite_x2',
            [
                Step('eval', ['?a'], '?0'),
                Step('eval', ['?b'], '?1'),  # Useless eval
                Step('rewrite', ['?0', '?c'], '?2'),
                Step('eval', ['?d'], '?3'),
                Step('rewrite', ['?3', '?2'], '?4'),
            ]
        ).is_connected()

        assert not Tactic(
            'eval_rewrite_x2',
            [
                Step('eval', ['?a'], '?0'),
                Step('rewrite', ['?0', '?b', '?c'], '?1'),
                Step('eval', ['?d'], '?2'),
            ]
        ).is_connected()

    def test_rewrite_episode_using_tactic(self):
        import domain
        import policy

        d = domain.make_domain('subst-eval', [])
        problem = d.start_derivation('(= x (* 10 (* 2 2)))', '(= x ?)')
        pi = policy.ConstantPolicy({'eval', 'rewrite'})
        episode = pi.beam_search(problem, depth=8, beam_size=1000)

        assert episode.success
        assert episode.actions[0] == 'eval'
        assert episode.actions[2] == 'rewrite'
        assert episode.actions[4] == 'eval'
        assert episode.actions[6] == 'rewrite'

        t1 = Tactic(
            't1',
            [
                Step('eval', ['?a'], '?0'),
                Step('rewrite', ['?0', '?b'], '?1'),
            ]
        )

        d.load_tactics([t1])

        e_rw = t1.rewrite_episode(episode, d)
        assert len(e_rw.actions) == 4
        assert e_rw.actions[0] == 't1'
        assert e_rw.actions[1] == '(= x (* 10 4))'
        assert e_rw.actions[2] == 't1'
        assert e_rw.actions[3] == '(= x 40)'

        e_rw.recompute_negatives(d)
        assert len(e_rw.negative_actions) == 4

    def test_tactics_scoping(self):
        import domain
        import policy

        # This test should fail if we disable scoping in the environment,
        # e.g. by returning false in
        # environment/src/universe/derivation.rs: fn _of_scope()

        d = domain.make_domain('subst-eval', [])
        problem = d.start_derivation('(= x (+ (+ 9 0) 0))', '(= x ?)')

        t1 = Tactic(
            'tactic000',
            [
                Step('+0_id', ['?a'], '?0'),
                Step('rewrite', ['?0', '?b'], '?1'),
            ]
        )

        d.load_tactics([t1])

        traces = t1.execute(problem.universe, d)

        for t in traces:
            if d.value_of(t.universe, t.definitions[-1][1]) == '(= x (+ 9 0))':
                # We should not be able to apply a rewrite using the result of +0_id to
                # get (= x 9), since the result of +0_id should be local to the tactic.
                defs = t.universe.apply('rewrite')

                for d_i in defs:
                    if t.universe.value_of(d_i) == '(= x 9)':
                        assert False, "This must have used a local result from the tactic"

def induce(cfg: DictConfig):
    from domain import make_domain

    with open(cfg.episodes, 'rb') as f:
        episodes = pickle.load(f)
        episodes = episodes[-5000:]

        if 'domain' in cfg:
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
