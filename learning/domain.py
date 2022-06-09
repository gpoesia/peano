#!/usr/bin/env python3

import os
import pickle
from fractions import Fraction
import random
from typing import Optional
import re

import peano


# Representing an existential type: each domain has /some/ associated problem type.
class Problem:
    def __init__(self, universe: peano.PyUniverse, description: str, goal: str):
        self.universe = universe
        self.description = description
        self.goal = goal


class Domain:
    def generate(self, seed: int) -> Problem:
        raise NotImplementedError()

    def make_problem(self, description: str, goal: str) -> Problem:
        raise NotImplementedError()

    def state(self, universe: peano.PyUniverse) -> str:
        raise NotImplementedError()

    def actions(self, universe: peano.PyUniverse) -> list[str]:
        raise NotImplementedError()

    def reward(self, universe: peano.PyUniverse) -> bool:
        raise NotImplementedError()


class EquationsDomain(Domain):
    def __init__(self, cached_problems='linear-equations.pkl'):
        blank_domain = peano.get_domain('blank')
        self.base_universe = blank_domain.generate(0)
        equations_theory = '''
real : type.

= : [(t : type) -> t -> t -> prop].
!= : [(t : type) -> t -> t -> prop].

+ : [real -> real -> real].
- : [real -> real -> real].
* : [real -> real -> real].
/ : [real -> real -> real].

/* Commutativity */
+_comm : [(a : real) -> (b : real) -> (= (+ a b) (+ b a))].
*_comm : [(a : real) -> (b : real) -> (= (* a b) (* b a))].

/* Associativity */
+_assoc : [(a : real) -> (b : real) -> (c : real) -> (= (+ a (+ b c)) (+ (+ a b) c))].
+-_assoc : [(a : real) -> (b : real) -> (c : real) -> (= (- (+ a b) c) (+ a (- b c)))]. /* (a + b) - c = a + (b - c) */
*/_assoc : [(a : real) -> (b : real) -> (c : real) -> (= (/ (* a b) c) (* a (/ b c)))]. /* (a * b) / c = a * (b / c) */

/* Distributivity */
+*_dist : [(a : real) -> (b : real) -> (c : real) -> (= (* (+ a b) c) (+ (* a c) (* b c)))].

/* Cancellation axioms */
+0_id : [(a : real) -> (= (+ a 0) a)].
-0_id : [(a : real) -> (= (- a 0) a)].
*1_id : [(a : real) -> (= (* a 1) a)].
/1_id : [(a : real) -> (= (/ a 1) a)].
div_self_id : [(a : real) -> (!= a 0) -> (= (/ a a) 1)].
-self_null: [(a : real) -> (= (- a a) 0)].

*0_null : [(a : real) -> (= (* a 0) 0)].
0_div_null : [(a : real) -> (!= a 0) -> (= (/ 0 a) 0)].
x : real.
'''
        self.base_universe.incorporate(equations_theory)
        self.base_derivation = peano.PyDerivation()
        self.base_derivation.incorporate(equations_theory)

        self.action_set = set(self.base_universe.actions())
        self.d_action_set = set(self.base_derivation.actions())

        self.ignore = self.action_set.union({'real'})
        self.d_ignore = self.d_action_set.union({'real'})
        self.action_set -= {'=', '!='}
        self.d_action_set -= {'=', '!='}

        if cached_problems:
            with open(os.path.join(os.path.dirname(__file__), cached_problems), 'rb') as f:
                problems = pickle.load(f)
            self.problems = list(problems.keys()) if isinstance(problems, dict) else problems
        else:
            self.problems = None

    def generate(self, seed: int):
        if self.problems:
            return self.make_problem(self.problems[seed % len(self.problems)], '(= x ?)')
        raise ValueError('No cached problems and no generator implemented.')

    def generate_derivation(self, seed: int):
        if self.problems:
            return self.start_derivation(self.problems[seed % len(self.problems)], '(= x ?)')
        raise ValueError('No cached problems and no generator implemented.')

    def make_problem(self, equation: str, goal: str):
        u = self.base_universe.clone()
        u.incorporate(f'equation: {equation}.')
        return Problem(u, equation, goal)

    def start_derivation(self, equation: str, goal: str):
        u = self.base_derivation.clone()
        u.incorporate(f'equation: {equation}.')
        return Problem(u, equation, goal)

    def reward(self, universe: peano.PyUniverse) -> bool:
        'Try to find a rational in the equivalence class of x'
        s = universe.state()
        x_class = [c for (c, _dtype) in s if 'x' in c][0]
        for obj in x_class:
            try:
                Fraction(obj)
                return True
            except ValueError:
                pass
        return False

    def derivation_done(self, universe: peano.PyDerivation) -> Optional[str]:
        'Try to an equality between x and a rational constant.'
        for name, val, is_prop, _deps in universe.state():
            if not is_prop:
                continue
            try:
                m = re.match(r'^\(= x (.*)\)$', val)
                if m:
                    Fraction(m.group(1))
                    return name
            except ValueError:
                pass
        return None

    def state(self, universe) -> str:
        return '; '.join(f'{{{"=".join(set(vals))}}} : {dtype}'
                         for vals, dtype in universe.state(self.ignore))

    def derivation_state(self, universe) -> str:
        return '. '.join(f'{name}: {val}'
                         for name, val, _deps in universe.state(self.d_ignore))

    def actions(self, _universe):
        return list(self.action_set)

    def derivation_actions(self, _universe):
        return list(self.d_action_set)


class SimplificationDomain(EquationsDomain):
    def __init__(self, level):
        super().__init__(f'simpl-{level}.pkl')


class Simpl0Domain(SimplificationDomain):
    def __init__(self):
        super().__init__(0)


class Simpl1Domain(SimplificationDomain):
    def __init__(self):
        super().__init__(1)


class Simpl2Domain(SimplificationDomain):
    def __init__(self):
        super().__init__(2)


class Simpl3Domain(SimplificationDomain):
    def __init__(self):
        super().__init__(3)


class Simpl4Domain(SimplificationDomain):
    def __init__(self):
        super().__init__(4)


def make_domain(name):
    return ({
        'equations': EquationsDomain,
        'simpl0': Simpl0Domain,
        'simpl1': Simpl1Domain,
        'simpl2': Simpl2Domain,
        'simpl3': Simpl3Domain,
        'simpl4': Simpl4Domain,
    })[name]()
