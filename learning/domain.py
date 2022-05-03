#!/usr/bin/env python3

import os
import pickle
from fractions import Fraction

import peano


# Representing an existential type: each domain has /some/ associated problem type.
class Problem:
    def __init__(self, universe: peano.PyUniverse, description: str):
        self.universe = universe
        self.description = description


class Domain:
    def generate(self, seed: int) -> Problem:
        raise NotImplementedError()

    def state(self, p) -> str:
        raise NotImplementedError()

    def actions(self, p) -> list[str]:
        raise NotImplementedError()

    def reward(self, p: Problem) -> bool:
        raise NotImplementedError()


class EquationsDomain(Domain):
    def __init__(self):
        blank_domain = peano.get_domain('blank')
        self.base_universe = blank_domain.generate(0)
        self.base_universe.incorporate('''
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

*0_null : [(a : real) -> (= (* a 0) 0)].
0_div_null : [(a : real) -> (!= a 0) -> (= (/ 0 a) 0)].
x : real.
''')
        self.action_set = set(self.base_universe.actions())
        self.ignore = self.action_set.union({'real'})
        self.action_set -= {'=', '!='}

        with open(os.path.join(os.path.dirname(__file__), 'linear-equations.pkl'), 'rb') as f:
            problems_dict = pickle.load(f)

        self.problems = list(problems_dict.keys())

    def generate(self, seed: int):
        problem_str = self.problems[seed % len(self.problems)]
        u = self.base_universe.clone()
        u.incorporate(f'equation: {problem_str}.')
        return Problem(u, problem_str)

    def reward(self, p: Problem) -> bool:
        'Try to find a rational in the equivalence class of x'
        s = p.universe.state()
        x_class = [c for (c, _dtype) in s if 'x' in c][0]
        for obj in x_class:
            try:
                Fraction(obj)
                return True
            except ValueError:
                pass
        return False

    def state(self, p) -> str:
        return '; '.join(f'{{{"=".join(set(vals))}}} : {dtype}'
                         for vals, dtype in p.universe.state(self.ignore))

    def actions(self, p):
        return list(self.action_set)
