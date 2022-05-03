#!/usr/bin/env python3

from fractions import Fraction

import peano


# Representing an existential type: each domain has /some/ associated problem type.
class Problem:
    def __init__(self, universe: peano.PyUniverse):
        self.universe = universe


class Domain:
    def generate(self, seed: int) -> Problem:
        raise NotImplementedError()

    def state(self, p) -> list[(list[str], str)]:
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

    def generate(self, seed: int):
        # TODO sample problem_str from generator.
        problem_str = '(= x (+ 2 3))'
        u = self.base_universe.clone()
        u.incorporate(f'equation: {problem_str}.')
        return Problem(u)

    def reward(self, p: Problem) -> bool:
        'Try to find a rational in the equivalence class of p'
        s = p.universe.state()
        x_class = [c for (c, _dtype) in s if 'x' in c][0]
        for obj in x_class:
            try:
                Fraction(obj)
                return True
            except ValueError:
                pass
        return False

    def state(self, p):
        return p.universe.state(self.ignore)

    def actions(self, p):
        return list(self.action_set)
