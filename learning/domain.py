#!/usr/bin/env python3

import functools
import os
import pickle
from fractions import Fraction
import itertools
from typing import Optional
import re
import random

import hydra
from omegaconf import DictConfig
import numpy as np

import peano
from util import choose_from_list, parse_sexp, format_sexp, randomize_atoms


# Representing an existential type: each domain has /some/ associated problem type.
class Problem:
    def __init__(self, universe: peano.PyUniverse, description: str, goal: str, domain: 'Domain'):
        self.universe = universe
        self.description = description
        self.goal = goal
        self.domain = domain


class Domain:
    def __init__(self):
        self.tactics = []

    def load_tactics(self, tactics):
        self.tactics = tactics

    def tactic_actions(self) -> list[str]:
        return [t.name for t in self.tactics]

    def get_tactic(self, name):
        return [t for t in self.tactics if t.name == name][0]

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
    def __init__(self, cached_problems='linear-equations.pkl', variables=['x']):
        super().__init__()
        blank_domain = peano.get_domain('blank')
#        self.base_universe = blank_domain.generate(0)
        equations_theory = '''
real : type.

= : [(t : type) -> t -> t -> prop].
!= : [(t : type) -> t -> t -> prop].

+ : [real -> real -> real].
- : [real -> real -> real].
* : [real -> real -> real].
/ : [real -> real -> real].

/* Operate on both sides of an equation */
add_eq : [(= 'a 'b) -> ('c : real) -> (= (+ 'a 'c) (+ 'b 'c))].
sub_eq : [(= 'a 'b) -> ('c : real) -> (= (- 'a 'c) (- 'b 'c))].
mul_eq : [(= 'a 'b) -> ('c : real) -> (= (* 'a 'c) (* 'b 'c))].
div_eq : [(= 'a 'b) -> ('c : real) -> (= (/ 'a 'c) (/ 'b 'c))].

/* Commutativity */
+_comm : [((+ 'a 'b) : real) -> (= (+ 'a 'b) (+ 'b 'a))].
*_comm : [((* 'a 'b) : real) -> (= (* 'a 'b) (* 'b 'a))].

/* Associativity */
+_assoc_l : [((+ (+ 'a 'b) 'c) : real) -> (= (+ (+ 'a 'b) 'c) (+ 'a (+ 'b 'c)))].
+_assoc_r : [((+ 'a (+ 'b 'c)) : real) -> (= (+ (+ 'a 'b) 'c) (+ 'a (+ 'b 'c)))].

+-_assoc_r : [((- (+ 'a 'b) 'c) : real) -> (= (- (+ 'a 'b) 'c) (+ 'a (- 'b 'c)))].
+-_assoc_l : [((+ 'a (- 'b 'c)) : real) -> (= (+ 'a (- 'b 'c)) (- (+ 'a 'b) 'c))].

*/_assoc_r : [((/ (* 'a 'b) 'c) : real) -> (= (/ (* 'a 'b) 'c) (* 'a (/ 'b 'c)))].
*/_assoc_l : [((* 'a (/ 'b 'c)) : real) -> (= (* 'a (/ 'b 'c)) (/ (* 'a 'b) 'c))].

/* Distributivity */
+_assoc : [((+ (+ 'a 'b) 'c) : real) -> (= (+ (+ 'a 'b) 'c) (+ 'a (+ 'b 'c)))].

+*_dist_l : [((+ (* 'a 'c) (* 'b 'c)) : real) -> (= (+ (* 'a 'c) (* 'b 'c)) (* (+ 'a 'b) 'c))].
-*_dist_l : [((- (* 'a 'c) (* 'b 'c)) : real) -> (= (- (* 'a 'c) (* 'b 'c)) (* (- 'a 'b) 'c))].

/* Cancellation axioms */
+0_id : [((+ 'a 0) : real) -> (= (+ 'a 0) 'a)].
-0_id : [((- 'a 0) : real) -> (= (- 'a 0) 'a)].
*1_id : [((* 'a 1) : real) -> (= (* 'a 1) 'a)].
/1_id : [((/ 'a 1) : real) -> (= (/ 'a 1) 'a)].
div_self_id : [((/ 'a 'a) : real) -> (= (/ 'a 'a) 1)].
-self_null: [((- 'a 'a) : real) -> (= (- 'a 'a) 0)].

*0_null : [((* 'a 0) : real) -> (= (* 'a 0) 0)].
/* 0_div_null : [((/ 0 'a) : real) -> (= (/ 0 'a) 0)]. */
'''
        for v in variables:
            equations_theory += f'{v} : real.'
#        self.base_universe.incorporate(equations_theory)
        self.base_derivation = peano.PyDerivation()
        self.base_derivation.incorporate(equations_theory)

#        self.action_set = set(self.base_universe.actions())
        self.d_action_set = set(self.base_derivation.actions())

#        self.ignore = self.action_set.union({'real'})
        self.d_ignore = self.d_action_set.union({'real'})
#        self.action_set -= {'=', '!=', '+', '-', '*', '/'}
        self.d_action_set -= {'=', '!=', '+', '-', '*', '/', 'eq_refl'}

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
        return Problem(u, equation, goal, self)

    def start_derivation(self, equation: str, goal: str):
        u = self.base_derivation.clone()
        u.incorporate(f'equation: {equation}.')
        return Problem(u, equation, goal, self)

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
                         for name, val, _is_prop, _deps in universe.state(self.d_ignore))

    def actions(self, _universe):
        return list(self.action_set)

    def derivation_actions(self, _universe):
        return list(self.d_action_set)


class EquationsCtDomain(EquationsDomain):
    def __init__(self):
        super().__init__(f'equations-ct.pkl')


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

class EquationsDomainFromTemplates(EquationsDomain):
    def __init__(self, templates, variables=['x']):
        super().__init__(None, variables)
        self.templates = templates

    def generate_derivation(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)

        template = random.choice(self.templates)
        sexp, _ = parse_sexp(template)

        # Randomize numeric literals.
        sexp = randomize_atoms(sexp,
                               lambda s: s == 'd',
                               lambda: int(np.random.randn() * 5))

        # Randomize operators.
        sexp = randomize_atoms(sexp,
                               lambda s: s == 'op',
                               lambda: random.choice("+-*/"))
        sexp = randomize_atoms(sexp,
                               lambda s: s == '+-',
                               lambda: random.choice("+-"))
        sexp = randomize_atoms(sexp,
                               lambda s: s == '*/',
                               lambda: random.choice("*/"))

        return self.start_derivation(format_sexp(sexp), '(= x ?)')


class SubstitutionAndEvaluatingExpressions(EquationsDomainFromTemplates):
    def __init__(self):
        super().__init__([
            "(= x (op d d))",
            "(= x (op (op d d) d))",
            "(= x (op d (op d d)))",
            "(= x (op (op d d) (op d d)))",
        ])

class CombiningLikeTerms(EquationsDomainFromTemplates):
    def __init__(self):
        super().__init__([
            "(= answer (+- (+- x d) d))",
            "(= answer (*/ (*/ x d) d))"
        ], variables=['x', 'answer'])

    def derivation_done(self, universe: peano.PyDerivation) -> Optional[str]:
        'Try to find an equality between answer and a simplified term.'
        for name, val, is_prop, _deps in universe.state():
            if not is_prop:
                continue
            try:
                # These match simplified expressions:
                rational = r'-?\d+(/\d+)?'
                patterns = [
                    fr'\(= answer {rational}\)',
                    fr'\(= answer x\)',
                    fr'\(= answer \([+-] x {rational}\)\)',  # x + k
                    fr'\(= answer \(\* {rational} x\)\)',  # k * x
                    fr'\(= answer \(/ x {rational}\)\)',  # x / k
                    fr'\(= answer \([+-] \(\* {rational} x\) {rational}\)\)',  # a*x +- b
                    fr'\(= answer \([+-] \(/ x {rational}\) {rational}\)\)',  # x/k +- b
                ]
                for pattern in patterns:
                    m = re.match(pattern, val)
                    if m:
                        return name
            except ValueError:
                pass
        return None


class OneStepAdditionAndSubtractionEquations(EquationsDomainFromTemplates):
    def __init__(self):
        super().__init__([
            "(= (+ x d) d)",
            "(= (- x d) d)",
            "(= (+ d x) d)",
            "(= (- d x) d)",
        ])

class OneStepMultiplicationAndDivisionEquations(EquationsDomainFromTemplates):
    def __init__(self):
        super().__init__([
            "(= (* x d) d)",
            "(= (/ x d) d)",
            "(= (* d x) d)",
        ])


class TwoStepEquations(EquationsDomainFromTemplates):
    def __init__(self):
        super().__init__([
            "(= (op (op x d) d) d)",
            "(= (op d (op x d)) d)",
            "(= (op d (op d x)) d)",
        ])


class CountingDomain(Domain):
    def __init__(self, max_term=50):
        blank_domain = peano.get_domain('blank')
        nat_theory = '''
nat : type.

z : nat.
s : [nat -> nat].

= : [('t : type) -> 't -> 't -> prop].

add : [nat -> nat -> nat].

add_z : [((add z 'n) : nat) -> (= (add z 'n) 'n)].
add_s : [((add (s 'm) 'n) : nat) -> (= (add (s 'm) 'n) (s (add 'm 'n)))].

n : nat.
'''
        self.base_derivation = peano.PyDerivation()
        self.base_derivation.incorporate(nat_theory)

        self.d_action_set = set(self.base_derivation.actions())

        self.d_ignore = self.d_action_set.union({'nat', 'z', 'n'})
        self.d_action_set -= {'='}

        self.max_term = max_term

    def generate_derivation(self, seed: int):
        random.seed(seed)

        n1 = random.randint(0, self.max_term)
        n2 = random.randint(0, self.max_term)

        return self.start_derivation(
            f'(= n (add {CountingDomain._format_nat(n1)} {CountingDomain._format_nat(n2)}))',
            goal=f'Compute n, the addition of {n1} and {n2}'
        )

    @staticmethod
    @functools.cache
    def _format_nat(n):
        return 'z' if n == 0 else f'(s {CountingDomain._format_nat(n-1)})'

    def start_derivation(self, equation: str, goal: str):
        u = self.base_derivation.clone()
        u.incorporate(f'equation: {equation}.')
        return Problem(u, equation, goal, self)

    def derivation_done(self, universe: peano.PyDerivation) -> Optional[str]:
        'Try to an equality between n and a reduced natural number.'
        for name, val, is_prop, _deps in universe.state():
            if not is_prop:
                continue
            m = re.match(r'^\(= n (.*)\)$', val)
            if m and m.groups()[0].find('add') == -1:
                return name
        return None

    def derivation_state(self, universe) -> str:
        return '. '.join(f'{name}: {val}'
                         for name, val, _is_prop, _deps in universe.state(self.d_ignore))

    def derivation_actions(self, _universe):
        return list(self.d_action_set)


class MixedDomain(Domain):
    def __init__(self, subdomains):
        super().__init__()
        self.subdomains = subdomains

    def generate_derivation(self, seed: int) -> Problem:
        domain = self.subdomains[seed % len(self.subdomains)]
        return domain.generate_derivation(seed)

    def derivation_done(self, universe: peano.PyDerivation) -> Optional[str]:
        raise NotImplementedError('Should call derivation_done from problem.domain')

    def derivation_state(self, universe: peano.PyDerivation) -> Optional[str]:
        raise NotImplementedError('Should call derivation_state from problem.domain')

    def derivation_actions(self, universe: peano.PyUniverse) -> list[str]:
        raise NotImplementedError('Should call derivation_actions from problem.domain')


def make_domain(name, tactics=[]):
    # Example syntax: mix(equations, comb-like, simpl0)
    if name.startswith('mix(') and name.endswith(')'):
        names = list(name[len('mix('):-1].split(','))
        return MixedDomain(list(map(make_domain, names)))

    d = ({
        'equations': EquationsDomain,
        'counting': CountingDomain,
        'equations-ct': EquationsCtDomain,
        'subst-eval': SubstitutionAndEvaluatingExpressions,
        'comb-like': CombiningLikeTerms,
        'one-step-add-eq': OneStepAdditionAndSubtractionEquations,
        'one-step-mul-eq': OneStepMultiplicationAndDivisionEquations,
        'two-step-eq': TwoStepEquations,
        'simpl0': Simpl0Domain,
        'simpl1': Simpl1Domain,
        'simpl2': Simpl2Domain,
        'simpl3': Simpl3Domain,
        'simpl4': Simpl4Domain,
    })[name.strip()]()

    d.load_tactics(tactics)

    return d


@hydra.main(version_base="1.2", config_path="config", config_name="trainer")
def main(cfg: DictConfig):
    domain = make_domain(cfg.domain)
    seeds = range(*cfg.seeds)

    for seed in seeds:
        if seed == -1:
            p = domain.start_derivation(input('Type problem: '), '(= x ?)')
        else:
            p = domain.generate_derivation(seed)
            print('Problem:', p.description)

        cnt = itertools.count(0)
        solution = [(p.description, 'assumption')]

        while not domain.derivation_done(p.universe):
            print('### Solution so far:')
            for j, (v, a) in enumerate(solution):
                print(f'#{j:02d} {v} by {a}')

            action = choose_from_list('Choose action:', domain.derivation_actions(p.universe))

            defs = p.universe.apply(action)

            if not defs:
                print('No outcomes!')
                continue

            o = choose_from_list('Choose outcome:', defs)
            p.universe.define(f'!step{next(cnt)}', o)

            solution.append((p.universe.value_of(o), action))

        print('Solved!\n\n')


if __name__ == '__main__':
    main()
