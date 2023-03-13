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
    def __init__(self, universe: peano.PyDerivation, description: str, goal: str, domain: 'Domain'):
        self.universe = universe
        self.description = description
        self.goal = goal
        self.domain = domain

    def domain_name(self):
        return [k for k, v in DOMAINS.items() if self.domain.__class__ is v][0]


class Domain:
    def __init__(self):
        self.tactics = {}

    def load_tactics(self, tactics):
        self.tactics = {t.name : t for t in tactics}

    def tactic_actions(self) -> list[str]:
        return list(self.tactics.keys())

    def get_tactic(self, name: str) -> 'Tactic':
        return self.tactics.get(name)

    def apply(self,
              arrow: str,
              universe: peano.PyDerivation,
              toplevel: bool = True,
              scope: list[str] = None,
              args: list[str] = None) -> list[peano.PyDefinition]:
        if arrow in self.tactics:
            return self.tactics[arrow].execute(universe, self, toplevel, scope, args)

        return universe.apply(arrow, scope, args)

    def value_of(self, universe, definition) -> str:
        if hasattr(definition, 'definitions'):
            return self.value_of(universe, definition.definitions[-1][1])
        return universe.value_of(definition)

    def define(self, universe, name, definition, toplevel=True) -> list[str]:
        if hasattr(definition, 'definitions'):
            subdefs = []

            for i, (d_name, d) in enumerate(definition.definitions):
                is_last = (i + 1) == len(definition.definitions)
                subdef_name = (name
                               if toplevel and name and is_last
                               else d_name)
                subdefs.extend(self.define(universe, subdef_name, d,
                                           toplevel and is_last))

            if definition.universe is not None:
                universe.fast_forward_next_id(definition.universe.peek_next_id())

            return subdefs

        return universe.define(name, definition)

    def generate(self, seed: int) -> Problem:
        raise NotImplementedError()

    def make_problem(self, description: str, goal: str) -> Problem:
        raise NotImplementedError()

    def state(self, universe: peano.PyDerivation) -> str:
        raise NotImplementedError()

    def actions(self, universe: peano.PyDerivation) -> list[str]:
        raise NotImplementedError()

    def reward(self, universe: peano.PyDerivation) -> bool:
        raise NotImplementedError()


class EquationsDomain(Domain):
    def __init__(self,
                 cached_problems=None,
                 variables=['x'],
                 actions=None):
        super().__init__()
        equations_theory = '''
= : [(t : type) -> t -> t -> prop].
!= : [(t : type) -> t -> t -> prop].

real : type.
+ : [real -> real -> real].
- : [real -> real -> real].
* : [real -> real -> real].
/ : [real -> real -> real].

/* Commutativity */
+_comm : [((+ 'a 'b) : real) -> (= (+ 'a 'b) (+ 'b 'a))].
*_comm : [((* 'a 'b) : real) -> (= (* 'a 'b) (* 'b 'a))].

/* Associativity */
+_assoc_l : [((+ (+ 'a 'b) 'c) : real) -> (= (+ (+ 'a 'b) 'c) (+ 'a (+ 'b 'c)))].
+_assoc_r : [((+ 'a (+ 'b 'c)) : real) -> (= (+ 'a (+ 'b 'c)) (+ (+ 'a 'b) 'c))].
-+_assoc : [((- (+ 'a 'b) 'c) : real) -> (= (- (+ 'a 'b) 'c) (+ 'a (- 'b 'c)))].
+-_assoc : [((+ (- 'a 'b) 'c) : real) -> (= (+ (- 'a 'b) 'c) (+ 'a (- 'c 'b)))].
*/_assoc_r : [((/ (* 'a 'b) 'c) : real) -> (= (/ (* 'a 'b) 'c) (* 'a (/ 'b 'c)))].
*/_assoc_l : [((* (/ 'a 'b) 'c) : real) -> (= (* (/ 'a 'b) 'c) (* 'a (/ 'b 'c)))].

/* Distributivity */
+*_dist_l : [((+ (* 'a 'c) (* 'b 'c)) : real) -> (= (+ (* 'a 'c) (* 'b 'c)) (* (+ 'a 'b) 'c))].
-*_dist_l : [((- (* 'a 'c) (* 'b 'c)) : real) -> (= (- (* 'a 'c) (* 'b 'c)) (* (- 'a 'b) 'c))].

/* Cancellation axioms */
+0_id : [((+ 'a 0) : real) -> (= (+ 'a 0) 'a)].
-0_id : [((- 'a 0) : real) -> (= (- 'a 0) 'a)].
*1_id : [((* 'a 1) : real) -> (= (* 'a 1) 'a)].
/1_id : [((/ 'a 1) : real) -> (= (/ 'a 1) 'a)].
*0_null : [((* 'a 0) : real) -> (= (* 'a 0) 0)].

/* Operate on both sides of an equation */
add_eq : [(= 'a 'b) -> ('c : real) -> (= (+ 'a 'c) (+ 'b 'c))].
sub_eq : [(= 'a 'b) -> ('c : real) -> (= (- 'a 'c) (- 'b 'c))].
mul_eq : [(= 'a 'b) -> ('c : real) -> (= (* 'a 'c) (* 'b 'c))].
div_eq : [(= 'a 'b) -> ('c : real) -> (= (/ 'a 'c) (/ 'b 'c))].

div_self_id : [((/ 'a 'a) : real) -> (= (/ 'a 'a) 1)].
-self_null: [((- 'a 'a) : real) -> (= (- 'a 'a) 0)].

/* 0_div_null : [((/ 0 'a) : real) -> (= (/ 0 'a) 0)]. */
'''
        for v in variables:
            equations_theory += f'{v} : real.'
        self.base_derivation = peano.PyDerivation()
        self.base_derivation.incorporate(equations_theory)

        self.d_action_set = set(self.base_derivation.actions())
        self.d_ignore = self.d_action_set.union({'real'})

        if actions is not None:
            self.d_action_set = actions
        else:
            self.d_action_set -= {'=', '!=', '+', '-', '*', '/', 'eq_refl'}
            self.d_action_set = list(self.d_action_set)

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

    def reward(self, universe: peano.PyDerivation) -> bool:
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
        'Try to find an equality between x and a rational constant.'
        for name, dtype, _, is_prop, _deps in universe.state():
            if not is_prop:
                continue
            try:
                m = re.match(r'^\(= x (.*)\)$', dtype)
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
        return universe.state(self.d_ignore)

    def actions(self, _universe):
        return list(self.action_set)

    def derivation_actions(self, _universe):
        return list(self.d_action_set)

class DomainFromTheory(Domain):
    'Creates a domain from one of the theories implemented in the environment.'
    def __init__(self, theory, actions):
        with open(os.path.join(os.path.dirname(__file__),
                               '..', 'environment', 'theories', theory)) as f:
            self.theory = str(f.read())

        self.base_derivation = peano.PyDerivation()
        self.base_derivation.incorporate(self.theory)

        self.initial_theory_state = set([s[0] for s in self.base_derivation.state()])
        self.actions = actions

    def derivation_actions(self, _universe):
        return self.actions

    def derivation_state(self, universe):
        return universe.state(self.initial_theory_state)


class NaturalAddition(DomainFromTheory):
    def __init__(self, max_n=10):
        super().__init__('arithmetic.p', ['rewrite', '+zl', '+zr', '+s'])
        self.max_n = max_n

    @staticmethod
    @functools.cache
    def _format_unary_nat(n):
        return 'z' if n == 0 else f'(s {NaturalAddition._format_unary_nat(n-1)})'

    def generate_derivation(self, seed: int):
        random.seed(seed)
        n1 = random.randint(0, self.max_n)
        n2 = random.randint(0, self.max_n)
        n_term = NaturalAddition._format_unary_nat(n1)
        m_term = NaturalAddition._format_unary_nat(n2)
        eq = f'(= ans (n+ {n_term} {m_term}))'
        return self.start_derivation(eq, 'nat+ ans')

    def start_derivation(self, eq, goal):
        u = self.base_derivation.clone()
        u.incorporate(f'ans : nat. eq : {eq}.')
        return Problem(u, eq, goal, self)

    def derivation_done(self, universe: peano.PyDerivation) -> Optional[str]:
        'Try to find an equality between ans and a reduced natural number.'
        for name, dtype, _, is_prop, _deps in universe.state():
            if not is_prop:
                continue
            m = re.match(r'^\(= ans (.*)\)$', dtype)
            if m and m.groups()[0].find('n+') == -1:
                return name
        return None


class NaturalSubtraction(DomainFromTheory):
    def __init__(self, max_n=10):
        super().__init__('arithmetic.p', ['rewrite', '-z', '-s'])
        self.max_n = max_n

    @staticmethod
    @functools.cache
    def _format_unary_nat(n):
        return 'z' if n == 0 else f'(s {NaturalSubtraction._format_unary_nat(n-1)})'

    def generate_derivation(self, seed: int):
        random.seed(seed)
        n1 = random.randint(0, self.max_n)
        n2 = random.randint(0, n1)
        n_term = NaturalSubtraction._format_unary_nat(n1)
        m_term = NaturalSubtraction._format_unary_nat(n2)
        eq = f'(= ans (n- {n_term} {m_term}))'
        return self.start_derivation(eq, 'nat- ans')

    def start_derivation(self, eq, goal):
        u = self.base_derivation.clone()
        u.incorporate(f'ans : nat. eq : {eq}.')
        return Problem(u, eq, goal, self)

    def derivation_done(self, universe: peano.PyDerivation) -> Optional[str]:
        'Try to find an equality between ans and a reduced natural number.'
        for name, dtype, _, is_prop, _deps in universe.state():
            if not is_prop:
                continue
            m = re.match(r'^\(= ans (.*)\)$', dtype)
            if m and m.groups()[0].find('n-') == -1:
                return name
        return None

class NatCombiningLikeTerms(DomainFromTheory):
    def __init__(self, max_n=10):
        super().__init__('arithmetic.p',
                         ['+zr', '+zl', '+s', '-z', '-s', 'rewrite',
                          'n+-_assoc', 'n-+_assoc'])
        self.max_n = max_n
        self.templates = [
            "(= ans (n+ (n- x d1) d2))",
            "(= ans (n- (n+ x d2) d1))",
        ]

    @staticmethod
    @functools.cache
    def _format_unary_nat(n):
        return 'z' if n == 0 else f'(s {NatCombiningLikeTerms._format_unary_nat(n-1)})'

    def generate_derivation(self, seed: int):
        random.seed(seed)

        d1 = random.randint(0, self.max_n)
        d2 = random.randint(d1, self.max_n)

        sexp, _ = parse_sexp(random.choice(self.templates))
        sexp = randomize_atoms(sexp,
                               lambda s: s.startswith('d'),
                               lambda: NotImplementedError,
                               {'d1': NatCombiningLikeTerms._format_unary_nat(d1),
                                'd2': NatCombiningLikeTerms._format_unary_nat(d2)})

        return self.start_derivation(format_sexp(sexp), 'simpl ans')

    def start_derivation(self, eq, goal):
        u = self.base_derivation.clone()
        u.incorporate(f'ans : nat. x : nat. eq : {eq}.')
        return Problem(u, eq, goal, self)

    @staticmethod
    def _check_pattern(s: str, pattern: str, forbidden_constants: list[set[str]]):
        m = re.match(pattern, s)

        if not m:
            return False

        for g, f in zip(m.groups(), forbidden_constants):
            if g in f:
                return False

        return True

    def derivation_done(self, universe: peano.PyDerivation) -> Optional[str]:
        'Try to find an equality between answer and a simplified term.'
        for name, dtype, _, is_prop, _deps in universe.state():
            if not is_prop:
                continue
            try:
                # These match simplified expressions:
                nat = r'(\(s )*z\)*'
                if (NatCombiningLikeTerms._check_pattern(dtype, fr'\(= ans {nat}\)', []) or
                    NatCombiningLikeTerms._check_pattern(dtype, r'\(= ans x\)', []) or
                    NatCombiningLikeTerms._check_pattern(dtype, fr'\(= ans \(n[+-] x ({nat})\)\)', [{'z'}])):

                    return name
            except ValueError:
                pass
        return None


class NatOneStepAddEq(DomainFromTheory):
    def __init__(self, max_n=10):
        super().__init__('arithmetic.p',
                         ['+zr', '+zl', '+s', '-z', '-s', 'rewrite',
                          'n+-_assoc', 'n-+_assoc', 'nadd_eq', 'nsub_eq'])

        self.max_n = max_n
        self.templates = [
            "(= (n+ x d1) d2)",
            "(= (n- x d1) d2)",
        ]

    @staticmethod
    @functools.cache
    def _format_unary_nat(n):
        return 'z' if n == 0 else f'(s {NatOneStepAddEq._format_unary_nat(n-1)})'

    def generate_derivation(self, seed: int):
        random.seed(seed)

        d1 = random.randint(0, self.max_n)
        d2 = random.randint(d1, self.max_n)

        sexp, _ = parse_sexp(random.choice(self.templates))
        sexp = randomize_atoms(sexp,
                               lambda s: s.startswith('d'),
                               lambda: NotImplementedError,
                               {'d1': NatOneStepAddEq._format_unary_nat(d1),
                                'd2': NatOneStepAddEq._format_unary_nat(d2)})

        return self.start_derivation(format_sexp(sexp), '(= x ?)')

    def start_derivation(self, eq, goal):
        u = self.base_derivation.clone()
        u.incorporate(f'x : nat. eq : {eq}.')
        return Problem(u, eq, goal, self)

    def derivation_done(self, universe: peano.PyDerivation) -> Optional[str]:
        'Try to find an equality between ans and a reduced natural number.'
        for name, dtype, _, is_prop, _deps in universe.state():
            if not is_prop:
                continue
            nat = r'(\(s )*z\)*'
            m = re.match(fr'^\(= x {nat}\)$', dtype)
            if m and m.groups()[0].find('n-') == -1:
                return name
        return None


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
    def __init__(self, templates, variables=['x'], actions=None):
        super().__init__(None, variables, actions)
        self.templates = templates

    def generate_derivation(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)

        template = random.choice(self.templates)
        sexp, _ = parse_sexp(template)

        # Randomize numeric literals.
        sexp = randomize_atoms(sexp,
                               lambda s: s.startswith('d'),
                               lambda: int(np.random.randn() * 5), {})

        sexp = randomize_atoms(sexp,
                               lambda s: s.startswith('nz'),
                               lambda: ((abs(int(np.random.randn())) + 1) *
                                        random.choice([1, -1])), {})

        # Randomize operators.
        sexp = randomize_atoms(sexp,
                               lambda s: s == 'op',
                               lambda: random.choice("+-*/"), {})
        sexp = randomize_atoms(sexp,
                               lambda s: s == '+-',
                               lambda: random.choice("+-"), {})
        sexp = randomize_atoms(sexp,
                               lambda s: s == '*/',
                               lambda: random.choice("*/"), {})

        return self.start_derivation(format_sexp(sexp), '(= x ?)')


class SubstitutionAndEvaluatingExpressions(EquationsDomainFromTemplates):
    def __init__(self):
        super().__init__([
            "(= x (op d1 nz2))",
            "(= x (op (op d1 nz2) nz3))",
            "(= x (op d1 (op nz2 nz3)))",
            "(= x (op (op d1 nz2) (op d3 nz4)))",
        ], actions=['eval', 'rewrite'])


class CombiningLikeTerms(EquationsDomainFromTemplates):
    def __init__(self):
        super().__init__([
#            "(= answer (+- x (+- d d)))",
            "(= answer (+ (- x d1) d2))",
            "(= answer (+ (- x d1) d1))",
            "(= answer (- (+ x d1) d2))",
            "(= answer (- (+ x d1) d1))",
            "(= answer (* (/ x nz1) d1))",
            "(= answer (* (/ x nz1) nz1))",
            "(= answer (/ (* x d1) d2))",
            "(= answer (/ (* x nz1) nz1))",
#            "(= answer (+- (+- d x) d))",
#            "(= answer (+- d (+- d x)))",
#            "(= answer (+- d (+- d x)))",
        ], variables=['x', 'answer'], actions=['eval', 'rewrite', '+_comm',
                                               '+_assoc_l', '+_assoc_r',
                                               '*_comm', '*/_assoc_l', '*/_assoc_r',
                                               '*1_id', '/1_id',
                                               '+0_id', '-0_id',
                                               '+-_assoc', '-+_assoc'])

    @staticmethod
    def _check_pattern(s: str, pattern: str, forbidden_constants: list[set[str]]):
        m = re.match(pattern, s)

        if not m:
            return False

        for g, f in zip(m.groups(), forbidden_constants):
            if g in f:
                return False

        return True

    def derivation_done(self, universe: peano.PyDerivation) -> Optional[str]:
        'Try to find an equality between answer and a simplified term.'
        for name, dtype, _, is_prop, _deps in universe.state():
            if not is_prop:
                continue
            try:
                # These match simplified expressions:
                rational = r'-?\d+(/\d+)?'
                if (CombiningLikeTerms._check_pattern(dtype, fr'\(= answer {rational}\)', []) or
                    CombiningLikeTerms._check_pattern(dtype, fr'\(= answer x\)', []) or
                    CombiningLikeTerms._check_pattern(dtype, fr'\(= answer \([+-] x ({rational})\)\)', [{'0'}]) or
                    CombiningLikeTerms._check_pattern(dtype, fr'\(= answer \(- {rational} x\)\)', []) or
                    CombiningLikeTerms._check_pattern(dtype, fr'\(= answer \(\* x ({rational})\)\)', [{'0', '1'}]) or
                    CombiningLikeTerms._check_pattern(dtype, fr'\(= answer \(/ x ({rational})\)\)', [{'1'}]) or
                    CombiningLikeTerms._check_pattern(dtype, fr'\(= answer \([+-] \(\* x ({rational})\) ({rational})\)\)', [{'0', '1'}, {'0'}]) or
                    CombiningLikeTerms._check_pattern(dtype, fr'\(= answer \([+-] \(/ x ({rational})\) ({rational})\)\)', [{'1'}, {'0'}])):

                    return name
            except ValueError:
                pass
        return None


class OneStepAdditionAndSubtractionEquations(EquationsDomainFromTemplates):
    def __init__(self):
        super().__init__([
            "(= (+ x d1) d2)",
            "(= (- x d1) d2)",
#            "(= (+ d x) d)",
#            "(= (- d x) d)",
        ], actions=['eval', 'rewrite', '+_comm',
                    '+0_id', '-0_id',
                    '+_assoc_l', '+_assoc_r',
                    '+-_assoc', '-+_assoc',
                    'add_eq', 'sub_eq'])

class OneStepMultiplicationAndDivisionEquations(EquationsDomainFromTemplates):
    def __init__(self):
        super().__init__([
            "(= (* x nz1) d2)",
            "(= (/ x nz1) d2)",
            "(= (* nz1 x) d2)",
        ], actions=['eval', 'rewrite', '*_comm',
                    '*1_id', '/1_id', '*0_null',
                    '*/_assoc_l', '*/_assoc_r',
                    'mul_eq', 'div_eq'])

class TwoStepEquations(EquationsDomainFromTemplates):
    def __init__(self):
        super().__init__([
            "(= (+- (*/ x d1) d2) d3)",
        ], actions=[
            'eval', 'rewrite', '*_comm',
            '*1_id', '/1_id',
            '*/_assoc_l', '*/_assoc_r',
            'mul_eq', 'div_eq',
            '+_comm', '+0_id', '-0_id',
            '+_assoc_l', '+_assoc_r',
            '+-_assoc', '-+_assoc',
            'add_eq', 'sub_eq'
        ])


class CountingDomain(Domain):
    def __init__(self, max_term=50):
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
        for name, dtype, _, is_prop, _deps in universe.state():
            if not is_prop:
                continue
            m = re.match(r'^\(= n (.*)\)$', dtype)
            if m and m.groups()[0].find('add') == -1:
                return name
        return None

    def derivation_state(self, universe) -> str:
        return universe.state(self.d_ignore)

    def derivation_actions(self, _universe):
        return list(self.d_action_set)


class MixedDomain(Domain):
    def __init__(self, subdomains, weights=None):
        super().__init__()
        self.subdomains = subdomains
        weights = np.array(weights or ([1] * len(subdomains)))
        self.probs = weights / weights.sum()

    def generate_derivation(self, seed: int) -> Problem:
        np.random.seed(seed)
        domain = self.subdomains[np.random.choice(list(range(len(self.probs))),
                                                  p=self.probs)]
        return domain.generate_derivation(seed)

    def load_tactics(self, tactics):
        for d in self.subdomains:
            d.load_tactics(tactics)

    def derivation_done(self, universe: peano.PyDerivation) -> Optional[str]:
        raise NotImplementedError('Should call derivation_done from problem.domain')

    def derivation_state(self, universe: peano.PyDerivation) -> Optional[str]:
        raise NotImplementedError('Should call derivation_state from problem.domain')

    def derivation_actions(self, universe: peano.PyDerivation) -> list[str]:
        raise NotImplementedError('Should call derivation_actions from problem.domain')


class PrecomputedProblemSet(Domain):
    def __init__(self, path):
        super().__init__()

        with open(path, 'rb') as f:
            self.problems = pickle.load(f)

    def generate_derivation(self, seed: int) -> Problem:
        random.seed(seed)
        problem = random.choice(self.problems)
        domain = make_domain(problem['domain'])
        domain.load_tactics(self.tactics.values())

        return domain.start_derivation(problem['problem'], problem['goal'])


DOMAINS = {
    'equations': EquationsDomain,
    'counting': CountingDomain,
    'nat-add': NaturalAddition,
    'nat-sub': NaturalSubtraction,
    'nat-comb-like': NatCombiningLikeTerms,
    'nat-one-step-add-eq': NatOneStepAddEq,
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
}


def make_domain(name, tactics=[]):
    name = name.strip()
    # Example syntax: mix(equations, comb-like, simpl0)
    if name.startswith('mix(') and name.endswith(')'):
        names = list(name[len('mix('):-1].split(','))
        weights = []

        for i in range(len(names)):
            if names[i].find('=') == -1:
                weights.append(1)
            else:
                names[i], weight = names[i].split('=')
                names[i] = names[i].strip()
                weights.append(int(weight))

        d = MixedDomain(list(map(make_domain, names)), weights)
    elif name.startswith('load(') and name.endswith(')'):
        path = name[len('load('):-1]
        d = PrecomputedProblemSet(path)
    else:
        d = DOMAINS[name]()

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
