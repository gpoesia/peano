#!/usr/bin/env python3

from enum import Enum
from tqdm import tqdm

import generate
import random
import pickle


class SimplificationDomain:
    '''Equations that only require "simplification" axioms (e.g., +0_id, *0_null)
    At each level, 2 new axioms are needed:
    - Level 0: +0_id and +_comm
    - Level 1: *1_id and *_comm, plus Level 0
    - Level 2: *0_null and -self_null, plus Level 1
    - Level 3: eval and /1_id, plus Level 2
    - Level 4: */_assoc and +-_assoc , plus Level 3
    '''
    def __init__(self, level=0, complexity=3):
        self.level = level
        self.max_complexity = complexity
        self.make_x_options = ['x', '+0', '0+']
        self.make_zero_options = ['0']
        self.make_one_options = ['1']
        self.make_const_options = ['c']
        self.make_expr_options = ['op', 'x', 'const', '1', '0']

        if level >= 1:
            self.make_x_options.extend(['*1', '1*'])

        if level >= 2:
            self.make_zero_options.extend(['e-e', 'e*0', '0*e'])

        if level >= 3:
            self.make_x_options.extend(['x/1'])
            self.make_const_options.extend(['op'])
            self.make_one_options.extend(['op', 'e/e'])

        if level >= 4:
            self.make_x_options.extend(['(x*e)/e', '(e*x)/e', '(x+e)-e', '(e+x)-e'])

        self.complexity = {
            'c': 0,
            '0': 0,
            '1': 0,
            'x': 0,
            '*1': 1,
            '1*': 1,
            '+0': 1,
            '0+': 1,
            'e-e': 1,
            'e*0': 1,
            '0*e': 1,
            'c+0': 1,
            'c*1': 1,
            '(x+e)-e': 2,
            '(e+x)-e': 2,
            'x/1': 1,
            '(c*e)/c': 2,
            '(e*c)/c': 2,
            'op': 1,
            'c/c': 1,
            'e/e': 1,
            '(x*e)/e': 2,
            '(e*x)/e': 2,
        }

    def generate(self, seed):
        random.seed(seed)
        lhs = self._generate_x(self.max_complexity)
        rhs = self._generate_const(self.max_complexity)

        if random.randint(0, 1):
            lhs, rhs = rhs, lhs

        return f'(= {lhs} {rhs})'

    def _generate_x(self, max_complexity):
        options = [o for o in self.make_x_options
                   if self.complexity[o] <= max_complexity]

        o = random.choice(options)

        if o == 'x':
            return 'x'
        if o == '+0':
            return f'(+ {self._generate_x(max_complexity - 1)} {self._generate_zero(max_complexity - 1)})'
        if o == '0+':
            return f'(+ {self._generate_zero(max_complexity - 1)} {self._generate_x(max_complexity - 1)})'
        if o == '*1':
            return f'(* {self._generate_x(max_complexity - 1)} {self._generate_one(max_complexity - 1)})'
        if o == '1*':
            return f'(* {self._generate_x(max_complexity - 1)} {self._generate_one(max_complexity - 1)})'
        if o == '(x+e)-e':
            e = self._generate_expr(max_complexity - 2)
            x = self._generate_x(max_complexity - 2)
            return f'(- (+ {x} {e}) {e})'
        if o == '(e+x)-e':
            e = self._generate_expr(max_complexity - 2)
            x = self._generate_x(max_complexity - 2)
            return f'(- (+ {e} {x}) {e})'
        if o == '(x*e)/e':
            e = self._generate_expr(max_complexity - 2)
            x = self._generate_x(max_complexity - 2)
            return f'(/ (* {x} {e}) {e})'
        if o == '(e*x)/e':
            e = self._generate_expr(max_complexity - 2)
            x = self._generate_x(max_complexity - 2)
            return f'(/ (* {e} {x}) {e})'
        if o == 'x/1':
            x = self._generate_x(max_complexity - 1)
            o = self._generate_one(max_complexity - 1)
            return f'(/ {x} {o})'

        raise NotImplementedError(f'_generate_x with {o}')

    def _generate_zero(self, max_complexity):
        options = [o for o in self.make_zero_options
                   if self.complexity[o] <= max_complexity]

        o = random.choice(options)

        if o == '0':
            return '0'

        e = self._generate_expr(max_complexity - 1)

        if o == 'e-e':
            return f'(- {e} {e})'
        if o == 'e*0':
            return f'(* {e} {self._generate_zero(max_complexity - 1)})'
        if o == '0*e':
            return f'(* {self._generate_zero(max_complexity - 1)} {e})'

        raise NotImplementedError(f'_generate_zero with {o}')

    def _generate_one(self, max_complexity):
        options = [o for o in self.make_one_options
                   if self.complexity[o] <= max_complexity]

        o = random.choice(options)

        if o == '1':
            return '1'
        if o == 'op':
            op = random.choice(['+', '-'])
            c = random.randint(-20, 21)
            if op == '+':
                return f'(+ {-c} {c+1})'
            else:
                return f'(- {c} {c-1})'
        if o == 'e/e':
            e = self._generate_expr(max_complexity - 1)
            return f'(/ {e} {e})'

        raise NotImplementedError(f'_generate_one with {o}')

    def _generate_const(self, max_complexity):
        options = [o for o in self.make_const_options
                   if self.complexity[o] <= max_complexity]

        o = random.choice(options)

        if o == 'c':
            return random.randint(-20, 20)
        if o == 'op':
            op = random.choice(['+', '-', '*', '/'])
            lhs = self._generate_const(max_complexity - 1)
            rhs = self._generate_const(max_complexity - 1)
            return f'({op} {lhs} {rhs})'

        raise NotImplementedError(f'_generate_const with {o}')

    def _generate_expr(self, max_complexity):
        options = [o for o in self.make_expr_options
                   if o != 'op' or max_complexity > 0]

        o = random.choice(options)

        if o == 'op':
            op = random.choice(['+', '-', '*', '/'])
            lhs = self._generate_expr(max_complexity - 1)
            rhs = self._generate_expr(max_complexity - 1)
            return f'({op} {lhs} {rhs})'
        if o == 'x':
            return self._generate_x(max_complexity)
        if o == '0':
            return self._generate_zero(max_complexity)
        if o == '1':
            return self._generate_one(max_complexity)
        if o == 'const':
            return self._generate_const(max_complexity)

        raise NotImplementedError(f'_generate_expr with {o}')


def generate_simplification_problems(level, max_complexity, n_problems, output):
    print('Generating', output)
    problems = []
    seen = set()

    d = SimplificationDomain(level, max_complexity)
    i = 0

    with tqdm(total=n_problems) as pbar:
        while len(problems) < n_problems:
            p = d.generate(i)
            i += 1

            if p in seen:
                continue
            seen.add(p)

            try:
                solution = generate.sympy_solve_equation(p)
            except Exception as e:
                continue

            problems.append(p)
            pbar.update(1)

    with open(output, 'wb') as f:
        pickle.dump(problems, f)

if __name__ == '__main__':
    #generate_simplification_problems(0, 20, 10000, 'simpl-0.pkl')
    #generate_simplification_problems(1, 20, 10000, 'simpl-1.pkl')
    #generate_simplification_problems(2, 3, 100000, 'simpl-2.pkl')
    # generate_simplification_problems(3, 3, 100000, 'simpl-3.pkl')
    generate_simplification_problems(4, 3, 100000, 'simpl-4.pkl')
