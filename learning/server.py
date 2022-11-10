#!/usr/bin/env python3

'''
Server to interactive webapp.
'''

import pickle

from bottle import route, run, request
import torch
import sympy

from domain import Domain, make_domain
from tactics import Tactic, Trace, reconstruct_solution_trace
from util import toggle_infix, format_sexp, parse_sexp
from policy import RandomPolicy


tactics = pickle.load(open('tactics.pkl', 'rb'))
episodes = pickle.load(open('episodes.pkl', 'rb'))
policy = torch.load('policy.pt', map_location=torch.device('cpu'))
policy = policy.to(torch.device('cpu'))

eq_domain = make_domain('two-step-eq', tactics)


def sympy_to_sexp(exp) -> str:
    if isinstance(exp, sympy.Mul):
        return f'(* {sympy_to_sexp(exp.args[0])} {sympy_to_sexp(exp.args[1])})'
    if isinstance(exp, sympy.Div):
        return f'(/ {sympy_to_sexp(exp.args[0])} {sympy_to_sexp(exp.args[1])})'


def format_value(v):
    return format_sexp(toggle_infix(parse_sexp(v)[0]))


def trace_to_dict(universe, tactic, trace, domain: Domain, arrow=None) -> dict:
    if isinstance(trace, Trace):
        d = {
            'type': 'trace',
            'arrow': arrow,
            'value': format_value(domain.value_of(universe, trace))
        }

        steps = []

        for i, (_, defn) in enumerate(trace.definitions):
            steps.append(trace_to_dict(universe,
                                       domain.get_tactic(tactic.steps[i].arrow),
                                       defn,
                                       domain,
                                       tactic.steps[i].arrow))

        d['steps'] = steps

    else:
        d = {
            'type': 'axiom',
            'arrow': arrow,
            'value': format_value(domain.value_of(universe, trace))
        }

    return d


@route('/example/<domain>/<index>')
def example(domain, index):
    episode = [e for e in reversed(episodes)
               if e.success and e.domain == domain][int(index)]
    domain = make_domain(episode.domain, tactics)
    sol_tactic, trace = reconstruct_solution_trace(episode, domain)

    return {
        'problem': format_value(episode.problem),
        'solution': format_value(domain.value_of(trace.universe, trace)),
        'trace': trace_to_dict(trace.universe, sol_tactic, trace, domain)
    }

@route('/check')
def check():
    equations = request.query.solution.split('\n')

    sols, checks = [], []

    for eq in equations:
        lhs, rhs = eq.split('=')
        lhs = sympy.parse_expr(lhs)
        rhs = sympy.parse_expr(rhs)
        sols.append(sympy.solve(sympy.Eq(lhs, rhs)))

        if len(sols) >= 2:
            checks.append(sols[-1] == sols[-2])
            if not checks[-1]:
                break

    return {'checks': checks}

@route('/solve')
def solve():
    equation = f'({request.query.equation})'

    episode = RandomPolicy().beam_search(
        eq_domain.start_derivation(format_value(equation), '(= x ?)'),
        8,
        0.0,
        10000,
        0
    )

    if not episode.success:
        return {'error': 'Sorry, this problem is above my pay grade.'}

    sol_tactic, trace = reconstruct_solution_trace(episode, eq_domain)

    return {
        'problem': format_value(episode.problem),
        'solution': format_value(eq_domain.value_of(trace.universe, trace)),
        'trace': trace_to_dict(trace.universe, sol_tactic, trace, eq_domain)
    }


if __name__ == '__main__':
    run(host='localhost', port=8080)
