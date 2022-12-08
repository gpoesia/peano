#!/usr/bin/env python3

import pickle
import json

from tqdm import tqdm

from domain import Domain, make_domain
from tactics import Tactic, Trace, reconstruct_solution_trace, next_parameter_name
from util import format_sexp


def trace_to_proof_object(tactic, trace, domain, universe, arguments={}, index=None):
    if index is None:
        index = len(trace.definitions) - 1

    step = tactic.steps[index]
    sexp_arguments = []

    step_tactic = domain.get_tactic(step.arrow)

    assignments = {}

    for prev_i in range(index):
        assignments[f'?{prev_i}'] = trace_to_proof_object(tactic, trace, domain, universe, arguments, prev_i)

    def resolve(name):
        if name in arguments:
            return arguments[name]
        if name in assignments:
            return assignments[name]
        return name

    if step_tactic is not None:
        step_arguments = {}

        for i, a in enumerate(step.arguments):
            arg_name = next_parameter_name(i)
            arg_value = resolve(a)
            step_arguments[arg_name] = arg_value

        return trace_to_proof_object(step_tactic, trace.definitions[index][1], domain, universe, step_arguments)

    for a in step.arguments:
        sexp_arguments.append(resolve(a))

    l = [step.arrow] + sexp_arguments
    return l


if __name__ == '__main__':
    tactics = pickle.load(open('tactics.pkl', 'rb'))
    episodes = pickle.load(open('episodes.pkl', 'rb'))

    DOMAIN = 'two-step-eq'

    episodes = [e for e in reversed(episodes)
                if e.success and e.domain == DOMAIN]

    domain = make_domain(DOMAIN, tactics)
    dataset = []

    for e in tqdm(episodes):
        try:
            tactic, trace = reconstruct_solution_trace(e, domain)
            obj = trace_to_proof_object(tactic, trace, domain, trace.universe)
            dataset.append(format_sexp(obj))
        except AssertionError:
            pass

    with open(f'proof_objects_{DOMAIN}.json', 'w') as f:
        json.dump(dataset, f)
