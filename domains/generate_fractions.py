from dataclasses import dataclass
from enum import Enum
from random import randint, choices, choice
from typing import List
from sympy import simplify
import click
import pickle
from utils import format


class TermFormatType(Enum):
    LOWER_DEGREE = 0
    LOWER_DEGREE_PRODUCT = 1
    DEGREE_SUM = 3
    X = 4


@dataclass
class TermFormat:
    type: TermFormatType
    probability: float
    complexity: float


@dataclass
class TermConfig:
    constants: List[str]
    formats: List[TermFormat]
    complexity: float


def sympy_simplify_fraction(problem: str):
    """Simplifies fraction expression using sympy, takes in a Peano formatted problem"""
    fraction = format(problem)
    return simplify(fraction)


@click.command()
@click.option("--n", default=100, help="Number of unique equations to generate.")
@click.option("--degree", default=1, help="The maximum degree of a generated equation.")
@click.option("--complexity", default=3, help="The complexity of generated equations.")
@click.option("--output", default="equations.pkl", help="Output filepath (.pkl)")
def generate(n, degree, complexity, output):
    """Generates N equations of degree DEGREE in a pickled file OUTPUT"""
    config = TermConfig(
        [str(i) for i in range(10)],
        [
            TermFormat(TermFormatType.LOWER_DEGREE, 0.25, 0),
            TermFormat(TermFormatType.LOWER_DEGREE_PRODUCT, 0.25, 1),
            TermFormat(TermFormatType.DEGREE_SUM, 0.25, 1),
            TermFormat(TermFormatType.X, 0.25, 0),
        ],
        complexity,
    )
    equations = dict()
    while len(equations) < n:
        equation = (
            f"(= {generate_term(degree, config)} {generate_term(degree, config)})"
        )
        if equation in equations.keys():
            continue
        solution = sympy_solve_equation(equation)
        if len(solution) == 0:
            continue
        solution = str(solution[0])  # Works for linear equations
        equations[equation] = solution
    with open(output, "ab") as output:
        pickle.dump(equations, output)


if __name__ == "__main__":
    generate()
