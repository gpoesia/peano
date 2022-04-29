from dataclasses import dataclass
from enum import Enum
from random import randint, choices, choice
from typing import List
from sympy.solvers import solve
from sympy import Symbol
import click
import pickle


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


def generate_term(degree: int, config: TermConfig) -> str:
    """Generates a term of at most degree d"""

    def gt_r(d: int, c: int) -> str:
        # Base case, degree is zero
        if d == 0:
            return choice(config.constants)
        # Recursive case, positive degree
        available_formats = [f for f in config.formats if f.complexity <= c]
        term_format = choices(
            available_formats, weights=[f.probability for f in available_formats]
        )[0]
        c = c - term_format.complexity
        match term_format.type:
            case TermFormatType.LOWER_DEGREE:
                return gt_r(d - 1, c)
            case TermFormatType.LOWER_DEGREE_PRODUCT:
                l = randint(0, d - 1)
                return f"(* {gt_r(d-l, c)} {gt_r(l, c)})"
            case TermFormatType.DEGREE_SUM:
                return f"(+ {gt_r(d, c)} {gt_r(d, c)})"
            case TermFormatType.X:
                # TODO: higher powers of x as well
                return "x"
        raise Exception(f"Unknown term format {term_format}")

    return gt_r(degree, config.complexity)


def format(
    term: str,
) -> str:
    """Formats an equation string to be human readable,
    currently supports only binary operators that function
    like +, e.g. + - / * = etc."""

    def f_r(e: str) -> str:
        # Base case, single term
        if len(e) <= 1 or e[0] != "(" or e[-1] != ")":
            return e
        # Recursive case
        e = e[1:-1]
        operator = e.split(" ")[0]

        def split(s: str) -> str:
            stack = []
            for i, c in enumerate(s):
                if c == "(":
                    stack.append(i)
                elif c == ")" and stack:
                    stack.pop()
                if len(stack) == 0:
                    return i + 1

        e = e[2:]
        s = split(e)
        return f"({f_r(e[:s])}{operator}{f_r(e[s+1:])})"

    return f_r(term)[1:-1]


def sympy_solve_equation(equation: str):
    """Solves using sympy, takes in a Peano formatted equation"""
    x = Symbol("x")  # Need this because of "eval"
    equation = format(equation)
    equation_parts = equation.split("=")
    lhs = equation_parts[0]
    rhs = equation_parts[1]
    equation = f"{lhs} - ({rhs})"
    return solve(eval(equation))


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
    with open(output, 'ab') as output:
        pickle.dump(equations, output)

if __name__ == "__main__":
    generate()
