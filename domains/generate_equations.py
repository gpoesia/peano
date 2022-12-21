from dataclasses import dataclass
from enum import Enum
from random import randint, choices, choice
from typing import List, Set, Tuple
from sympy.solvers import solve
from sympy import Symbol, simplify
from math import sqrt
import click
import pickle
from utils import format


class Generator:
    def generate_term(self):
        raise NotImplementedError()

    def solve(self):
        raise NotImplementedError()

    def split_term(self, s: str) -> str:
        stack = []
        if s[0] == "(":
            for i, c in enumerate(s):
                if c == "(":
                    stack.append(i)
                elif c == ")" and stack:
                    stack.pop()
                if len(stack) == 0:
                    return i + 1
        else:
            i = 0
            while s[i] != " ":
                i += 1
            return i

    def format(
        self,
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

            e = e[2:]
            s = self.split_term(e)
            return f"({f_r(e[:s])}{operator}{f_r(e[s+1:])})"

        return f_r(term)[1:-1]


class FractionGeneratorActionType(Enum):
    MULTIPLY_NEW_FRACTION = 0
    ADD_NEW_FRACTION = 1
    MULTIPLY_NEW_CONSTANT = 0
    ADD_NEW_CONSTANT = 1
    ADD_MODIFY_NUMERATOR = 3
    MULTIPLY_MODIFY_NUMERATOR = 4


@dataclass
class FractionGeneratorAction:
    type: FractionGeneratorActionType
    probability: float
    complexity: float


@dataclass
class FractionGeneratorConfig:
    constants: List[int]
    scales: List[int]
    actions: List[FractionGeneratorAction]
    complexity: float


class FractionGenerator(Generator):
    def __init__(self, config: FractionGeneratorConfig):
        self.config = config
        self.denominators = []

    def is_valid_denominator(self, n: int) -> bool:

        if n == 0:
            return False

        def prime_factor(n: int) -> Set[int]:
            factors = set()
            while n % 2 == 0 and n != 0:
                n = n / 2
                if 2 not in factors:
                    factors.add(2)
            for i in range(3, int(sqrt(n)) + 1, 2):
                while n % i == 0 and n != 0:
                    factors.add(i)
                    n = n / i
            if n > 2:
                factors.add(n)
            return factors

        return len(prime_factor(n).difference(set(self.config.scales))) == 0

    def generate_fraction(self) -> str:
        if len(self.denominators) == 0:
            self.denominators = [
                c for c in self.config.constants if self.is_valid_denominator(c)
            ]

        return f"(/ {choice(self.config.constants)} {choice(self.denominators)})"

    def generate_term(self) -> str:
        def get_leftmost_fraction_numerator(p: str) -> tuple[int, int]:
            frac = p.rfind("/")
            check = frac + 2
            term_start = check
            term_end = term_start + self.split_term(p[term_start:])
            return term_start, term_end

        def gt_r(p: str, c: float) -> str:
            available_actions = [a for a in self.config.actions if a.complexity <= c]
            # Base case, no available actions
            if len(available_actions) == 0:
                return p  # Return current problem
            # Recursive case, modify current problem
            action = choices(
                available_actions, weights=[a.probability for a in available_actions]
            )[0]
            c = c - action.complexity

            match action.type:
                case FractionGeneratorActionType.ADD_NEW_FRACTION:
                    return gt_r(f"(+ {p} {self.generate_fraction()})", c)
                case FractionGeneratorActionType.MULTIPLY_NEW_FRACTION:
                    return gt_r(f"(* {p} {self.generate_fraction()})", c)
                case FractionGeneratorActionType.ADD_NEW_CONSTANT:
                    return gt_r(f"(+ {p} {choice(self.config.constants)})", c)
                case FractionGeneratorActionType.MULTIPLY_NEW_CONSTANT:
                    return gt_r(f"(* {p} {choice(self.config.constants)})", c)
                case FractionGeneratorActionType.ADD_MODIFY_NUMERATOR:
                    numerator_start, numerator_end = get_leftmost_fraction_numerator(p)
                    return gt_r(
                        (
                            p[:numerator_start]
                            + f"(+ {p[numerator_start:numerator_end]} {choice(self.config.constants)})"
                            + p[numerator_end:]
                        ),
                        c,
                    )
                case FractionGeneratorActionType.MULTIPLY_MODIFY_NUMERATOR:
                    numerator_start, numerator_end = get_leftmost_fraction_numerator(p)
                    return gt_r(
                        (
                            p[:numerator_start]
                            + f"(* {p[numerator_start:numerator_end]} {choice(self.config.constants)})"
                            + p[numerator_end:]
                        ),
                        c,
                    )

            raise Exception(f"Unknown action {action}")

        return gt_r(self.generate_fraction(), self.config.complexity)

    def solve(self, problem: str):
        fraction = self.format(problem)
        return simplify(fraction)


class EquationTermFormatType(Enum):
    LOWER_DEGREE = 0
    LOWER_DEGREE_PRODUCT = 1
    DEGREE_SUM = 3
    X = 4


@dataclass
class EquationTermFormat:
    type: EquationTermFormatType
    probability: float
    complexity: float


@dataclass
class EquationTermConfig:
    constants: List[str]
    formats: List[EquationTermFormat]
    complexity: float


class EquationGenerator(Generator):
    def __init__(self, degree: int, config: EquationTermConfig):
        self.degree = degree
        self.config = config

    def generate_term(self) -> str:
        """Generates a term of at most degree d"""

        def gt_r(d: int, c: int) -> str:
            # Base case, degree is zero
            if d == 0:
                return choice(self.config.constants)
            # Recursive case, positive degree
            available_formats = [f for f in self.config.formats if f.complexity <= c]
            term_format = choices(
                available_formats, weights=[f.probability for f in available_formats]
            )[0]
            c = c - term_format.complexity
            match term_format.type:
                case EquationTermFormatType.LOWER_DEGREE:
                    return gt_r(d - 1, c)
                case EquationTermFormatType.LOWER_DEGREE_PRODUCT:
                    l = randint(0, d - 1)
                    return f"(* {gt_r(d-l, c)} {gt_r(l, c)})"
                case EquationTermFormatType.DEGREE_SUM:
                    return f"(+ {gt_r(d, c)} {gt_r(d, c)})"
                case EquationTermFormatType.X:
                    # TODO: higher powers of x as well
                    return "x"
            raise Exception(f"Unknown term format {term_format}")

        return gt_r(self.degree, self.config.complexity)

    def solve(self, problem: str):
        """Solves using sympy, takes in a Peano formatted equation"""
        x = Symbol("x")  # Need this because of "eval"
        problem = self.format(problem)
        equation_parts = problem.split("=")
        lhs = equation_parts[0]
        rhs = equation_parts[1]
        problem = f"{lhs} - ({rhs})"
        return solve(eval(problem))


@click.command()
@click.option("--n", default=100, help="Number of unique equations to generate.")
@click.option("--degree", default=1, help="The maximum degree of a generated equation.")
@click.option("--complexity", default=3, help="The complexity of generated equations.")
@click.option("--output", default="equations.pkl", help="Output filepath (.pkl)")
def generate(n, degree, complexity, output):
    """Generates N equations of degree DEGREE in a pickled file OUTPUT"""
    config = EquationTermConfig(
        [str(i) for i in range(10)],
        [
            EquationTermFormat(EquationTermFormatType.LOWER_DEGREE, 0.25, 0),
            EquationTermFormat(EquationTermFormatType.LOWER_DEGREE_PRODUCT, 0.25, 1),
            EquationTermFormat(EquationTermFormatType.DEGREE_SUM, 0.25, 1),
            EquationTermFormat(EquationTermFormatType.X, 0.25, 0),
        ],
        complexity,
    )
    equations = dict()
    equation_generator = EquationGenerator(degree, config)
    while len(equations) < n:
        equation = f"(= {equation_generator.generate_term(degree, config)} {generate_term(degree, config)})"
        if equation in equations.keys():
            continue
        solution = equation_generator.solve(equation)
        if len(solution) == 0:
            continue
        solution = str(solution[0])  # Works for linear equations
        equations[equation] = solution
    with open(output, "ab") as output:
        pickle.dump(equations, output)


if __name__ == "__main__":
    # generate()
    config = FractionGeneratorConfig(
        [i for i in range(20)],
        [1, 2, 3, 5, 7],
        [
            FractionGeneratorAction(
                FractionGeneratorActionType.ADD_NEW_FRACTION, 0.16, 1
            ),
            FractionGeneratorAction(
                FractionGeneratorActionType.MULTIPLY_NEW_FRACTION, 0.16, 1
            ),
            FractionGeneratorAction(
                FractionGeneratorActionType.ADD_NEW_CONSTANT, 0.16, 1
            ),
            FractionGeneratorAction(
                FractionGeneratorActionType.MULTIPLY_NEW_CONSTANT, 0.16, 1
            ),
            FractionGeneratorAction(
                FractionGeneratorActionType.ADD_MODIFY_NUMERATOR, 0.16, 1
            ),
            FractionGeneratorAction(
                FractionGeneratorActionType.MULTIPLY_MODIFY_NUMERATOR, 0.2, 1
            ),
        ],
        5,
    )
    fraction_generator = FractionGenerator(config)
    for i in range(20):
        term = fraction_generator.generate_term()
        print(f"Problem: {term} | Solution: {fraction_generator.solve(term)}")
