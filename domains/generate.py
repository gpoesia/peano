from dataclasses import dataclass
from enum import Enum
from random import randint, choices, choice
from typing import List
from sympy.solvers import solve
from sympy import Symbol


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
