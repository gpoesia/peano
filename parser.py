#!/usr/bin/env python3

from lark import Lark, Transformer
import unittest
from foundation import Application, Arrow, Atom, Declaration, Term, Lambda


class WorldTransformer(Transformer):
    def __init__(self):
        super().__init__()
        self.declarations = []

    def start(self, declarations):
        return declarations

    def declaration(self, c):
        name, type = c
        return Declaration(name, type, None)

    def definition(self, c):
        name, type, value = c
        return Declaration(name, type, value)

    def atom(self, c):
        (name,) = c
        return Atom(name)

    def arrow(self, c):
        *input_types, output_type = c
        return Arrow(tuple(input_types), output_type)

    def application(self, c):
        (function, *args) = c
        return Application(function, tuple(args))

    def name(self, n):
        return str(n[0])

    def lambda_f(self, c):
        *parameters, body = c
        return Lambda(tuple(parameters), body)

grammar = Lark(
    """
    start: ((declaration | definition) _END)*

    declaration: name _COLON type
    definition: name _COLON type _EQUALS term

    ?term: atom | lambda_f | application
    atom: name
    lambda_f: _LAMBDA "(" declaration ("," declaration)* ")" term

    ?type: atom | application | arrow

    arrow: "[" (arrow_constituent_type _ARROW)+ arrow_constituent_type "]"
    ?arrow_constituent_type: type | _OPEN declaration _CLOSE

    application: _OPEN term term+ _CLOSE

    COMMENT: "#" /[^\\n]*/
    _COLON: ":"
    _END: "."
    _EQUALS: "="
    _LAMBDA: "lambda"
    _ARROW: "->"
    _OPEN: "("
    _CLOSE: ")"
    _OPEN_T: "{"
    _CLOSE_T: "}"
    name: IDENTIFIER
    IDENTIFIER: /[a-zA-Z_]+/

    %import common.WS
    %ignore (WS | COMMENT)+
    """,
    parser="lalr",
    start=["start", "term"],
    transformer=WorldTransformer(),
)


def parse_declarations(declarations: str) -> list[Declaration]:
    return grammar.parse(declarations, start="start")


def parse_file(path: str) -> list[Declaration]:
    with open(path, encoding="utf8") as f:
        return parse_declarations(str(f.read()))


def parse_term(term: str) -> Term:
    return grammar.parse(term, start="term")


class TestGrammar(unittest.TestCase):
    def test_parse_comments(self):
        parse_declarations("""# Simple comment""")
        parse_declarations("""# Hello
            # One comment

        # Another comment
        """)

    def test_parse_simple(self):
        parse_declarations("""
        # hello
        nat : type.
        z : nat.
        s : [nat -> nat].
        leq : [nat -> nat -> prop].
        leq_n_n : [(n : nat) -> (leq n n)].
        leq_n_sn : [(n : nat) -> (leq n (s n))].
        leq_z : [nat -> prop] = lambda (n : nat) (leq z n).
        """)

    def test_parse_term(self):
        self.assertIsInstance(parse_term('type'), Atom)
        self.assertIsInstance(parse_term('(s z)'), Application)
        self.assertIsInstance(parse_term('lambda (n : nat) (s n)'), Lambda)
