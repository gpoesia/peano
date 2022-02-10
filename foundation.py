#!/usr/bin/env python3

from dataclasses import dataclass
import collections


class Term:
    pass


@dataclass(frozen=True)
class Reference:
    dtype: Term
    value: Term


class Context:
    def __init__(self):
        self.declarations = collections.defaultdict(list)

    def lookup(self, name: str) -> Reference:
        d = self.declarations[name]
        return None if not d else d[-1]

    def declare(self, name: str, type: Term, value: Term = None):
        self.declarations[name].append(Reference(type, value))

    def destroy(self, name):
        self.declarations[name].pop()


@dataclass(frozen=True)
class Declaration(Term):
    name: str
    dtype: Term
    value: Term

    def __str__(self):
        return (self.name +
                ' : ' +
                str(self.dtype) +
                (f' = {str(self.value)}' if self.value else ''))

    def type(self, ctx: Context):
        return self.dtype.eval(ctx)

    def eval(self, ctx: Context):
        return Declaration(self.name,
                           self.dtype.eval(ctx),
                           self.value and self.value.eval(ctx))

    def replace(self, name, value):
        if name == self.name:
            return self
        return Declaration(self.name,
                           self.dtype.replace(name, value),
                           self.value and self.value.replace(name, value))


@dataclass(frozen=True)
class Atom(Term):
    identifier: str

    def eval(self, ctx: Context) -> Term:
        ref = ctx.lookup(self.identifier)
        if not ref:
            return self
        return ref.value or self

    def type(self, ctx: Context) -> Term:
        ref = ctx.lookup(self.identifier)
        if not ref:
            return self
        return ref.dtype.eval(ctx) or self

    def replace(self, name: str, value: Term) -> Term:
        if self.identifier == name:
            return value
        return self

    def __str__(self):
        return self.identifier


@dataclass(frozen=True)
class Arrow(Term):
    input_types: tuple[Term]
    output_type: Term

    def eval(self, ctx: Context) -> Term:
        return self

    def type(self, ctx: Context) -> Term:
        return Atom('type')

    def replace(self, name: str, value: Term) -> Term:
        return Arrow(tuple(it.replace(name, value) for it in self.input_types),
                     self.output_type.replace(name, value))

    def __str__(self):
        return ('[' +
                ' -> '.join(map(str, self.input_types + (self.output_type,)))
                + ']')


@dataclass(frozen=True)
class Lambda(Term):
    parameters: tuple[Declaration]
    body: Term

    def eval(self, ctx: Context) -> Term:
        return self

    def type(self, ctx: Context) -> Term:
        return Arrow(tuple(p.dtype for p in self.parameters), self.body.type(ctx))

    def replace(self, name: str, value: Term) -> Term:
        if any(name == p.name for p in self.parameters):
            return self
        return Lambda(tuple(Declaration(d.name,
                                        d.dtype.replace(name, value))
                            for d in self.parameters),
                      self.body.replace(name, value))

    def __str__(self):
        return ('lambda (' +
                ', '.join(map(str, self.parameters)) +
                ') ' +
                str(self.body))


@dataclass(frozen=True)
class Application(Term):
    function: Term
    arguments: tuple[Term]

    def eval(self, ctx: Context) -> Term:
        # Check if function is a Lambda or None
        f = self.function.eval(ctx)

        if isinstance(f, Lambda):
            remaining, b = self._apply_lambda(f, ctx)
            if remaining:
                return Lambda(remaining, b)
            return b
        elif isinstance(f, Application):
            return Application(f.function, f.arguments + self.arguments)

        return Application(f, tuple(a.eval(ctx) for a in self.arguments))

    def type(self, ctx: Context) -> Term:
        ft = self.function.eval(ctx).type(ctx)

        input_types = list(ft.input_types)
        output_type = ft.output_type

        for i, a in enumerate(self.arguments):
            if isinstance(input_types[i], Declaration):
                v = a.eval(ctx)
                name = input_types[i].name
                for j in range(i+1, len(input_types)):
                    input_types[j] = input_types[j].replace(name, v).eval(ctx)
                output_type = output_type.replace(name, v).eval(ctx)

        if len(self.arguments) == len(ft.input_types):
            return output_type

        return Arrow(tuple(input_types[len(self.arguments):]), output_type)

    def _apply_lambda(self, f: Lambda, ctx: Context) -> tuple:
        p, b = list(f.parameters), f.body
        for i, a in enumerate(self.arguments):
            a = a.eval(ctx)
            b = b.replace(p[i].name, a)
            for j in range(i+1, len(p)):
                p[j] = p[j].replace(p[i].name, a)

        remaining = tuple(p[len(self.arguments):])

        return remaining, b

    def replace(self, name: str, value: Term) -> Term:
        return Application(self.function.replace(name, value),
                           tuple(a.replace(name, value) for a in self.arguments))

    def __str__(self):
        return ('(' +
                ' '.join(map(str, (self.function,) + self.arguments)) +
                ')')
