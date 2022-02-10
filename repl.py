#!/usr/bin/env python3

from foundation import Context, Arrow, Declaration
from parser import parse_file, parse_term, parse_declarations


def repl():
    ctx = Context()

    try:
        while True:
            l = input('> ')

            if l.startswith("!"):
                cmd = l[1:].split()

                if cmd[0] == 'load':
                    decls = parse_file(cmd[1])

                    for d in decls:
                        ctx.declare(d.name, d.dtype, d.value)

                    print(len(decls), 'declarations loaded from', cmd[1])

                elif cmd[0] == 'actions':
                    actions = []

                    for k, v in ctx.declarations.items():
                        if not v:
                            continue
                        ref = v[-1]
                        if isinstance(ref.dtype, Arrow):
                            actions.append(k)

                    print(len(actions), 'actions:')
                    print(actions)
                elif cmd[0] == 'context':
                    for k, v in ctx.declarations.items():
                        if v:
                            print(Declaration(k, v[-1].dtype, None))# , v[-1].value))
                elif cmd[0] == 'debug':
                    breakpoint()
                else:
                    print('Unknown command', cmd[0])
            else:
                t = parse_term(l)
                result = t.eval(ctx)
                print(result, ':', result.type(ctx))

    except (KeyboardInterrupt, EOFError):
        print('Bye!')


if __name__ == '__main__':
    repl()
