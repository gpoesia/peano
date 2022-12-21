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
