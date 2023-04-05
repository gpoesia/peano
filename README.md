# Peano - Learning Formal Mathematical Reasoning

Peano is a formal theorem proving environment based on a dependent type system and a finitely axiomatized proof system.
Given any theory (some simple examples are in `theories`) and a problem, Peano provides a finite action space to produce /derivations/
(e.g. proofs, computations or or constructions). Like in [Metamath](https://us.metamath.org/),
steps of a formal solution in Peano are designed to be easy to manually follow. However, the formal system is based on dependent types,
a foundation that unifies elementary mathematical constructions (like numbers, sets and functions) and propositions
(like facts about particular numbers, or even properties of all numbers).

The main goal of having a finite action space is to enable proof search in general theories.
In particular, we want to be able to /learn/ to solve problems in a new theory using
reinforcement learning and no prior data. To make progress in a given domain, an agent
must not just learn how to solve problems, but also be able to learn new /abstractions/
given its solutions found so far. This gets around the issue that solutions to hard problems
get increasingly longer in terms of the base axioms (making them unlikely to be found
by an agent), but they will be short when expressed through a proper abstractions.
This is very similar to the idea of abstraction learning used in [DreamCoder](https://arxiv.org/abs/2006.08381)
(in fact, precisely so given the [Curry-Howard correspondence](https://en.wikipedia.org/wiki/Curry%E2%80%93Howard_correspondence).

While the Peano language is based on a simpler version of the Calculus of Constructions,
with an impredicative `Prop` type used to represent propositions.
The proof system, however, is not *yet* complete - only a "forward" fragment is currently supported in the action space.
As a practical implication, while one can manually write down a proof by induction in the language,
that construction is not yet available from the environment (i.e., an agent wouldn't find it).

### Paper

The current system, along with a set of experiments in formalizing and learning to solve
sections of the [Khan Academy](khanacademy.org/) platform, is explained in the following paper:

[*Peano: Learning Formal Mathematical Reasoning*](https://arxiv.org/abs/2211.15864). Gabriel Poesia and Noah D. Goodman. to appear in the Transactions of the Royal Society A in 2023.

### Compiling the environment

The Peano enviroment is written in Rust and has a Python API via [PyO3](https://pyo3.rs/v0.18.2/).

To compile it, you'll first need to install the Rust toolchain. For that, use [rustup](https://rustup.rs/).

With Rust installed, you can now compile the Peano environment:

```sh
[peano] $ cd environment
[environment] $ cargo build --release
```

This should eventually terminate without errors and produce a binary library
in `target/release` (it will be called `libpeano.so` on Linux, or something like `peano.dylib` on Mac).
To use this library as a Python module, we'll use a simple symbolic link:

```sh
[environment] $ cd ../learning
[learning] $ ln -s ../environment/target/release/libpeano.so ./peano.so
```

Note that this must be slightly adjusted on Mac (i.e., you'll link `peano.dylib` instead). With that, you should be able to do the following:


```sh
[learning] $ python
>>> import peano
>>>
```

If this works, then you're ready to use Peano from Python.

The main file to use to reproduce the Khan Academy experiments from the paper is `learning.py`, which will start an agent
to learn to solve problems using reinforcement learning and tactic induction. The config files and exact commands to run will come soon -
feel free to open an issue if you're interested in those and this hasn't been updated yet!

The Python dependencies can be installed with:

```sh
[learning] $ pip install -r requirements.txt
```
