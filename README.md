# Peano - Learning Formal Mathematical Reasoning

*(Note: this repository is frozen at the version we released in the paper linked below. Peano is being heavily actively developed, with a new major release by early August 2024, accompanying [this paper](https://arxiv.org/abs/2407.00695)).*

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

The Peano environment is written in Rust and has a Python API via [PyO3](https://pyo3.rs/v0.18.2/).

To compile it, you'll first need to install the Rust toolchain. For that, use [rustup](https://rustup.rs/).

With Rust installed, you can now compile the Peano environment:

```sh
[peano] $ cd environment
[environment] $ cargo build --release
```

This should eventually terminate without errors and produce a binary library
in `target/release` (it will be called `libpeano.so` on Linux, or something like `libpeano.dylib` on Mac).
To use this library as a Python module, we'll use a simple symbolic link:

```sh
[environment] $ cd ../learning
[learning] $ ln -s ../environment/target/release/libpeano.so ./peano.so
```

> [!TIP]
> On Mac, link `libpeano.dylib` instead of `libpeano.so`:
> ```sh
> [learning] $ ln -s ../environment/target/release/libpeano.dylib ./peano.so 
> ```
>
> On Apple Silicon, to avoid `ld: symbol(s) not found for architecture arm64` errors with PyO3, you may need to create a `.cargo/config.toml` file with:
> ```toml
> [target.x86_64-apple-darwin]
> rustflags = [
>   "-C", "link-arg=-undefined",
>   "-C", "link-arg=dynamic_lookup",
> ]
>
> [target.aarch64-apple-darwin]
> rustflags = [
>   "-C", "link-arg=-undefined",
>   "-C", "link-arg=dynamic_lookup",
> ]
> ```

With that, you should be able to do the following:


```sh
[learning] $ python
>>> import peano
>>>
```

If this works, then you're ready to use Peano from Python.

The main file to use to reproduce the Khan Academy experiments from the paper is `trainer.py`, which will start an agent
to learn to solve problems using reinforcement learning and tactic induction.

The Python dependencies can be installed with:

```sh
[learning] $ pip install -r requirements.txt
```

We use hydra for configuring the runs: the main configuration file to drive runs is `learning/config/trainer.yaml`.
By default, this config file will run an agent that (a) trains on all 5 Khan Academy domains at once (i.e., no curriculum),
and (b) does tactic induction. This behavior can be changed in the config to run other ablations.

To run the default experiment, simply run:
```sh
[learning] $ python trainer.py ++trainer.n_searchers=1 ++trainer.gpus=[0]
```

This spawns a trainer job with the configuration in `config/trainer.yaml`, using one "searcher" process at each iteration, and one GPU (at rank 0). If you pass no GPUs, it will run on the CPU (not recommended, since it will be slow). The trainer job will basically run a training loop which, at each iteration (a) generates a batch of problems, (b) spawns searchers to try to solve the problems, (c) evaluates the current policy in each domain, and (d) learns from the solutions found by the searchers (in the default config, this means both inducing tactics and training the policy).

The main experiments in the paper ran in 4 hours using 8 searchers (you can likely fit several searchers in a single GPU, since the models we use don't use that much GPU memory). Proof search starts to be significantly faster once the agent induces a few useful tactics, since it will then solve easy problems very quickly.

By default, evaluation performance on all domains will be tracked on wandb, which we recommend you to set up. The hydra output directory created for your run (`learning/outputs/<date>/<time>`) will also contain all policy checkpoints, JSON files with all intermediate results, etc.

### Play the policy yourself

To get a concrete sense of the search problem that the agent is learning to solve, you can use the following command to interact with the environment yourself, choose a problem and pick actions until you solve it.

```sh
[learning] $ python interact.py --environment  --domain subst-eval
```

You can type b to choose a problem, then the number of the problem. For example:

```sh
[learning] $ python interact.py --environment  --domain subst-eval
a) Type problem, b) select one from list, or Enter for debug mode: b
Pick a problem:
[...]
 2 -  (= x (+ -2 1))
[...]

> 2
### Solution:
G:(= x ?)
(= x (+ -2 1))
Action:
 0 -  eval
 1 -  rewrite
> 0
### Solution:
G:(= x ?)
(= x (+ -2 1))
eval:-###
Action:
 0 -  (= (+ -2 1) -1)
> 0
### Solution:
G:(= x ?)
(= x (+ -2 1))
eval:-(= (+ -2 1) -1)
Action:
 0 -  eval
 1 -  rewrite
> 1
### Solution:
G:(= x ?)
(= x (+ -2 1))
eval:-(= (+ -2 1) -1)
rewrite:-###
Action:
 0 -  (= (+ -2 1) (+ -2 1))
 1 -  (= x -1)
 2 -  (= -1 -1)
> 1
Solved in 4 steps!
Solution:
 interactive: ?0 <- eval ?a@*; ?1 <- rewrite ?0, ?a@*
Arguments: [['equation@type@2'], ['!step0', 'equation@type@2']]
Probability of this trajectory for a random policy: 0.08333333333333333
```

If you want to be more adventurous, you can try one of the equation domains interactively, like two-step-eq. It will quickly get tedious to solve these problems by hand from the low-level axioms, but you can get a sense of the action space in this way.
