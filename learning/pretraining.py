#!/usr/bin/env python3

import random
import time
import pickle
import os
import collections

import torch
import wandb
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from transformers import AdamW

from domain import make_domain, Domain
from environment import Universe, Definition
from policy import Episode, make_updated_universe, make_policy
from util import shuffle_state, randomly_mask_goal_terms, sample_batch, format_parameter_count
from main import setup_wandb


def pick_equality(state: list[(list[str], str)], counts: dict) -> (str, str):
    all_equalities, weights = [], []

    for vals, _ in state:
        # NOTE: It could be good to take a random string realization
        # of each term, instead of the one in the state, to add more
        # variation to the training data and reinforce the underlying
        # notion of equivalence.
        vals = list(set(vals))
        for i, v1 in enumerate(vals):
            for v2 in vals[i+1:]:
                counts[(v1, v2)] += 1
                counts[(v2, v1)] += 1
                all_equalities.append((v1, v2))
                weights.append(1 / counts[(v1, v2)])

    return random.choices(all_equalities, weights, k=1)[0]


def equality_holds_after_replay(u: Universe, actions: list, equality: (str, str)):
    for i, (a_i, o_i) in enumerate(actions):
        outcomes = u.apply(a_i)

        for o in outcomes:
            if o.clean_str(u) == o_i:
                u = make_updated_universe(u, o, f'!subd{i}')
                break

    return u.are_equivalent(equality[0], equality[1])


def generate_pretraining_episode(d: Domain,
                                 max_steps: int,
                                 max_state_length: int,
                                 seed: int,
                                 counts: dict):
    # Generates a random episode with the following steps:
    # 1- Sample a problem / starting state.
    # 2- Take random actions / choose random outcomes.
    #    Ignore steps with no outcomes.
    # 3- Take the final state and choose one equality at random from it.
    # 4- Remove all the unnecessary steps to get that equality.
    # 5- Return the resulting episode.

    # Step #1: get starting state
    problem = d.generate(seed)
    # print('Problem:', problem.description)

    u = problem.universe
    universes = [u]
    n_steps = 0

    e = Episode(problem.description)
    e.states.append(d.state(u))

    # Step #2: random rollout
    while n_steps < max_steps:
        actions = d.actions(u)

        if not actions:
            break

        action = random.choice(actions)

        outcomes = u.apply(action)

        if not outcomes:
            # Ignore this step if the action produced no outcomes.
            continue

        outcomes_strs = [o.clean_str(u) for o in outcomes]

        for o in outcomes_strs:
            counts[o] += 1

        weights = [1 / counts[o] for o in outcomes_strs]

        o_idx = random.choices(list(range(len(outcomes_strs))), weights=weights, k=1)[0]
        outcome = outcomes[o_idx] # random.choice(outcomes)
        outcome_str = outcome.clean_str(u)

        neg_actions = [a for a in actions if a != action]
        neg_outcomes = outcomes_strs[:o_idx] + outcomes_strs[o_idx + 1:]

        u = make_updated_universe(u, outcome, '!subd{n_steps}')

        s = d.state(u)

        if len(s) > max_state_length:
            # print('State exploded after', n_steps, 'steps.')
            break

        e.states.append(s)
        e.actions.append((action, outcome_str))
        e.negative_actions.append(neg_actions)
        e.negative_outcomes.append(neg_outcomes)
        universes.append(u)

        n_steps += 1

    # Step #3: choose random equality.
    equality = pick_equality(u.state(d.ignore), counts)

    if equality is None:
        raise ValueError('No equality to be extracted.')

    # Step #4: remove all actions that are actually not needed to get this equality.
    needed = [True] * len(e.actions)

    for i in range(len(e.actions)):
        # Try to replay the episode without action i.
        actions_without_i = [a for j, a in enumerate(e.actions) if needed[j] and j != i]

        if equality_holds_after_replay(universes[0], actions_without_i, equality):
            needed[i] = False

    # Step #5: make and return the resulting cleaned-up episode.
    ce = Episode(e.initial_observation)
    ce.goal = f'(= {equality[0]} {equality[1]})'
    ce.actions = [e.actions[i] for i in range(len(e.actions)) if needed[i]]
    ce.states = [e.states[0]]

    u = universes[0]

    for i, (a_i, o_i) in enumerate(ce.actions):
        actions = d.actions(u)

        outcomes = u.apply(a_i)
        o_defs = [o for o in outcomes if o.clean_str(u) == o_i]

        if len(o_defs) == 0:
            pass
            # print('Did not find definition', o_i, 'in',
            #      ', '.join(o.clean_str(u) for o in outcomes), 'after applying', a_i)
        else:
            o_def = o_defs[0]

            ce.negative_actions.append([a for a in actions if a != a_i])
            ce.negative_outcomes.append([o.clean_str(u) for o in outcomes if o != o_def])

            u.define(f'!subd{i}', o_def)
            ce.states.append(d.state(u))

    return ce


def train(cfg: DictConfig):
    episodes = []
    datasets = []

    # Load training episodes.
    if 'shards' in cfg:
        for i in range(cfg.shards):
            datasets.append(cfg.dataset.format(i))
    else:
        datasets.append(cfg.dataset)

    for path in datasets:
        with open(path, 'rb') as f:
            episodes.extend(pickle.load(f))

    policy = make_policy(cfg.policy)

    print('Model has', format_parameter_count(policy), 'trainable parameters.')

    if cfg.policy.get('gpu') is not None:
        policy = policy.to(torch.device(cfg.policy.gpu))

    examples = []

    print('Extracting examples from episodes...')

    episodes = episodes[:cfg.get('max_episodes', len(episodes))]

    for e in tqdm(episodes):
        e.success = True

        if len(e.negative_actions) != len(e.actions):
            continue

        for _ in range(cfg.n_augmentations):
            examples.extend(policy.extract_examples(
                e,
                transform_state=shuffle_state,
                transform_goal=lambda g: randomly_mask_goal_terms(
                    g, cfg.goal_mask_probability)))

    print(len(examples), 'examples.')

    if cfg.get('debug_examples'):
        breakpoint()

    setup_wandb(cfg)
    print('Saving checkpoints at', os.getcwd())

    optimizer = AdamW(policy.parameters(), lr=cfg.learning_rate)

    batch_size = cfg.batch_size
    policy.train()
    n_checkpoints = 0

    def checkpoint():
        nonlocal n_checkpoints
        torch.save(policy, f'lm.{n_checkpoints}.pt')
        n_checkpoints += 1

    # FIXME: We should use a lightning Trainer instead.
    for i in tqdm(range(cfg.gradient_steps)):
        batch = sample_batch(examples, batch_size)
        if not batch:
            continue

        optimizer.zero_grad()
        loss = policy.get_loss(batch)
        loss.backward()
        optimizer.step()
        wandb.log({'train_loss': loss})

        if i % cfg.checkpoint_every == 0:
            checkpoint()

    checkpoint()


def generate(cfg: DictConfig):
    domain = make_domain(cfg.domain)
    n = cfg.number_of_episodes

    if 'shard' in cfg:
        output_path = cfg.output_path.format(cfg.shard)
        random.seed(f'shard-{cfg.shard}')
    else:
        output_path = cfg.output_path

    episodes = []
    counts = collections.defaultdict(int)

    with tqdm(total=n) as pbar:
        while len(episodes) < n:
            e = generate_pretraining_episode(domain,
                                             cfg.max_steps,
                                             cfg.max_state_length,
                                             random.randint(0, 10**7),
                                             counts)
            if e is not None:
                episodes.append(e)
                pbar.update(1)

                if len(episodes) % cfg.save_every == 0:
                    with open(output_path, 'wb') as f:
                        pickle.dump(episodes, f)
                    print('Wrote', output_path)

    with open(output_path, 'wb') as f:
        pickle.dump(episodes, f)
    print('Wrote', output_path)


@hydra.main(version_base="1.2", config_path="config", config_name="pretraining")
def main(cfg: DictConfig):
    if cfg.task == 'train':
        train(cfg)
    elif cfg.task == 'generate':
        generate(cfg)
    else:
        raise ValueError(f'Unknown pretraining command {cfg.task}')


if __name__ == '__main__':
    main()
