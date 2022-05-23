#!/usr/bin/env python3

import random
import time
import pickle

from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from domain import make_domain, Domain
from environment import Universe, Definition
from policy import Episode, make_updated_universe


def pick_equality(state: list[(list[str], str)]) -> (str, str):
    random.shuffle(state)

    for vals, _ in state:
        vals = list(set(vals))
        if len(vals) > 1:
            # TODO: It could be good to take a random string realization
            # of the term, instead of the one in the state, to add more
            # variation to the training data and reinforce the underlying
            # notion of equivalence.
            return random.sample(vals, k=2)

    return None


def equality_holds_after_replay(u: Universe, actions: list, equality: (str, str)):
    for i, (a_i, o_i) in enumerate(actions):
        outcomes = u.apply(a_i)

        for o in outcomes:
            if o.clean_str(u) == o_i:
                u = make_updated_universe(u, o, f'!subd{i}')
                break

    return u.are_equivalent(equality[0], equality[1])


def generate_pretraining_episode(d: Domain, max_steps: int, max_state_length: int, seed: int):
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

        outcome = random.choice(outcomes)
        outcome_str = outcome.clean_str(u)

        neg_actions = [a for a in actions if a != action]
        neg_outcomes = [o.clean_str(u) for o in outcomes if o is not outcome]

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
    equality = pick_equality(u.state(d.ignore))

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


@hydra.main(config_path="config", config_name="pretraining")
def main(cfg: DictConfig):
    domain = make_domain(cfg.domain)
    n = cfg.number_of_episodes

    if 'shard' in cfg:
        output_path = cfg.output_path.format(cfg.shard)
        random.seed(f'shard-{cfg.shard}')
    else:
        output_path = cfg.output_path

    episodes = []

    with tqdm(total=n) as pbar:
        while len(episodes) < n:
            e = generate_pretraining_episode(domain,
                                             cfg.max_steps,
                                             cfg.max_state_length,
                                             random.randint(0, 10**7))
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

if __name__ == '__main__':
    main()
