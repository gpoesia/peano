#!/usr/bin/env python3

import collections
from dataclasses import dataclass, field
from typing import Any, Optional
import logging
import random
from domain import Domain, EquationsDomain, Problem
from enum import Enum

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from transformers import ReformerModelWithLMHead, ReformerConfig

from environment import Universe
from util import log, softmax, pop_max


logger = logging.getLogger(__name__)


@dataclass
class SearchNode:
    universe: Universe = None
    state: str = None
    value: float = 0.0
    parent: 'SearchNode' = None
    action: str = None
    outcome: str = None
    reward: bool = False
    value_target: float = None
    depth: int = 0

    def expand(self, domain: Domain) -> list['SearchNode']:
        c = []

        if self.action:
            # Expand outcomes.
            for o in self.universe.apply(self.action):
                u = make_updated_universe(self.universe, o, f'!subd{self.depth}')
                c.append(SearchNode(u, domain.state(u),
                                    depth=self.depth + 1,
                                    value=None, parent=self,
                                    action=None, outcome=o.clean_str(self.universe),
                                    reward=domain.reward(u)))
        else:
            # Expand actions.
            for a in domain.actions(self.universe):
                c.append(SearchNode(self.universe, f'S: {self.state} A: {a}', depth=self.depth,
                                    value=None, parent=self, action=a, reward=self.reward))

        return c

    def __getstate__(self):
        'Prevent pickle from trying to save the universe.'
        d = self.__dict__.copy()
        del d['universe']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)


@dataclass
class Episode:
    initial_observation: str
    goal: str = None
    success: bool = False
    actions: list[tuple[str, str]] = field(default_factory=list)
    states: list[str] = field(default_factory=list)
    negative_actions: list[list[str]] = field(default_factory=list)
    negative_outcomes: list[list[str]] = field(default_factory=list)
    searched_nodes: list[SearchNode] = None


@dataclass
class TreeSearchEpisode:
    initial_observation: str
    success: bool = False
    visited: list[SearchNode] = field(default_factory=list)
    goal_state: Optional[SearchNode] = None


PAD = 0
BOS = 1
EOS = 2
POSITIVE = ord(';')
NEGATIVE = ord('$')
EMPTY = '\x03'

EMPTY_OUTCOME_PROBABILITY = 1e-3

def encode_batch(b: list[str], device: torch.device, bos=True, eos=True) -> torch.LongTensor:
    if not b:
        return torch.tensor([], dtype=torch.long, device=device)

    max_len = max(map(len, b))

    return torch.tensor([[BOS] * bos +
                         list(map(ord, o)) +
                         [EOS] * eos +
                         [PAD] * (max_len - len(o))
                         for o in b],
                        dtype=torch.long,
                        device=device)

def decode_batch(b: torch.LongTensor) -> list[str]:
    return [''.join(chr(c) for c in row if c > EOS) for row in b]

def make_updated_universe(universe, definition, name):
    'Returns a clone of `universe` where `definition` is defined with the given name.'
    if definition == EMPTY:
        return universe
    u = universe.clone()
    u.define(name, definition)
    return u

def recover_episode(problem, final_state, success):
    states, actions, negative_actions, negative_outcomes = [], [], [], []

    current = final_state

    while current != None:
        states.append(current.state)

        # The first action is associated with the second state, so for the
        # first state there's no action preceding it. Thus, `states` is one element
        # larger than the other lists.
        if current.parent is not None:
            actions.append((current.action, current.outcome))
            negative_actions.append(current.negative_actions)
            negative_outcomes.append(current.negative_outcomes)

        current = current.parent

    return Episode(problem.description,
                   success,
                   actions[::-1],
                   states[::-1],
                   negative_actions[::-1],
                   negative_outcomes[::-1])

@dataclass
class BeamElement:
    universe: object
    state: str
    action: Optional[str] = None
    outcome: Optional[str] = None
    parent: Optional['BeamElement'] = None
    logprob: float = 0
    negative_actions: list = field(default_factory=list)
    negative_outcomes: list = field(default_factory=list)

    def __str__(self) -> str:
        return f'BeamElement({self.state}, logprob={self.logprob})'


class Policy(nn.Module):
    def __init__(self):
        super().__init__()

    def score_arrows(self, arrows: list[str], state: Any) -> torch.Tensor:
        'Scores the arrows that can be called.'
        raise NotImplementedError()

    def score_outcomes(self, outcomes: list[str], state: Any) -> torch.Tensor:
        'Scores the results that were produced by a given arrow.'
        raise NotImplementedError()

    def initial_state(self, observation: str) -> Any:
        'Returns the initial hidden state of the policy given the starting observation.'
        raise NotImplementedError()

    def next_state(self, state: Any, observation: str) -> Any:
        'Implements the recurrent rule to update the hidden state.'
        raise NotImplementedError()

    def rollout(self, domain: Domain, problem: Problem,
                depth: int,
                temperature: float = 1,
                beam_size: int = 1,
                epsilon: float = 0) -> Episode:

        with torch.no_grad():
            beam = [BeamElement(universe=problem.universe, 
                                state=domain.state(problem.universe),
                                logprob=0.0)]

            # Each iteration happens in five stages:
            # 0- Check if a solution was found
            # 1- Score arrows for each state in the beam
            # 2- Get an intermediate beam with top state/arrow pairs
            # 3- Score arrow outcomes for each state/arrow in the intermediate beam
            # 4- Filter top outcomes and apply outcome to states to obtain next beam
            for it in range(depth):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'Beam #{it}:')
                    for e in beam:
                        logger.debug('  %s', e)

                # 0- Check if a solution was found
                solution = next((s for s in beam if domain.reward(s.universe)), None)

                if solution is not None:
                    logger.debug('Solution state: %s', solution)
                    return recover_episode(problem, solution, True)

                # epsilon-greedy: pick random actions in this iteration with probability eps.
                take_random_action = random.random() < epsilon

                # 1- Score arrows for each state in the beam
                actions = [domain.actions(s) for s in beam]
                arrow_probabilities = [(self.score_arrows(a, s.state) / temperature).softmax(-1)
                                       for a, s in zip(actions, beam)]

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('Arrow probabilities:')
                    for i, s in enumerate(beam):
                        logger.debug('  %s => %s', s.state,
                                     sorted(list(zip(actions[i], arrow_probabilities[i])),
                                            key=lambda aap: aap[1], reverse=True))

                beam = [BeamElement(universe=s.universe,
                                    state=s.state,
                                    action=a,
                                    outcome=None,
                                    logprob=s.logprob + log(arrow_probabilities[i][j].item()),
                                    parent=s,
                                    negative_actions=actions[i][:j] + actions[i][j+1:],
                                    negative_outcomes=None)
                        for i, s in enumerate(beam)
                        for j, a in enumerate(actions[i])]

                # 2- Get an intermediate beam with top state/arrow pairs.
                if take_random_action:
                    beam = random.sample(beam, k=min(len(beam), beam_size))
                else:
                    beam = sorted(beam, key=lambda s: -s.logprob)[:beam_size]

                # 3- Score arrow outcomes.
                outcomes = [s.universe.apply(s.action) for s in beam]
                outcome_probabilities = [(self.score_outcomes([o.clean_str(s.universe) for o in outs],
                                                              s.action,
                                                              s.state) / temperature).softmax(-1)
                                         for outs, s in zip(outcomes, beam)]

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('Outcome probabilities:')
                    for i, s in enumerate(beam):
                        logger.debug('  %s / %s => %s', s.state, s.action,
                                     sorted(list(zip(outcomes[i], outcome_probabilities[i])),
                                            key=lambda oop: oop[1], reverse=True))

                new_beam = []
                for i, s in enumerate(beam):
                    # If an arrow did not produce any outcomes, that path would
                    # simply disappear in the next beam. This would make it possible
                    # for the beam to get empty (especially likely with a beam size of 1).
                    # To avoid that, we add the same state with an EMPTY outcome to new_beam
                    # with a very low probability to penalize it (but it will be picked up
                    # if it's the only option).
                    if not outcomes[i]:
                        outcomes[i].append(EMPTY)
                        outcome_probabilities[i] = torch.tensor([EMPTY_OUTCOME_PROBABILITY])

                    for j, o in enumerate(outcomes[i]):
                        u = make_updated_universe(s.universe, o, f'!subd{it}')

                        new_beam.append(BeamElement(
                            universe=u,
                            state=domain.state(u),
                            action=s.action,
                            outcome=o.clean_str(u) if not isinstance(o, str) else o,
                            logprob=s.logprob + log(outcome_probabilities[i][j].item()),
                            parent=s.parent,
                            negative_actions=s.negative_actions,
                            negative_outcomes=[o_k.clean_str(u)
                                               for k, o_k in enumerate(outcomes[i])
                                               if k != j]))

                # 4- Obtain next beam.
                if take_random_action:
                    beam = random.sample(new_beam, k=min(len(new_beam), beam_size))
                else:
                    beam = sorted(new_beam, key=lambda s: -s.logprob)[:beam_size]

            return recover_episode(problem, beam[0], False)

    def best_first_search(self, domain: Domain, problem: Problem,
                          max_nodes: int) -> Episode:
        root = SearchNode(problem.universe,
                          domain.state(problem.universe),
                          reward=domain.reward(problem.universe))
        with torch.no_grad():
            root.value = self.estimate_values([root.state]).item()
        queue = [root]
        visited = []
        goal_state = root if root.reward else None

        while queue and goal_state is None and len(visited) < max_nodes:
            node, queue = pop_max(queue, lambda node: node.value)
            visited.append(node)

            logger.debug('Visiting %s (estimated value: %f)', node.state, node.value)

            children = node.expand(domain)

            if children:
                with torch.no_grad():
                    children_values = self.estimate_values([c.state for c in children])

                for c, v in zip(children, children_values):
                    c.value = v
                    logger.debug('\tEstimated value for children %s / %s: %f',
                                 c.state, c.action, c.value)

                    if c.reward:
                        goal_state = c

                queue.extend(children)

        # For all nodes that are not in the path to the solution, aim to reduce their
        # value estimates. This will happen to all nodes in case no solution is found
        # and the agent doesn't ignore unsolved problems.
        for node in visited:
            node.value_target = 0

        if goal_state:
            visited.append(goal_state)
            node = goal_state
            value = 1.0
            while node is not None:
                node.value_target = value
                value *= 0.98
                node = node.parent

        return TreeSearchEpisode(problem.description,
                                 goal_state is not None,
                                 visited,
                                 goal_state)

    def extract_examples(self, episode) -> list[str]:
        raise NotImplementedError()

    def get_loss(self, batch) -> torch.Tensor:
        raise NotImplementedError()

    def embed_raw(self, strs: list[str]) -> torch.Tensor:
        raise NotImplementedError()

    def embed_states(self, batch: list[str]) -> torch.Tensor:
        return self.embed_raw([f'S{s}S' for s in batch])

    def embed_arrows(self, batch: list[str]) -> torch.Tensor:
        return self.embed_raw([f'A{s}A' for s in batch])

    def embed_outcomes(self, batch: list[str]) -> torch.Tensor:
        batch = batch or [EMPTY]
        return self.embed_raw([f'O{s}O' for s in batch])

class RNNObservationEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        embedding_size = 16
        hidden_size = 32
        layers = 1

        self.rnn = nn.GRU(embedding_size, hidden_size // 2, layers, bidirectional=True)
        self.char_embedding = nn.Embedding(128, embedding_size)

    def forward(self, observations: list[str]) -> torch.Tensor:
        o = encode_batch(observations, self.char_embedding.weight.device)
        o = o.transpose(0, 1)
        o_char_emb = self.char_embedding(o)
        out, h_n = self.rnn(o_char_emb)
        return h_n[-2:, :, :].transpose(0, 1).reshape((len(observations), -1))


class GRUPolicy(Policy):
    def __init__(self, config, all_arrows):
        super().__init__()
        hidden_size = 32

        self.rnn_cell = nn.GRUCell(hidden_size, hidden_size)

        self.arrow_readout = nn.Linear(hidden_size, hidden_size)
        self.outcome_readout = nn.Linear(hidden_size, hidden_size)

        self.embedding = RNNObservationEmbedding({})
        self.arrow_embedding = nn.Embedding(len(all_arrows), hidden_size)
        self.arrow_to_index = {a: i for i, a in enumerate(all_arrows)}

    def initial_state(self, observation: str) -> torch.Tensor:
        return self.embedding([observation])[0]

    def next_state(self, state: torch.Tensor, observation: str):
        obs_emb = self.embedding([observation])[0]
        return self.rnn_cell(obs_emb, state)

    def score_arrows(self, arrows: list[str], state: torch.Tensor) -> torch.Tensor:
        arrow_index = torch.LongTensor([self.arrow_to_index[a] for a in arrows],
                                       device=self.arrow_embedding.weight.device)
        arrow_embeddings = self.arrow_embedding(arrow_index)
        readout = self.arrow_readout(state)
        H = readout.shape[0]
        return arrow_embeddings.matmul(readout.unsqueeze(1)).squeeze(1) / H

    def score_outcomes(self, outcomes: list[str], state: torch.Tensor) -> torch.Tensor:
        outcome_embeddings = self.embedding(outcomes)
        readout = self.outcome_readout(state)
        H = readout.shape[0]
        return outcome_embeddings.matmul(readout.unsqueeze(1)).squeeze(1) / H


class RandomPolicy(Policy):
    def __init__(self, config, all_arrows):
        super().__init__()

    def initial_state(self, observation: str) -> torch.Tensor:
        return torch.tensor([])

    def next_state(self, state: torch.Tensor, observation: str):
        return state

    def score_arrows(self, arrows: list[str], state: torch.Tensor) -> torch.Tensor:
        return torch.rand((len(arrows),))

    def score_outcomes(self, outcomes: list[str], state: torch.Tensor) -> torch.Tensor:
        return torch.rand((len(outcomes),))


class DecisionTransformer(Policy):
    def __init__(self, config):
        super().__init__()

        configuration = ReformerConfig(
            vocab_size=128,
            attn_layers=['local', 'lsh'] * (config.reformer.num_hidden_layers // 2),
            #            axial_pos_shape=(32, 32), # Default (64, 64) -- must multiply to seq len when training
            # Default (64, 64) -- must multiply to seq len when training
            axial_pos_embds=(64, config.reformer.hidden_size - 64),
            bos_token_id=BOS,
            eos_token_id=EOS,
            pad_token_id=PAD,
            is_decoder=True,
            **config['reformer']
        )

        # Initializing a Reformer model
        self.lm = ReformerModelWithLMHead(configuration)
        self.train_len_multiple = 64*64
        self.batch_size = 4000
        self.mask_non_decision_tokens = config.mask_non_decision_tokens

    def initial_state(self, observation: str) -> torch.Tensor:
        raise NotImplementedError()
        return encode_batch([f'G (= x ?);S {observation}'],
                            self.lm.device,
                            eos=False)[0]

    def next_state(self, state: torch.Tensor, action: str, observation: str):
        raise NotImplementedError()
        return torch.cat((state,
                          encode_batch([f';A {action};O {observation}'],
                                       device=state.device, bos=False, eos=False)[0]))

    def score_arrows(self, arrows: list[str], state: str) -> torch.Tensor:
        return self._score_continuations(state, ';A ', arrows)

    def score_outcomes(self, outcomes: list[str], action: str, state: str) -> torch.Tensor:
        return self._score_continuations(state, f';A {action};O ', outcomes)

    def _score_continuations(self,
                             state: str,
                             prefix: str,
                             continuations: list[str]) -> torch.Tensor:
        if not continuations:
            return torch.tensor([])

        state = encode_batch([f'S {state}'],
                             self.lm.device,
                             eos=False)[0]
        P = encode_batch([prefix for _ in continuations],
                         bos=False, eos=False, device=state.device)
        C = encode_batch(continuations,
                         bos=False, eos=False, device=state.device)

        input_ids = torch.cat((state.repeat((len(C), 1)), P, C), dim=1)

        # Run the LM on smaller batches if needed to avoid running it on
        # more than self.batch_size tokens at a time.
        outputs = []
        batch_rows = max(1, self.batch_size // input_ids.shape[1])

        for row in range((input_ids.shape[0] + batch_rows - 1) // batch_rows):
            i = row * batch_rows
            j = min(i + batch_rows, input_ids.shape[0])
            X = input_ids[i:j, :]
            outputs.append(self.lm(X, attention_mask=(X != PAD).float()).logits)

        output = torch.cat(outputs, dim=0)

        prediction = output.softmax(dim=-1)

        skip = state.shape[0] + P.shape[1]
        action_predictions = prediction[range(len(continuations)),
                                        [skip + len(c) - 1 for c in continuations],
                                        :]

        pos_logit = action_predictions[:, POSITIVE]
        neg_logit = action_predictions[:, NEGATIVE]
        scores = pos_logit - neg_logit

        logger.debug('{"location": "_score_continuations", "input_ids.shape": %s, "prefix": "%s",'
                     '"continuations": %s, "scores": %s}',
                     input_ids.shape, prefix, continuations, scores)

        return scores

    def pad_train_batch(self, tensor: torch.Tensor):
        m = self.train_len_multiple
        n = tensor.shape[-1]
        next_multiple_of_m = (n + m - 1) // m * m
        return F.pad(tensor, (0, next_multiple_of_m - n))

    def extract_examples(self, episode) -> list[str]:
        if not episode.success:
            return []

        # Positive.
        def format_example(s, a, c):
            return f'S {s}; {a}{c}'

        examples = []

        for i, (a, o) in enumerate(episode.actions):
            # Negative examples of actions.
            examples.extend([format_example(episode.states[i],
                                            f'A {neg}', chr(NEGATIVE))
                             for neg in episode.negative_actions[i]])

            # Negative examples of outcomes.
            examples.extend([format_example(episode.states[i],
                                            f'A {a}; O {neg}', chr(NEGATIVE))
                             for neg in episode.negative_outcomes[i]])

            # Positives
            examples.append(format_example(episode.states[i],
                                           f'A {a}', chr(POSITIVE)))
            examples.append(format_example(episode.states[i],
                                           f'A {a}; O {o}', chr(POSITIVE)))

        return examples

    def get_loss(self, batch) -> torch.Tensor:
        t = encode_batch(batch, self.lm.device)

        # NOTE: The Reformer implementation already shifts X and y.
        # Normally, we'd have to do this manually.
        X = self.pad_train_batch(t)
        y = X.clone()

        # Ignore non-decision tokens (or at least PAD tokens) when computing the loss
        # (-100 is the label mask ID from the huggingface API).
        if self.mask_non_decision_tokens:
            y[(y != POSITIVE) & (y != NEGATIVE)] = -100
        else:
            y[y == PAD] = -100

        output = self.lm(X,
                         attention_mask=(X != PAD).float(),
                         labels=y)

        return output.loss


class DecisionGRU(Policy):
    def __init__(self, config):
        super().__init__()

        self.lm = nn.GRU(input_size=config.gru.embedding_size,
                         hidden_size=config.gru.hidden_size,
                         num_layers=config.gru.layers)

        self.output = nn.Linear(config.gru.hidden_size, 128)
        self.embedding = nn.Embedding(128, config.gru.embedding_size)
        self.batch_size = config.batch_size
        self.mask_non_decision_tokens = False

    def score_arrows(self, arrows: list[str], state: str) -> torch.Tensor:
        return self._score_continuations(state, ';A ', arrows)

    def score_outcomes(self, outcomes: list[str], action: str, state: str) -> torch.Tensor:
        return self._score_continuations(state, f';A {action};O ', outcomes)

    def get_device(self):
        return self.embedding.weight.device

    def _score_continuations(self,
                             state: str,
                             prefix: str,
                             continuations: list[str]) -> torch.Tensor:
        if not continuations:
            return torch.tensor([])

        S = encode_batch([f'S {state}'],
                         self.get_device(),
                         eos=False)[0]
        P = encode_batch([prefix for _ in continuations],
                         bos=False, eos=False, device=S.device)
        C = encode_batch(continuations,
                         bos=False, eos=False, device=S.device)

        input_ids = torch.cat((S.repeat((len(C), 1)), P, C), dim=1)

        # Run the LM on smaller batches if needed to avoid running it on
        # more than self.batch_size tokens at a time.
        outputs = []
        batch_rows = max(1, self.batch_size // input_ids.shape[1])

        for row in range((input_ids.shape[0] + batch_rows - 1) // batch_rows):
            i = row * batch_rows
            j = min(i + batch_rows, input_ids.shape[0])
            X = self.embedding(input_ids[i:j, :].transpose(0, 1))
            y, _ = self.lm(X)
            outputs.append(self.output(y.transpose(0, 1)))

        output = torch.cat(outputs, dim=0)

        prediction = output.softmax(dim=-1)

        skip = S.shape[0] + P.shape[1]
        action_predictions = prediction[range(len(continuations)),
                                        [skip + len(c) - 1 for c in continuations],
                                        :]

        pos_logit = action_predictions[:, POSITIVE]
        neg_logit = action_predictions[:, NEGATIVE]
        scores = pos_logit - neg_logit

        logger.debug('{"location": "_score_continuations", "input_ids.shape": %s, "prefix": "%s",'
                     '"continuations": %s, "scores": %s}',
                     input_ids.shape, prefix, continuations, scores)

        return scores

    def extract_examples(self, episode) -> list[str]:
        if not episode.success:
            return []

        # Positive.
        def format_example(s, a, c):
            return f'S {s}; {a}{c}'

        examples = []

        for i, (a, o) in enumerate(episode.actions):
            # Negative examples of actions.
            examples.extend([format_example(episode.states[i],
                                            f'A {neg}', chr(NEGATIVE))
                             for neg in episode.negative_actions[i]])

            # Negative examples of outcomes.
            examples.extend([format_example(episode.states[i],
                                            f'A {a}; O {neg}', chr(NEGATIVE))
                             for neg in episode.negative_outcomes[i]])

            # Positives
            examples.append(format_example(episode.states[i],
                                           f'A {a}', chr(POSITIVE)))
            examples.append(format_example(episode.states[i],
                                           f'A {a}; O {o}', chr(POSITIVE)))

        return examples

    def get_loss(self, batch) -> torch.Tensor:
        t = encode_batch(batch, self.get_device()).transpose(0, 1)

        X = self.embedding(t[:-1, :])
        y = t[1:, :].clone()

        # Ignore non-decision tokens (or at least PAD tokens) when computing the loss
        # (-100 is the label mask ID for cross_entropy_loss).
        if self.mask_non_decision_tokens:
            y[(y != POSITIVE) & (y != NEGATIVE)] = -100
        else:
            y[y == PAD] = -100

        y_hat, _ = self.lm(X)
        output = self.output(y_hat)

        # output shape is (L, N, C), cross_entropy needs (N, C, L).
        return F.cross_entropy(output.permute((1, 2, 0)), y.transpose(0, 1))


class ExampleType(Enum):
    STATE_ACTION = 1
    STATE_OUTCOME = 2
    STATE_VALUE = 3


@dataclass
class ContrastivePolicyExample:
    type: ExampleType
    state: str
    positive: str = None
    negatives: list[str] = None
    value: float = None

    def __len__(self):
        return (len(self.state) +
                len(self.positive or '') + 
                sum(map(len, self.negatives or [])))


class ContrastivePolicy(Policy):
    def __init__(self, config):
        super().__init__()

        self.lm = nn.GRU(input_size=config.gru.embedding_size,
                         hidden_size=config.gru.hidden_size,
                         bidirectional=True,
                         num_layers=config.gru.layers)

        self.arrow_readout = nn.Linear(2*config.gru.hidden_size, 2*config.gru.hidden_size)
        self.outcome_readout = nn.Linear(2*config.gru.hidden_size, 2*config.gru.hidden_size)
        self.value_readout = nn.Sequential(
            nn.Linear(2*config.gru.hidden_size, 2*config.gru.hidden_size),
            nn.ReLU(),
            nn.Linear(2*config.gru.hidden_size, 1)
        )

        self.embedding = nn.Embedding(128, config.gru.embedding_size)
        self.discard_unsolved = config.discard_unsolved
        # Truncate states/actions to avoid OOMs.
        self.max_len = 300
        self.discount = 0.99

    def score_arrows(self, arrows: list[str], state: str) -> torch.Tensor:
        if len(arrows) <= 1:
            return torch.ones(len(arrows), dtype=torch.float, device=self.get_device())
        # state_embedding : (1, H)
        state_embedding = self.embed_states([state])
        # arrow_embedding : (B, H)
        arrow_embeddings = self.embed_arrows(arrows)
        # state_t : (H, 1)
        state_t = self.arrow_readout(state_embedding).transpose(0, 1)
        # Result: (B,)
        return arrow_embeddings.matmul(state_t).squeeze(1)

    def score_outcomes(self, outcomes: list[str], action: str, state: str) -> torch.Tensor:
        if len(outcomes) <= 1:
            return torch.ones(len(outcomes), dtype=torch.float, device=self.get_device())
        # state_embedding : (1, H)
        state_embedding = self.embed_states([state])
        # outcome_embeddings : (B, H)
        outcome_embeddings = self.embed_outcomes(outcomes)
        # state_t : (H, 1)
        state_t = self.outcome_readout(state_embedding).transpose(0, 1)
        # Result: (B,)
        return outcome_embeddings.matmul(state_t).squeeze(1)

    def estimate_values(self, states: list[str]) -> torch.Tensor:
        logger.debug('Estimating values for %d states, maxlen = %d',
                     len(states), max(map(len, states)))
        state_embedding = self.embed_states(states)
        return self.value_readout(state_embedding).squeeze(1)

    def get_device(self):
        return self.embedding.weight.device

    def extract_examples(self, episode) -> list[str]:
        examples = []

        if not episode.success and self.discard_unsolved:
            return examples

        if isinstance(episode, Episode):
            for i, (a, o) in enumerate(episode.actions):
                if episode.success:
                    if episode.negative_actions[i]:
                        examples.append(ContrastivePolicyExample(type=ExampleType.STATE_ACTION,
                                                                 state=episode.states[i],
                                                                 positive=a,
                                                                 negatives=episode.negative_actions[i]))

                    if episode.negative_outcomes[i]:
                        examples.append(ContrastivePolicyExample(type=ExampleType.STATE_OUTCOME,
                                                                 state=episode.states[i],
                                                                 positive=o,
                                                                 negatives=episode.negative_outcomes[i]))

            for i, s in enumerate(episode.states):
                value = (0 if not episode.success
                         else self.discount ** (len(episode.states) - (i + 1)))
                examples.append(ContrastivePolicyExample(type=ExampleType.STATE_VALUE,
                                                         state=episode.states[i],
                                                         value=value))
        elif isinstance(episode, TreeSearchEpisode):
            for node in episode.visited:
                examples.append(ContrastivePolicyExample(type=ExampleType.STATE_VALUE,
                                                         state=node.state,
                                                         value=node.value_target))

        return examples

    def get_loss(self, batch) -> torch.Tensor:
        losses = []

        # HACK: This can be vectorized & batched, but it will be more complicated. This
        # simple implementation lets us just test out the architecture.
        for e in batch:
            if e.type == ExampleType.STATE_ACTION:
                p = self.score_arrows([e.positive] + e.negatives, e.state).softmax(-1)
                y = torch.zeros_like(p)
                y[0] = 1
                losses.append(F.binary_cross_entropy(p, y))
            elif e.type == ExampleType.STATE_OUTCOME:
                p = self.score_outcomes([e.positive] + e.negatives, None, e.state).softmax(-1)
                y = torch.zeros_like(p)
                y[0] = 1
                losses.append(F.binary_cross_entropy(p, y))
            elif e.type != ExampleType.STATE_VALUE:
                raise ValueError(f'Unknown example type {e.type}')

        state_values_x = [e.state for e in batch if e.type == ExampleType.STATE_VALUE]

        if len(state_values_x):
            y = [e.value for e in batch if e.type == ExampleType.STATE_VALUE]
            y_hat = self.estimate_values(state_values_x)
            losses.append(((y_hat - torch.tensor(y, device=y_hat.device))**2).mean())

        return torch.stack(losses, dim=0).mean()

    def embed_raw(self, strs: list[str]) -> torch.Tensor:
        strs = [s[:self.max_len] for s in strs]
        input = encode_batch(strs, self.get_device(), bos=True, eos=True)
        input = self.embedding(input.transpose(0, 1))
        output, _ = self.lm(input)
        return output[0, :, :]


def make_policy(config):
    if 'type' not in config:
        raise ValueError(f'Policy config must have a \'type\'')
    if config.type == 'DecisionTransformer':
        return DecisionTransformer(config)
    elif config.type == 'DecisionGRU':
        return DecisionGRU(config)
    elif config.type == 'ContrastivePolicy':
        return ContrastivePolicy(config)
    raise ValueError(f'Unknown policy type {config.type}')


if __name__ == '__main__':
    import environment
    from omegaconf import DictConfig, OmegaConf
    d = EquationsDomain()

    cfg = DictConfig({'reformer': {'hidden_size': 256,
                                   'n_hidden_layers': 1,
                                   'n_attention_heads': 1},
                      'gru': {'hidden_size': 256,
                              'embedding_size': 64,
                              'layers': 2},
                      'batch_size': 512})

    # policy = DecisionTransformer(cfg, arrows)
    # policy = DecisionGRU(cfg)
    policy = ContrastivePolicy(cfg)
    policy = policy.to(torch.device(1))
    policy.eval()

    problem = d.make_problem('(= (* x 2) 3)')

    import time
    before = time.time()
    episode = policy.rollout(d, problem, depth=10, beam_size=10)
    print(time.time() - before)

    # print('Episode:', episode)
