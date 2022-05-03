#!/usr/bin/env python3

from dataclasses import dataclass, field
from typing import Any
import logging
import random
from domain import Domain

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from transformers import ReformerModelWithLMHead, ReformerConfig

from environment import Universe


logger = logging.getLogger(__name__)


@dataclass
class Episode:
    initial_observation: str
    success: bool = False
    actions: list[tuple[str, str]] = field(default_factory=list)
    states: list[str] = field(default_factory=list)
    negative_actions: list[list[str]] = field(default_factory=list)
    negative_outcomes: list[list[str]] = field(default_factory=list)

PAD = 0
BOS = 1
EOS = 2
POSITIVE = ord(';')
NEGATIVE = ord('$')
EMPTY = '\x03'

def encode_batch(b: list[str], device: torch.device, bos=True, eos=True) -> torch.LongTensor:
    if not b:
        return torch.tensor(dtype=torch.long, device=device)

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

    def rollout(self, domain: Domain, problem: Universe,
                depth: int, temperature: float = 1) -> Episode:
        with torch.no_grad():
            episode = Episode(problem.description)

            state = domain.state(problem)

            for i in range(depth):
                if domain.reward(problem):
                    break

                actions = domain.actions(problem)
                arrow_scores = self.score_arrows(actions, state)
                sampled_arrow = actions[Categorical((arrow_scores / temperature)
                                                    .softmax(-1)).sample()]

                outcomes = problem.universe.apply(sampled_arrow)
                if outcomes:
                    outcomes_scores = self.score_outcomes(list(map(str, outcomes)), sampled_arrow, state)
                    sampled_outcome = outcomes[Categorical((outcomes_scores / temperature)
                                                           .softmax(-1)).sample()]
                    problem.universe.define(f'!sub{i}', sampled_outcome)
                else:
                    sampled_outcome = EMPTY

                state = domain.state(problem)

                episode.states.append(state)
                episode.actions.append((sampled_arrow, str(sampled_outcome)))
                episode.negative_actions.append([a for a in actions if a != sampled_arrow])
                episode.negative_outcomes.append([o for o in outcomes if o != sampled_outcome])

            episode.success = domain.reward(problem)
            return episode


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

        state = encode_batch([f'S {state}'],
                             self.lm.device,
                             eos=False)[0]
        P = encode_batch([prefix for _ in continuations],
                         bos=False, eos=False, device=state.device)
        C = encode_batch(continuations,
                         bos=False, eos=False, device=state.device)

        input_ids = torch.cat((state.repeat((len(C), 1)), P, C), dim=1)
        output = self.lm(input_ids, attention_mask=(input_ids != PAD).float())

        prediction = output.logits.softmax(dim=-1)

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


if __name__ == '__main__':
    import environment
    from omegaconf import DictConfig, OmegaConf
    e = environment.SingleDomainEnvironment('equations')

    arrows = e.sample_problem(0).actions()

    cfg = DictConfig({'reformer': {'hidden_size': 256,
                                   'n_hidden_layers': 1,
                                   'n_attention_heads': 1}})
    policy = DecisionTransformer(cfg, arrows)
    policy.eval()

    problem = e.sample_problem(10)
    episode = policy.rollout(problem, 10)

    print('Episode:', episode)
