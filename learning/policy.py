#!/usr/bin/env python3

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.distributions.categorical import Categorical

from environment import Universe


@dataclass
class Episode:
    initial_observation: str
    success: bool = False
    actions: list[tuple[str, str]] = field(default_factory=list)
    negative_actions: list[list[str]] = field(default_factory=list)
    negative_outcomes: list[list[str]] = field(default_factory=list)

PAD = 0
BOS = 1
EOS = 2
EMPTY = '\x03'

def encode_batch(b: list[str], device: torch.device) -> torch.LongTensor:
    if not b:
        return torch.LongTensor()

    max_len = max(map(len, b))

    return torch.LongTensor([[BOS] + list(map(ord, o)) + [EOS] + [PAD] * (max_len - len(o))
                             for o in b],
                            device=device)

class Policy(nn.Module):
    def __init__(self):
        super().__init__()

    def score_arrows(self, arrows: list[str], state: torch.Tensor) -> torch.Tensor:
        'Scores the arrows that can be called.'
        raise NotImplementedError()

    def score_outcomes(self, outcomes: list[str], state: torch.Tensor) -> torch.Tensor:
        'Scores the results that were produced by a given arrow.'
        raise NotImplementedError()

    def initial_state(self, observation: str) -> torch.Tensor:
        'Returns the initial hidden state of the policy given the starting observation.'
        raise NotImplementedError()

    def next_state(self, state: torch.Tensor, observation: str) -> torch.Tensor:
        'Implements the recurrent rule to update the hidden state.'
        raise NotImplementedError()

    def rollout(self, problem: Universe, depth: int) -> Episode:
        with torch.no_grad():
            initial_observation = problem.starting_state()

            episode = Episode(initial_observation)
            actions = problem.actions()

            state = self.initial_state(initial_observation)

            for i in range(depth):
                arrow_scores = self.score_arrows(actions, state)
                sampled_arrow = actions[Categorical(arrow_scores.softmax(-1)).sample()]

                outcomes = problem.apply(sampled_arrow)
                if outcomes:
                    outcomes_scores = self.score_outcomes(list(map(str, outcomes)), state)
                    sampled_outcome = outcomes[Categorical(outcomes_scores.softmax(-1)).sample()]
                    problem.define(f'r{i}', sampled_outcome)
                else:
                    sampled_outcome = EMPTY

                state = self.next_state(state, str(sampled_outcome))

                episode.actions.append((sampled_arrow, str(sampled_outcome)))
                episode.negative_actions.append([a for a in actions if a != sampled_arrow])
                episode.negative_outcomes.append([o for o in outcomes if o != sampled_outcome])

            episode.success = problem.reward()
            return episode


class RNNObservationEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        embedding_size = 64
        hidden_size = 128
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
        hidden_size = 128

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

if __name__ == '__main__':
    import environment
    e = environment.SingleDomainEnvironment('equations')

    arrows = e.sample_problem(0).actions()

    policy = GRUPolicy({}, arrows)

    problem = e.sample_problem(10)
    episode = policy.rollout(problem, 10)

    print('Episode:', episode)
