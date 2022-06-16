#!/usr/bin/env python3

import random
import math
import logging
import hydra
import pickle
from dataclasses import dataclass
import numpy as np

import torch
import wandb
from torch import nn
from tqdm import tqdm
from omegaconf import DictConfig

from policy import encode_batch
from main import setup_wandb


logger = logging.getLogger(__name__)


class SearchHeuristic:
    'Implements the core components of a search heuristic for proof search.'

    def group(self, definition, depth) -> str:
        'Returns an arbitrary identifier to group this definition\'s priority.'
        raise NotImplementedError()

    def utility(self, problem, values) -> list[float]:
        'Estimates the utility of each definition in solving the problem.'
        raise NotImplementedError()



@dataclass
class ContrastiveExample:
    problem: str
    positive: str
    negatives: list[str]


class GRUUtilityFunction(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.lm = nn.GRU(input_size=config.gru.embedding_size,
                         hidden_size=config.gru.hidden_size,
                         bidirectional=True,
                         num_layers=config.gru.layers)

        self.utility_readout = nn.Sequential(
            nn.Linear(2*config.gru.hidden_size, config.gru.hidden_size),
            nn.ReLU(),
            nn.Linear(config.gru.hidden_size, 1)
        )

        self.embedding = nn.Embedding(128, config.gru.embedding_size)
        self.max_len = 300
        self.config = config

    def get_device(self):
        return self.embedding.weight.device

    def embed_raw(self, strs: list[str]) -> torch.Tensor:
        strs = [s[:self.max_len] for s in strs]
        lens = [len(s) + 2 for s in strs]
        X = encode_batch(strs, self.get_device(), bos=True, eos=True)
        embeddings = self.embedding(X.transpose(0, 1))
        seq = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lens, enforce_sorted=False)
        seq_out, _ = self.lm(seq)
        h_n, _ = torch.nn.utils.rnn.pad_packed_sequence(seq_out)
        return h_n[0, :, :]

    def group(self, definition, depth) -> str:
        return f'({definition.generating_action()}, {depth})'

    def utility(self, problem, values) -> torch.Tensor:
        problems = [problem] * len(values)

        if self.training:
            return self(problems, values)

        with torch.no_grad():
            return self(problems, values).cpu().tolist()

    def forward(self, problems, objects):
        strs = [f'{p} => {o}' for p, o in zip(problems, objects)]
        embeddings = self.embed_raw(strs)
        return self.utility_readout(embeddings).squeeze(-1)

    def nce_loss(self, problem, positive, negatives):
        all_examples = [positive] + negatives
        logits = self.utility(problem, all_examples).exp()
        return -(logits[0] / logits.sum()).log()

    def fit(self, dataset: list['ProofSearchEpisode'], checkpoint_callback=lambda: None):
        self.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)

        for e in range(self.config.n_epochs):
            wandb.log({'epoch': e})
            print('Epoch', e + 1)
            examples = []

            # Assemble contrastive examples
            for episode in dataset:
                if episode.success and len(episode.visited_negatives) and len(episode.discovered_negatives):
                    for positive in episode.solution:
                        negatives = (random.choices(list(episode.visited_negatives),
                                                    k=math.ceil(self.config.batch_size / 2)) +
                                     random.choices(list(episode.discovered_negatives),
                                                    k=math.floor(self.config.batch_size / 2)))
                        examples.append(ContrastiveExample(episode.problem, positive, negatives))

            random.shuffle(examples)

            for example in tqdm(examples):
                optimizer.zero_grad()
                loss = self.nce_loss(example.problem, example.positive, example.negatives)
                loss.backward()
                optimizer.step()

                wandb.log({'train_loss': loss.cpu()})

            checkpoint_callback()


class LengthUtilityFunction:
    def __init__(self):
        pass

    def group(self, definition, depth):
        return f'({definition.generating_action()}, {depth})'

    def utility(self, problem, values):
        return [-len(val) for val in values]


class TwoStageUtilityFunction:
    def __init__(self, fn_fast, fn_slow, k, large_negative_utility=-10**9):
        self.fn_fast = fn_fast
        self.fn_slow = fn_slow
        self.k = k
        self.large_negative_utility = large_negative_utility

    def group(self, definition, depth):
        return self.fn_fast.group(definition, depth)

    def utility(self, problem, values):
        if len(values) <= self.k:
            return self.fn_slow(problem, values)

        u_fast = self.fn_fast.utility(problem, values)

        top_k = np.argsort(u_fast)[-self.k:]
        top_k_values = [values[i] for i in top_k]
        top_k_indices = {}

        for i, idx in enumerate(top_k):
            top_k_indices[idx] = i

        u_slow = self.fn_slow.utility(problem, top_k_values)
        u_slow.append(self.large_negative_utility)

        return [u_slow[top_k_indices.get(i, -1)] for i in range(len(values))]


def pretrain_utility_function(config: DictConfig):
    with open(config.dataset, 'rb') as f:
        dataset = pickle.load(f)

    u = GRUUtilityFunction(config.utility)

    if config.get('gpu') is not None:
        u = u.to(torch.device(config.gpu))

    i = 0

    def checkpoint():
        nonlocal i
        torch.save(u, f'{i}.pt')
        i += 1

    u.fit(dataset, checkpoint)


def debug_utility_function(config: DictConfig):
    m = torch.load(config.model_path)
    m.eval()

    for p, v, e in [
        ('(= (+ x 0) 3)', '(= (+ x 0) x)', 'high'),
        ('(= (+ x 0) 3)', '(= (+ x 0) (+ 0 x))', 'low'),
        ('(= (+ 0 x) 3)', '(= (+ x 0) (+ 0 x))', 'high'),
        ('(= (+ 0 x) 3)', '(+ x x)', 'low'),
    ]:
        print(p, v, m.utility(p, [v]), f'(should be {e})')

    if config.breakpoint:
        breakpoint()
        print('Model is in `m`')


@hydra.main(version_base="1.2", config_path="config", config_name="utility")
def main(cfg: DictConfig):
    if cfg.task == 'pretrain':
        setup_wandb(cfg)
        pretrain_utility_function(cfg)
    elif cfg.task == 'debug':
        debug_utility_function(cfg)
    else:
        raise ValueError(f'Unknown command {cfg.task}')


if __name__ == '__main__':
    main()
