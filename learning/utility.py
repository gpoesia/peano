#!/usr/bin/env python3

import random
import math
import logging
import hydra
import pickle
from dataclasses import dataclass

import torch
import wandb
from torch import nn
from tqdm import tqdm
from omegaconf import DictConfig

from policy import encode_batch
from search import ProofSearchEpisode
from main import setup_wandb


logger = logging.getLogger(__name__)


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
        return self(problems, values)

    def forward(self, problems, objects):
        strs = [f'{p} => {o}' for p, o in zip(problems, objects)]
        embeddings = self.embed_raw(strs)
        return self.utility_readout(embeddings)

    def nce_loss(self, problem, positive, negatives):
        all_examples = [positive] + negatives
        logits = self.utility(problem, all_examples).exp()
        return -(logits[0] / logits.sum()).log()

    def fit(self, dataset: list[ProofSearchEpisode], checkpoint_callback=lambda: None):
        optimizer = torch.optim.Adam(self.parameters())

        for e in range(self.config.n_epochs):
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

                wandb.log({'epoch': e, 'train_loss': loss})
                logger.debug('{"train_loss": %f}', loss)


def pretrain_utility_function(config: DictConfig):
    with open(config.dataset, 'rb') as f:
        dataset = pickle.load(f)

    u = GRUUtilityFunction(config.utility)

    i = 0

    def checkpoint():
        nonlocal i
        torch.save(u, f'{i}.pt')
        i += 1

    u.fit(dataset, checkpoint)


@hydra.main(version_base="1.2", config_path="config", config_name="utility")
def main(cfg: DictConfig):
    setup_wandb(cfg)

    if cfg.task == 'pretrain':
        pretrain_utility_function(cfg)
    else:
        raise ValueError(f'Unknown command {cfg.task}')


if __name__ == '__main__':
    main()
