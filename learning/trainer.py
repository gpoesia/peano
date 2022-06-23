#!/usr/bin/env python3


import os
import random
import pickle
import logging
from concurrent.futures import ProcessPoolExecutor

import torch
import hydra
import wandb
from omegaconf import DictConfig

from main import setup_wandb
from domain import make_domain
from utility import GRUUtilityFunction, LengthUtilityFunction, TwoStageUtilityFunction
from util import get_device
from episode import ProofSearchEpisode
from search import SearcherAgent, SearcherResults, run_trained_utility_function


logger = logging.getLogger(__name__)


def spawn_searcher(domain, max_depth, rerank_top_k, model_path, seeds, gpu):
    if model_path is not None:
        device = torch.device(gpu)
        m = torch.load(model_path, map_location=device).to(device)
        m.eval()
        h = TwoStageUtilityFunction(LengthUtilityFunction(), m, k=rerank_top_k)
    else:
        h = LengthUtilityFunction()

    agent = SearcherAgent(make_domain(domain), h, max_depth)

    return agent.run_batch(seeds)


class TrainerAgent:
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.iterations = config.iterations
        self.n_searchers = config.n_searchers
        self.rerank_top_k = config.rerank_top_k
        self.domain = config.domain
        self.max_depth = config.max_depth
        self.config = config
        self.searcher_futures = []

    def learn(self, seeds, eval_seeds):
        m = GRUUtilityFunction(self.config.utility)
        m = m.to(get_device(self.config.gpus[0]))

        last_checkpoint = None
        episodes = []

        with ProcessPoolExecutor() as executor:
            for it in range(self.config.iterations):
                logger.info("### Iteration %d ###", it)
                # Spawn N searchers.
                logger.info('Spawning searchers...')
                for j in range(self.n_searchers):
                    self.searcher_futures.append(
                        executor.submit(
                            spawn_searcher,
                            domain=self.domain,
                            max_depth=self.max_depth,
                            rerank_top_k=self.rerank_top_k,
                            model_path=last_checkpoint,
                            seeds=random.choices(seeds, k=self.batch_size),
                            gpu=self.config.gpus[j],
                        )
                    )
                # Evaluate the current agent.
                logger.info('Evaluating...')

                eval_results = run_trained_utility_function(
                    make_domain(self.domain),
                    eval_seeds,
                    last_checkpoint,
                    get_device(self.config.gpus[-1]),
                    f'eval-episodes-{it}.pkl',
                    debug=False,
                    max_depth=self.max_depth,
                    rerank_top_k=self.rerank_top_k,
                )

                wandb.log({'success_rate': eval_results.success_rate()})

                # Aggregate episodes from searchers.
                episodes = []

                for i, f in enumerate(self.searcher_futures):
                    logger.info('Waiting for searcher #%d...', i)
                    result_i = f.result()
                    episodes.extend(result_i.episodes)

                self.searcher_futures = []

                # Fit model and update checkpoint.
                logger.info('Training model on %d episodes (%d successful)',
                            len(episodes), sum(1 for e in episodes if e.success))
                m.fit(episodes)
                last_checkpoint = os.path.join(os.getcwd(), f'{it}.pt')

                with open(f'episodes-{it}.pkl', 'wb') as f:
                    pickle.dump(episodes, f)

                torch.save(m, last_checkpoint)
                logger.info('Wrote %s', last_checkpoint)


@hydra.main(version_base="1.2", config_path="config", config_name="trainer")
def main(cfg: DictConfig):
    torch.multiprocessing.set_start_method('spawn')

    if cfg.task == 'train':
        setup_wandb(cfg)

        domain = make_domain(cfg.trainer.domain)
        seeds = range(*cfg.train_interval)
        eval_seeds = range(*cfg.eval_interval)

        trainer = TrainerAgent(cfg.trainer)
        trainer.learn(seeds, eval_seeds)


if __name__ == '__main__':
    main()
