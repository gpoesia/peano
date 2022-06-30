#!/usr/bin/env python3


import os
import random
import pickle
import logging
import itertools
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
from search import SearcherAgent, SearcherResults, run_utility_function


logger = logging.getLogger(__name__)


def spawn_searcher(rank, iteration, domain, max_depth, rerank_top_k, model_path, seeds, gpu):
    out_path = f'rollouts/it{iteration}/searcher{rank}.pt'

    if os.path.exists(out_path):
        with open(out_path, 'rb') as f:
            return pickle.load(f)

    if model_path is not None:
        device = torch.device(gpu)
        m = torch.load(model_path, map_location=device).to(device)
        m.eval()
        h = TwoStageUtilityFunction(LengthUtilityFunction(), m, k=rerank_top_k)
    else:
        h = LengthUtilityFunction()

    agent = SearcherAgent(make_domain(domain), h, max_depth)

    episodes = agent.run_batch(seeds)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, 'wb') as f:
        pickle.dump(episodes, f)

    return episodes


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

    def start(self):
        'Reloads the last checkpoint.'

        if 'run_dir' in self.config:
            os.chdir(self.config.run_dir)

        last_checkpoint = None

        for i in itertools.count(0):
            path_i = os.path.join(os.getcwd(), f'{i}.pt')

            if os.path.exists(path_i):
                last_checkpoint = path_i
            else:
                iteration = i
                break

        episodes_path_i = os.path.join(os.getcwd(), f'episodes-{iteration}.pkl')

        if os.path.exists(episodes_path_i):
            with open(episodes_path_i, 'rb') as f:
                episodes = pickle.load(f)
            episodes_iteration = i
        else:
            episodes, episodes_iteration = [], -1

        device = get_device(self.config.gpus[-1])

        if last_checkpoint is None:
            m = GRUUtilityFunction(self.config.utility)
        else:
            print('Loading', last_checkpoint)
            m = torch.load(last_checkpoint, map_location=device)

        m = m.to(device)
        return m, iteration, last_checkpoint, episodes, episodes_iteration

    def learn(self, seeds, eval_seeds):
        m, iteration, last_checkpoint, episodes, ep_it = self.start()

        with ProcessPoolExecutor() as executor:
            for it in range(iteration, self.config.iterations):
                logger.info("### Iteration %d ###", it)

                if ep_it >= it:
                    logger.info('Loaded dumped episodes for this iteration.')
                    logger.info('%d total, %d successful',
                                len(episodes),
                                sum(1 for e in episodes if e.success))
                else:
                    # Spawn N searchers.
                    logger.info('Spawning searchers...')
                    for j in range(self.n_searchers):
                        self.searcher_futures.append(
                            executor.submit(
                                spawn_searcher,
                                iteration=it,
                                rank=j,
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

                    if not os.path.exists(f'eval-episodes-{it}.pkl'):
                        run_utility_function(
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
                    for i, f in enumerate(self.searcher_futures):
                        logger.info('Waiting for searcher #%d...', i)
                        result_i = f.result()
                        episodes.extend(result_i.episodes)

                    self.searcher_futures = []

                    with open(f'episodes-{it}.pkl', 'wb') as f:
                        pickle.dump(episodes, f)

                # Fit model and update checkpoint.
                logger.info('Training model on %d episodes (%d successful)',
                            len(episodes), sum(1 for e in episodes if e.success))
                m.fit(episodes)
                last_checkpoint = os.path.join(os.getcwd(), f'{it}.pt')

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
