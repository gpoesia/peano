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
from search import SearcherAgent, SearcherResults, run_search_on_batch, load_search_model
from policy import ContrastivePolicy, RandomPolicy
from tactics import induce_tactics


logger = logging.getLogger(__name__)


def spawn_searcher(rank, iteration, domain, tactics, max_nodes, max_depth,
                   rerank_top_k, model_type, model_path, seeds, device):
    out_path = f'rollouts/it{iteration}/searcher{rank}.pt'

    m = load_search_model(model_type, model_path, device=device)

    if os.path.exists(out_path):
        with open(out_path, 'rb') as f:
            return pickle.load(f)

    algorithm = ('policy-beam-search'
                 if model_type == 'contrastive-policy'
                 else 'best-first-search')

    agent = SearcherAgent(make_domain(domain, tactics),
                          m, max_nodes, max_depth,
                          algorithm=algorithm)

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
        self.max_nodes = config.max_nodes
        self.max_depth = config.max_depth
        self.model_type = config.model.type
        self.algorithm = config.algorithm
        self.train_domains = config.train_domains
        self.eval_domains = config.eval_domains
        self.config = config
        self.searcher_futures = []

    def start(self):
        'Reloads the last checkpoint.'

        if 'run_dir' in self.config:
            os.chdir(self.config.run_dir)

        last_checkpoint = None
        tactics = []

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

            tactics_path_i = os.path.join(os.getcwd(), f'tactics-{iteration}.pkl')
            if os.path.exists(tactics_path_i):
                with open(episodes_path_i, 'rb') as f:
                    tactics = pickle.load(f)

            episodes_iteration = i
        else:
            episodes, episodes_iteration = [], -1

        device = get_device(self.config.get('gpus') and self.config.gpus[-1])

        if last_checkpoint is None:
            if self.config.model.type == 'utility':
                m = GRUUtilityFunction(self.config.model)
            elif self.config.model.type == 'contrastive-policy':
                m = ContrastivePolicy(self.config.model)
        else:
            print('Loading', last_checkpoint)
            m = torch.load(last_checkpoint, map_location=device)

        m = m.to(device)
        return m, iteration, last_checkpoint, episodes, tactics, episodes_iteration

    def get_train_domain(self, it: int):
        return self.train_domains[min(it, len(self.train_domains) - 1)]

    def _get_searcher_device(self, searcher_index: int):
        if 'gpus' in self.config:
            return self.config.gpus[:-1][searcher_index % (len(self.config.gpus) - 1)]
        return 'cpu'

    def learn(self, seeds, eval_seeds):
        m, iteration, last_checkpoint, episodes, tactics, ep_it = self.start()

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
                                domain=self.get_train_domain(it),
                                tactics=tactics,
                                max_nodes=self.max_nodes,
                                max_depth=self.max_depth,
                                rerank_top_k=self.rerank_top_k,
                                model_type=self.model_type,
                                model_path=last_checkpoint,
                                seeds=random.choices(seeds, k=self.batch_size),
                                device=self._get_searcher_device(j),
                            )
                        )
                    # Evaluate the current agent.
                    logger.info('Evaluating...')

                    for d in self.eval_domains:
                        if not os.path.exists(f'eval-episodes-{d}-{it}.pkl'):
                            eval_results = run_search_on_batch(
                                make_domain(d),
                                eval_seeds,
                                m,
                                self.algorithm,
                                self.max_nodes,
                                self.max_depth,
                                output_path=f'eval-episodes-{d}-{it}.pkl',
                                debug=False,
                            )

                            wandb.log({f'success_rate_{d}': eval_results.success_rate()})

                    latest_episodes = []

                    # Aggregate episodes from searchers.
                    for i, f in enumerate(self.searcher_futures):
                        logger.info('Waiting for searcher #%d...', i)
                        result_i = f.result()
                        latest_episodes.extend(result_i.episodes)
                        episodes.extend(result_i.episodes)

                    # Induce tactics from new episodes.
                    if self.config.get('induce_tactics'):
                        proposals = induce_tactics(latest_episodes,
                                                   self.config.n_tactics,
                                                   self.config.min_tactic_score,
                                                   tactics)
                        new_tactics = []

                        for p in proposals:
                            new_tactics.append(p.rename(f'tactic{len(tactics) + len(new_tactics):03d}'))

                        logging.info('Incorporating %d new tactics', len(new_tactics))

                        for t in new_tactics:
                            logging.info('%s\n', t)

                        tactics.extend(new_tactics)

                    self.searcher_futures = []

                    with open(f'episodes-{it}.pkl', 'wb') as f:
                        pickle.dump(episodes, f)

                    with open(f'tactics-{it}.pkl', 'wb') as f:
                        pickle.dump(tactics, f)

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

        seeds = range(*cfg.train_interval)
        eval_seeds = range(*cfg.eval_interval)

        trainer = TrainerAgent(cfg.trainer)
        trainer.learn(seeds, eval_seeds)


if __name__ == '__main__':
    main()
