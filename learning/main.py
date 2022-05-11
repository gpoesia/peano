#!/usr/bin/env python3

import logging
import sys

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from agent import LMPolicyLearning
from policy import DecisionTransformer, DecisionGRU, make_policy
from environment import SingleDomainEnvironment
from domain import EquationsDomain


def setup_wandb(cfg: DictConfig):
    if cfg.job.get("wandb_project"):
        wandb.init(project=cfg.job.wandb_project, config=cfg)
    else:
        # Disable wandb (i.e., make log() a no-op).
        wandb.log = lambda *args, **kwargs: None

@hydra.main(config_path="config", config_name="test")
def main(cfg: DictConfig):
    setup_wandb(cfg)

    domain = EquationsDomain()
    # policy = DecisionTransformer(cfg['policy'])
    policy = make_policy(cfg['policy'])

    if cfg['policy'].get('gpu') is not None:
        policy = policy.to(torch.device(cfg['policy']['gpu']))

    agent = LMPolicyLearning(policy, cfg['agent'])
    agent.learn_domain(domain)

if __name__ == '__main__':
    main()
