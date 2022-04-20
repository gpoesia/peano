#!/usr/bin/env python3

import logging
import sys

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from agent import LMPolicyLearning
from policy import DecisionTransformer
from environment import SingleDomainEnvironment


def setup_wandb(jobCfg: DictConfig):
    if jobCfg.get("wandb_project"):
        wandb.init(project=jobCfg.wandb_project)
    else:
        # Disable wandb (i.e., make log() a no-op).
        wandb.log = lambda *args, **kwargs: None

@hydra.main(config_path="config", config_name="test")
def main(cfg: DictConfig):
    setup_wandb(cfg.job)

    env = SingleDomainEnvironment('equations-easy')
    arrows = env.sample_problem(0).actions() + ['eval']

    policy = DecisionTransformer(cfg['policy'], arrows)

    if cfg['policy'].get('gpu') is not None:
        policy.to(torch.device(cfg['policy']['gpu']))

    agent = LMPolicyLearning(policy, cfg['agent'])

    agent.learn_from_environment(env)

if __name__ == '__main__':
    main()
