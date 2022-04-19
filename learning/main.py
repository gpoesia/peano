#!/usr/bin/env python3

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from agent import LMPolicyLearning
from policy import DecisionTransformer
from environment import SingleDomainEnvironment


@hydra.main(config_path="config", config_name="test")
def main(cfg: DictConfig):
    env = SingleDomainEnvironment('equations-easy')
    arrows = env.sample_problem(0).actions() + ['eval']

    policy = DecisionTransformer(cfg['policy'], arrows)

    if cfg['policy'].get('gpu') is not None:
        policy.to(torch.device(cfg['policy']['gpu']))

    agent = LMPolicyLearning(policy, cfg['agent'])

    agent.learn_from_environment(env)

if __name__ == '__main__':
    main()
