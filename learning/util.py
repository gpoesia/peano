#!/usr/bin/env python3

import math
import random
import os
import logging

import torch
import wandb
from omegaconf import DictConfig, OmegaConf, open_dict


PAD = 0
BOS = 1
EOS = 2
POSITIVE = ord(';')
NEGATIVE = ord('$')
EMPTY = '\x03'

def count_parameters(model):
    return sum(math.prod(p.shape) for p in model.parameters())


def format_parameter_count(model):
    n = count_parameters(model)

    if n < 1000:
        return str(n)
    if n < 10**6:
        return f'{n // 10**3}K'
    if n < 10**9:
        return f'{n / 10**6:.1f}M'

    return f'{n / 10**6:.1f}B'


def encode_batch(b: list[str], device: torch.device, bos=True, eos=True) -> torch.LongTensor:
    if not b:
        return torch.tensor([], dtype=torch.long, device=device)

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


def sample_batch(examples: list[str], batch_size: int) -> list[str]:
    'Samples a batch of examples with a bounded total number of characters'
    batch = []
    max_size = 0

    while True:
        example = random.choice(examples)
        max_size = max(max_size, len(example))

        if max_size * (1 + len(batch)) > batch_size:
            break

        batch.append(example)

    return batch

def log(x):
    'Safe version of log'
    return math.log(1e-50 + max(0, x))

def softmax(logits: torch.Tensor, temperature = 1.0):
    s = logits.exp() / temperature
    return s / s.sum()


def pop_max(l: list, key) -> (object, list):
    if not l:
        return None, l

    i_max = max(range(len(l)), key=lambda i: key(l[i]))
    l[-1], l[i_max] = l[i_max], l[-1]
    return l[-1], l[:-1]


def shuffle_state(s: str) -> str:
    'Perform data augmentation for Peano states by shuffling e-classes and e-nodes'

    eclasses = s.split('; ')
    random.shuffle(eclasses)

    for i in range(len(eclasses)):
        enodes, dtype = eclasses[i].split(' : ')
        # Strip '{' and '}'
        enodes = enodes.lstrip('{').rstrip('}').split('=')
        random.shuffle(enodes)

        eclasses[i] = f'{{{"=".join(enodes)}}} : {dtype}'

    return '; '.join(eclasses)


def parse_sexp(s: str, ptr: int = 0) -> (object, int):
    while ptr < len(s) and s[ptr] == ' ':
        ptr += 1

    if s[ptr] == '(':
        # Read list
        ptr += 1 # Consume (
        l = []
        while s[ptr] != ')':
            elem, ptr = parse_sexp(s, ptr)
            l.append(elem)
        ptr += 1 # Consume )
        return l, ptr
    else:
        # Read atom
        before = ptr
        while ptr < len(s) and s[ptr] not in ' ()':
            ptr += 1
        return s[before:ptr], ptr


def randomize_atoms(sexp, criteria, sample, mapping):
    if isinstance(sexp, str):
        if sexp in mapping:
            return mapping[sexp]

        if criteria(sexp):
            v = str(sample())
            mapping[sexp] = v
            return v

        return sexp
    return [randomize_atoms(s, criteria, sample, mapping) for s in sexp]


def format_sexp(sexp):
    if isinstance(sexp, str):
        return sexp
    return '(' + ' '.join(map(format_sexp, sexp)) + ')'


def randomly_mask_atoms(sexp, probability):
    if isinstance(sexp, str):
        if random.random() < probability:
            return '?'
        return sexp

    return list(map(lambda elem: randomly_mask_atoms(elem, probability), sexp))


def randomly_mask_goal_terms(goal: str, probability=0.1) -> str:
    'Perform data augmentation for Peano goals by masking some sub-terms'

    sexp, _ = parse_sexp(goal)
    sexp = randomly_mask_atoms(sexp, probability)
    return format_sexp(sexp)


def get_device(cfg):
    if cfg is None:
        return torch.device('cpu')

    if isinstance(cfg, int):
        return torch.device(cfg)

    if cfg.get('gpu') is not None:
        return torch.device(cfg.gpu)

    return torch.device('cpu')


def choose_from_list(prompt, l, to_str=str):
    print(prompt)
    for i, e in enumerate(l):
        print(f'{i:2d} - ', to_str(e))

    return l[int(input('> '))]


def setup_wandb(cfg: DictConfig):
    if cfg.job.get("wandb_project"):
        with open_dict(cfg.job):
            cfg.job.cwd = os.getcwd()
        wandb.init(
                project=cfg.job.wandb_project,
                config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
        for key in logging.Logger.manager.loggerDict.keys():
            if key.startswith('wandb'):
                logging.getLogger(key).setLevel(logging.WARNING)
    else:
        # Disable wandb (i.e., make log() a no-op).
        wandb.log = lambda *args, **kwargs: None
