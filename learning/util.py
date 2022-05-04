#!/usr/bin/env python3

import torch
import math
import random


def count_parameters(model):
    return sum(math.prod(p.shape) for p in model.parameters())


def format_parameter_count(model):
    n = count_parameters(model)

    if n < 1000:
        return str(n)
    elif n < 10**6:
        return f'{n // 10**3}K'
    elif n < 10**9:
        return f'{n / 10**6:.1f}M'

    return f'{n / 10**6:.1f}B'


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
