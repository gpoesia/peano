#!/usr/bin/env python3

import torch
import math


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
