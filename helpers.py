"""
Helper functions used in the cohort models. 
"""
import numpy as np


def map_exp(x, alpha=1):
    return 1 - np.exp(-alpha * x)

def map_tanh(x, alpha=1):
    return np.tanh(alpha * x)

def map_linear(x, delta_max=10):
    assert delta_max != 0
    return np.min([1, x / delta_max])

def continuous_multinomial(n, pvals, precision=4):
    result = np.zeros(len(pvals), dtype='float64')
    i = 0
    while n > 0:
        int_part, dec_part = divmod(n, 1)
        result += np.random.multinomial(int_part, pvals) / 10**i
        n = np.around(dec_part * 10, precision)
        i += 1
    return result

def rolling_average(arr, steps=10):
    return [np.mean(arr[max(0, i-steps+1):i+1]) for i in range(len(arr))]

def linear_min(x, lb, ub):
    return max(0, min(1, x / (lb - ub) + 1 - lb / (lb - ub)))

def to_sym_int(x):
    assert 0 <= x <= 1
    return 2 * (x - 0.5)

def to_unit_int(x):
    assert -1 <= x <= 1
    return x / 2 + 0.5
    