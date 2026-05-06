import pytest
import torch
import math
from functional.utils import (
    exponential_moving_average,
    get_linear_epsilon,
    get_exponential_epsilon,
    get_linear_beta,
)

pytestmark = pytest.mark.unit


def test_exponential_moving_average():
    old = torch.tensor([1.0, 2.0])
    new = torch.tensor([3.0, 4.0])
    alpha = 0.1
    
    # (1-0.1)*1.0 + 0.1*3.0 = 0.9 + 0.3 = 1.2
    # (1-0.1)*2.0 + 0.1*4.0 = 1.8 + 0.4 = 2.2
    expected = torch.tensor([1.2, 2.2])
    
    res = exponential_moving_average(old, new, alpha)
    torch.testing.assert_close(res, expected)


def test_get_linear_epsilon():
    start, end = 1.0, 0.1
    decay_steps = 100
    
    # Start
    assert math.isclose(get_linear_epsilon(0, start, end, decay_steps), 1.0)
    # Middle (step 50)
    # 1.0 - 0.5 * (1.0 - 0.1) = 1.0 - 0.45 = 0.55
    assert math.isclose(get_linear_epsilon(50, start, end, decay_steps), 0.55)
    # End
    assert math.isclose(get_linear_epsilon(100, start, end, decay_steps), 0.1)
    # Capped
    assert math.isclose(get_linear_epsilon(150, start, end, decay_steps), 0.1)


def test_get_exponential_epsilon():
    start, end = 1.0, 0.1
    decay_rate = 100.0
    
    # step 0: 0.1 + (1.0 - 0.1) * exp(0) = 0.1 + 0.9 = 1.0
    assert math.isclose(get_exponential_epsilon(0, start, end, decay_rate), 1.0)
    # step 100: 0.1 + 0.9 * exp(-1) approx 0.1 + 0.9 * 0.367879 = 0.1 + 0.33109 = 0.43109
    expected = end + (start - end) * math.exp(-1.0)
    assert math.isclose(get_exponential_epsilon(100, start, end, decay_rate), expected)


def test_get_linear_beta():
    start, end = 0.4, 1.0
    steps = 100
    assert math.isclose(get_linear_beta(0, start, end, steps), 0.4)
    assert math.isclose(get_linear_beta(50, start, end, steps), 0.7)
    assert math.isclose(get_linear_beta(100, start, end, steps), 1.0)
    assert math.isclose(get_linear_beta(150, start, end, steps), 1.0)
