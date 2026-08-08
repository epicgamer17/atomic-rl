import pytest
import math
from atomic_rl.schedules import (
    get_linear_schedule,
    get_exponential_schedule,
    get_ape_x_epsilon,
)

pytestmark = pytest.mark.unit


def test_get_linear_schedule():
    start, end = 1.0, 0.1
    decay_steps = 100

    # Start
    assert math.isclose(get_linear_schedule(0, start, end, decay_steps), 1.0)
    # Middle (step 50)
    # 1.0 + 0.5 * (0.1 - 1.0) = 1.0 - 0.45 = 0.55
    assert math.isclose(get_linear_schedule(50, start, end, decay_steps), 0.55)
    # End
    assert math.isclose(get_linear_schedule(100, start, end, decay_steps), 0.1)
    # Capped
    assert math.isclose(get_linear_schedule(150, start, end, decay_steps), 0.1)


def test_get_exponential_schedule():
    start, end = 1.0, 0.1
    decay_rate = 100.0

    # step 0: 0.1 + (1.0 - 0.1) * exp(0) = 0.1 + 0.9 = 1.0
    assert math.isclose(get_exponential_schedule(0, start, end, decay_rate), 1.0)
    # step 100: 0.1 + 0.9 * exp(-1) approx 0.1 + 0.9 * 0.367879 = 0.1 + 0.33109 = 0.43109
    expected = end + (start - end) * math.exp(-1.0)
    assert math.isclose(get_exponential_schedule(100, start, end, decay_rate), expected)


def test_get_linear_schedule_beta():
    # Testing beta-like use case (annealing up)
    start, end = 0.4, 1.0
    steps = 100
    assert math.isclose(get_linear_schedule(0, start, end, steps), 0.4)
    assert math.isclose(get_linear_schedule(50, start, end, steps), 0.7)
    assert math.isclose(get_linear_schedule(100, start, end, steps), 1.0)
    assert math.isclose(get_linear_schedule(150, start, end, steps), 1.0)


def test_ape_x_epsilon():
    """Test Ape-X fixed epsilon calculation."""
    # If num_actors <= 1, return base_eps
    assert get_ape_x_epsilon(0, 1, base_eps=0.4) == 0.4

    # Check extremes for multiple actors
    # actor 0 should have base_eps ^ (1 + 0) = base_eps
    assert math.isclose(get_ape_x_epsilon(0, 5, base_eps=0.4), 0.4)
    # actor last should have base_eps ^ (1 + alpha)
    expected_last = 0.4 ** (1 + 7.0)
    assert math.isclose(get_ape_x_epsilon(4, 5, base_eps=0.4, alpha=7.0), expected_last)


def test_linear_schedule():
    """Test linear schedule decay."""
    # start 1.0, end 0.1, decay_steps 10
    assert math.isclose(get_linear_schedule(0, 1.0, 0.1, 10), 1.0)
    assert math.isclose(
        get_linear_schedule(5, 1.0, 0.1, 10), 0.55
    )  # 1.0 + 0.5 * (-0.9)
    assert math.isclose(get_linear_schedule(10, 1.0, 0.1, 10), 0.1)
    assert math.isclose(
        get_linear_schedule(20, 1.0, 0.1, 10), 0.1
    )  # Capped at 1.0 fraction


def test_exponential_schedule():
    """Test exponential schedule decay."""
    # start 1.0, end 0.1, decay_rate 10
    # val = end + (start - end) * exp(-step/rate)
    assert math.isclose(get_exponential_schedule(0, 1.0, 0.1, 10), 1.0)
    expected_middle = 0.1 + 0.9 * math.exp(-5 / 10)
    assert math.isclose(get_exponential_schedule(5, 1.0, 0.1, 10), expected_middle)
