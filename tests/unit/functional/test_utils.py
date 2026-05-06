import pytest
import torch
from functional.utils import exponential_moving_average

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
