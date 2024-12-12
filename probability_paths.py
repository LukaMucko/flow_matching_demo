import torch
import abc
from dataclasses import dataclass

@dataclass
class Sample:
    x0: torch.Tensor
    x1: torch.Tensor
    t: torch.Tensor
    xt: torch.Tensor
    dxt: torch.Tensor


class ProbPath(abc.ABC):
    @abc.abstractmethod
    def sample(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> Sample:
        pass

class LinearPath(ProbPath):
    def __init__(self):
        pass
    
    def sample(self, x0, x1, t):
        t_shape = [1] * len(x0.shape)
        t_shape[0] = x0.shape[0]
        t = t.view(t_shape)
        xt = t * x1 + (1 - t) * x0
        dxt = x1 - x0
        return Sample(x0, x1, t, xt, dxt)


class GaussianPath(ProbPath):
    def __init__(self, const=1e-4):
        self.const = const
    
    def sample(self, x0, x1, t):
        t_shape = [1] * len(x0.shape)
        t_shape[0] = x0.shape[0]
        t = t.view(t_shape)
        xt = t*x1 + (1-(1-self.const)*t)*torch.randn_like(x0)
        dxt = (x1 - (1-self.const)*xt)/(1-(1-self.const)*t)
        return Sample(x0, x1, t, xt, dxt)
    