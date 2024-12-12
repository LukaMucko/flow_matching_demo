import torch
from torch import Tensor

class ODESampler:
    def __init__(self):
        pass
    
    def sample(self, f: torch.nn.Module, x0: Tensor, t0: float = 0.0, t1: float = 1.0, timesteps=50, method="euler", return_path=True):
        """
        :param method: one of 'euler', 'midpoint', or 'rk4'
        x0: Tensor of shape (batch_size, *shape)
        t0: Tensor of shape ()
        t1: Tensor of shape ()
        timesteps: int
        """
        f.eval()
        t_shape = [1] * len(x0.shape)
        t_shape[0] = x0.shape[0]
        t0 = torch.full(t_shape, t0).to(f.device)
        t1 = torch.full(t_shape, t1).to(f.device)
        with torch.no_grad():
            if method=="euler":
                return self.euler(f, x0, t0, t1, timesteps, return_path)
            if method=="midpoint":
                return self.midpoint(f, x0, t0, t1, timesteps, return_path)
            if method=="rk4":
                return self.rk4(f, x0, t0, t1, timesteps, return_path)
            
    def euler(self, f, x, t0, t1, timesteps, return_path):
        h = (t1 - t0) / timesteps
        t = t0
        path = [x.cpu()]
        for i in range(timesteps):
            x = x + h * f(x, t)
            t += h
            path.append(x.cpu())
        return torch.stack(path).squeeze(1) if return_path else x
    
    def midpoint(self, f, x, t0, t1, timesteps, return_path):
        h = (t1 - t0) / timesteps
        t = t0
        path = [x.cpu()]
        for i in range(timesteps):
            k = f(x, t)
            x_mid = x + h/2 * k
            k_mid = f(x_mid, t + h/2)
            x = x + h * k_mid
            t += h
            path.append(x.cpu())
        return torch.stack(path).squeeze(1) if return_path else x
    
    def rk4(self, f, x, t0, t1, timesteps, return_path):
        h = (t1 - t0) / timesteps
        t = t0
        path = [x.cpu()]
        for i in range(timesteps):
            k1 = f(x, t)
            k2 = f(x + h/2 * k1, t + h/2)
            k3 = f(x + h/2 * k2, t + h/2)
            k4 = f(x + h * k3, t + h)
            x = x + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
            t += h
            path.append(x.cpu())
        return torch.stack(path).squeeze(1) if return_path else x
