import numpy as np

from .config import rng

class OptFunc:
    def __init__(self, bounds=(-10, 10), measurement_noise=0.1, shift=0.0):
        self.measurement_noise = measurement_noise
        self.shift = shift
        self.lb, self.ub = bounds
        
    def __func__(self, x):
        return NotImplementedError 
        
    def __call__(self, x):
        return self.__func__(x) + rng.normal(scale=self.measurement_noise, size=x.shape)
        
    def sample_x(self, n=1):
        return rng.uniform(self.lb, self.ub, n)
    
    def sample(self, n=1):
        x = self.sample_x(n)
        return x, self.__call__(x)
    
    def sweep_domain(self, n=100):
        x = np.linspace(self.lb, self.ub, n)
        y = self.__func__(x)
        return x, y
    
class SISOLinear(OptFunc):
    """Simple SISO linear function y=mx+b"""
    def __init__(self, m=None, b=None, bounds=(-10, 10), measurement_noise=0.1, shift=0.0):
        super().__init__(bounds=bounds, measurement_noise=measurement_noise, shift=shift)
        self.m = m if m is not None else rng.uniform(-3, 3)
        self.b = b if b is not None else rng.uniform(-5, 5)
    def __func__(self, x):
        return self.m * (x + self.shift) + self.b 
        
class GramacyLee2012(OptFunc):
    """Gramacy, R. B., & Lee, H. K. (2012). Cases for the nugget in modeling computer experiments. Statistics and Computing, 22(3), 713-722."""
    def __init__(self, measurement_noise=0.1, shift=0.0):
        super().__init__(bounds=(0.5, 2.5), measurement_noise=measurement_noise, shift=shift)
    
    def __func__(self, x):
        return np.sin(10*np.pi*x)/(2*x) + (x-1)**4 
    
class Gramacy2016(OptFunc):
    """https://arxiv.org/abs/1403.4890"""
    def __func__(self, x, y):
        """s.t. NL constraints"""
        return -x + -y