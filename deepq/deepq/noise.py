import numpy as np

class GaussianWhiteNoise:
    def __init__(self, scale=1.0, rand=None):  
        if rand is None:
            rand = np.random
        self._rand  = rand
        self.scale = scale

    def __call__(self):
        return rand.normal() * self._scale


class OrnsteinUhlenbeckNoise:
    def __init__(self, mu, theta, sigma, rand=None):
        if rand is None:
            rand = np.random
        self._rand = rand
        self.mu = mu
        self.theta = theta
        self._state = theta
        self.sigma = sigma

    def __call__(self):
        inc = self._rand.normal() * self.sigma
        self._state += self.theta * (self.mu - self._state) + inc
        return self._state