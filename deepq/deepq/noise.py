
class GaussianWhiteNoise:
    def __init__(self, rand, scale):  
        self._rand  = rand
        self.scale = scale

    def __call__(self):
        return rand.normal() * self._scale


class OrnsteinUhlenbeckNoise:
    def __init__(self, rand, mu, theta, scale):
        self._rand = rand
        self.mu = mu
        self.theta = theta
        self._state = theta
        self.scale = scale

    def __call__(self):
        inc = self._rand.normal() * self.scale
        self._state += self.theta * (self.mu - self._state) + inc
        return self._state