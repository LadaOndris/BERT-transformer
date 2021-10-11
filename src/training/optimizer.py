

class NoamOptimizer:

    def __init__(self, d_model, warmup_steps, optimizer, lr_coeff):
        self.optimizer = optimizer
        self._step = 0
        self.warmup_steps = warmup_steps
        self.model_size = d_model
        self._rate = 0
        self._lr_coeff = lr_coeff

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self._lr_coeff * (self.model_size ** (-0.5) *
                                 min(step ** (-0.5), step * self.warmup_steps ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()
