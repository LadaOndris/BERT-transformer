import torch


class SingleGPULossCompute:
    """
    A single GPU loss computation.
    """

    def __init__(self, model, criterion, optimizer=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def __call__(self, x, y):
        loss = self.criterion(x, y)
        loss.backward()
        if self.optimizer is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss.data