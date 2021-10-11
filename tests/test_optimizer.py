from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np

from src.training.optimizer import NoamOptimizer


class TestNoamOptimizer(TestCase):

    def test_rate(self):
        params_list = [[512, 5000, 1], [256, 5000, 1], [128, 5000, 1]]
        optimizers = []
        legends = []
        for params in params_list:
            opt = NoamOptimizer(d_model=params[0], warmup_steps=params[1], optimizer=None, lr_coeff=params[2])
            legends.append(','.join(map(str, params)))
            optimizers.append(opt)
        rates_list = []
        xs = np.arange(1, 30000)
        for optimizer in optimizers:
            rates = [optimizer.rate(i) for i in xs.flat]
            rates_list.append(rates)

        for rates in rates_list:
            plt.plot(xs, rates)
        plt.legend(legends)
        plt.show()
