import os
import unittest
from recipe.datasets import CIFAR10


RUNNIN_ON_H2 = os.path.exists("/checkpoint")


class TestCIFAR10:
    @unittest.skipUnless(RUNNIN_ON_H2, "Test requires access to H2 checkpoint")
    def test_instantiation(self):
        batch_size = 4
        cifar10 = CIFAR10(batch_size=batch_size)
        assert cifar10.batch_size == 4

    @unittest.skipUnless(RUNNIN_ON_H2, "Test requires access to H2 checkpoint")
    def test_get_trainloader(self):
        batch_size = 4
        cifar10 = CIFAR10(batch_size=batch_size)
        trainloader = cifar10.get_trainloader()
        assert len(trainloader) > 10
