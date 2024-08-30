import unittest
import torch
from recipe.models import CNN


class CNNTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.cnn = CNN()

    def test_run_cnn(self):
        batch_size = 4
        x = torch.randn(batch_size, 3, 32, 32)
        y = self.cnn(x)

        self.assertIsNotNone(y)
        self.assertEqual(y.shape, (batch_size, 10))
