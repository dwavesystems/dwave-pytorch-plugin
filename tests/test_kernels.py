import unittest

import torch
from parameterized import parameterized

from dwave.plugins.torch.nn.modules.kernels import Kernel, GaussianKernel


class TestKernel(unittest.TestCase):
    def test_forward(self):
        class One(Kernel):
            def _kernel(self, x, y):
                return 1
        one = One()
        x = torch.rand((5, 3))
        y = torch.randn((9, 3))
        self.assertEqual(1, one(x, y))

    @parameterized.expand([(1, 2), (2, 1)])
    def test_sample_size(self, nx, ny):
        class One(Kernel):
            def _kernel(self, x, y):
                return 1
        one = One()
        x = torch.rand((nx, 5))
        y = torch.randn((ny, 5))
        self.assertRaisesRegex(ValueError, "must be at least two", one, x, y)

    def test_shape_mismatch(self):
        class One(Kernel):
            def _kernel(self, x, y):
                return 1
        one = One()
        x = torch.rand((5, 4))
        y = torch.randn((9, 3))
        self.assertRaisesRegex(ValueError, "Input dimensions must match", one, x, y)

class TestGaussianKernel(unittest.TestCase):

    def test_has_config(self):
        rbf = GaussianKernel(5, 2.1, 0.1)
        self.assertDictEqual(dict(rbf.config), dict(module_name="GaussianKernel",
                             n_kernels=5, factor=2.1, bandwidth=0.1))

    @parameterized.expand([
        (torch.randn((5, 12)), torch.rand((7, 12))),
        (torch.randn((5, 12, 34)), torch.rand((7, 12, 34))),
    ])
    def test_shape(self, x, y):
        rbf = GaussianKernel(2, 2.1, 0.1)
        k = rbf(x, y)
        self.assertEqual(tuple(k.shape), (x.shape[0], y.shape[0]))

    def test_get_bandwidth_default(self):
        rbf = GaussianKernel(2, 2.1, 0.1)
        d = torch.tensor(123)
        self.assertEqual(0.1, rbf._get_bandwidth(d))

    def test_get_bandwidth(self):
        rbf = GaussianKernel(2, 2.1, None)
        d = torch.tensor([[0.0, 3.4,], [3.4, 0.0]])
        self.assertEqual(3.4, rbf._get_bandwidth(d))

    def test_get_bandwidth_no_grad(self):
        rbf = GaussianKernel(2, 2.1, None)
        d = torch.tensor([[0.0, 3.4,], [3.4, 0.0]], requires_grad=True)
        self.assertEqual(3.4, rbf._get_bandwidth(d))
        self.assertIsNone(rbf._get_bandwidth(d).grad)

    def test_single_factors(self):
        rbf = GaussianKernel(1, 2.1, None)
        self.assertListEqual(rbf.factors.tolist(), [1.0])

    def test_two_factors(self):
        rbf = GaussianKernel(2, 2.1, None)
        torch.testing.assert_close(torch.tensor([2.1**-1, 1]), rbf.factors)

    def test_three_factors(self):
        rbf = GaussianKernel(3, 2.1, None)
        torch.testing.assert_close(torch.tensor([2.1**-1, 1, 2.1]), rbf.factors)

    def test_kernel(self):
        x = torch.tensor([[1.0, 1.0],
                          [2.0, 3.0]], requires_grad=True)
        y = torch.tensor([[0.0, 1.0],
                          [-3.0, 5.0],
                          [1.2, 9.0]], requires_grad=True)
        dist = torch.cdist(x, y)

        with self.subTest("Adaptive bandwidth"):
            rbf = GaussianKernel(1, 2.1, None)
            bandwidths = rbf._get_bandwidth(dist) * rbf.factors
            manual = torch.exp(-dist/bandwidths)
            torch.testing.assert_close(manual, rbf(x, y))

        with self.subTest("Simple bandwidth"):
            rbf = GaussianKernel(1, 2.1, 12.34)
            bandwidths = 12.34 * rbf.factors
            manual = torch.exp(-dist/bandwidths)
            torch.testing.assert_close(manual, rbf(x, y))

        with self.subTest("Multiple kernels"):
            rbf = GaussianKernel(3, 2.1, 123)
            bandwidths = rbf._get_bandwidth(dist) * rbf.factors
            manual = torch.exp(-dist/bandwidths.reshape(-1, 1, 1)).sum(0)
            torch.testing.assert_close(manual, rbf(x, y))


if __name__ == "__main__":
    unittest.main()
