import unittest

from dimod import SPIN, SampleSet
from torch import Tensor

from dwave.plugins.torch.utils import sampleset_to_tensor


class TestUtils(unittest.TestCase):
    def test_sample_to_tensor(self):
        bogus_energy = [999] * 3
        spins_in = [[1, -1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        ss = SampleSet.from_samples((spins_in, list("dbca")), SPIN, bogus_energy)
        spins = sampleset_to_tensor(list("cabd"), ss)
        self.assertTupleEqual((3, 4), tuple(spins.shape))
        self.assertIsInstance(spins, Tensor)
        # Test variable ordering is respected
        self.assertListEqual(
            spins.tolist(), [[1, 1, -1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        )
