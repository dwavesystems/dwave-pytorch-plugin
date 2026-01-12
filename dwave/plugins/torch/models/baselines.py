# Copyright 2025 D-Wave
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from math import pi as PI

import numpy as np
import numpy.linalg as la
import torch
from torch import nn

from dwave.plugins.torch.utils import bit2spin_soft, spin2bit_soft


class SphericalBinary(nn.Module):

    @staticmethod
    def nearestPD_np(A):
        # https://github.com/ok1zjf/LBAE/blob/master/sampler.py#L41C2-L41C3
        def chol(B):
            try:
                L = la.cholesky(B)
                return torch.tensor(L)
            except la.LinAlgError:
                return None
        B = (A + A.T) / 2
        _, s, V = la.svd(B)

        H = np.dot(V.T, np.dot(np.diag(s), V))
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2

        L = chol(A3)
        if L is not None:
            return torch.tensor(A3.tolist()), torch.tensor(L.tolist())

        # Still H is not PSD, fixing it...
        spacing = np.spacing(la.norm(A))
        # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
        # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
        # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
        # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
        # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
        # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
        # `spacing` will, for Gaussian random matrixes of small dimension, be on
        # othe order of 1e-16. In practice, both ways converge, as the unit test
        # below suggests.
        I = np.eye(A.shape[0])
        k = 1
        while True:
            L = chol(A3)
            if L is not None:
                break

            # Increasing eigen values...
            mineig = np.min(np.real(la.eigvals(A3)))
            A3 += I * (-mineig * k**2 + spacing)
            k += 1

        return torch.tensor(A3.tolist()), torch.tensor(L.tolist())

    @staticmethod
    def nearestPD_torch(A):
        # https://github.com/ok1zjf/LBAE/blob/master/sampler.py#L41C2-L41C3
        la = torch.linalg

        def chol(B):
            try:
                L = la.cholesky(B)
                return torch.tensor(L)
            except la.LinAlgError:
                return None
        B = (A + A.T) / 2
        _, s, V = la.svd(B)

        H = V.T@(s.diag() @ V)
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2

        L = chol(A3)
        if L is not None:
            return A3, L

        # Still H is not PSD, fixing it...
        spacing = 1e-30
        # spacing = np.spacing(la.norm(A)).item()
        # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
        # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
        # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
        # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
        # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
        # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
        # `spacing` will, for Gaussian random matrixes of small dimension, be on
        # othe order of 1e-16. In practice, both ways converge, as the unit test
        # below suggests.
        I = torch.eye(A.shape[0], device=A.device)
        k = 1
        while True:
            L = chol(A3)
            if L is not None:
                break

            # Increasing eigen values...
            mineig = torch.min(torch.real(la.eigvals(A3)))
            A3 += I * (-mineig * k**2 + spacing)
            k += 1

        return A3, L

    def __init__(self, d: int):
        # Based on latent bernoulli autoencoder paper https://proceedings.mlr.press/v119/fajtl20a.html
        super().__init__()
        self.d = d
        self.V = None

    def fit(self, x):
        if x.abs().min() != x.abs().max() != 1:
            raise RuntimeError("`x` must be spin-valued.")
        bs, d = x.shape
        first_moment = x.mean(0)
        second_moment = torch.einsum("bi, bj -> ij", x, x) / bs
        moments_matrix = torch.eye(d + 1, device=x.device)
        # Upper-left corner is second moment
        moments_matrix.data[:self.d, :self.d] = second_moment
        # Right-most column (up to last second last entry) is first moment
        moments_matrix.data[:-1, -1] = first_moment
        # Bottom-most row (up to last second last entry) is first moment
        moments_matrix.data[-1, :-1] = first_moment
        # Bottom-right entry is one (redundant code; moments matrix is initialized with `torch.eye`)
        moments_matrix.data[-1, -1] = 1
        _, V = self.nearestPD_torch(torch.cos(PI/2*(1 - moments_matrix)))
        self.V = torch.nn.Parameter(V.to(x.device), requires_grad=False)

    def sample(self, n):
        if self.V is None:
            raise RuntimeError("Model needs to be fitted before sampling.")
        r = torch.randn((self.d+1, n), device=self.V.device)
        spins_augmented = (self.V @ r).sign()
        spins_augmented[spins_augmented == 0] = 1
        spins = (spins_augmented[:-1]
                 * spins_augmented[-1].unsqueeze(0)).mT
        return spins


class GaussianBinary(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.rand(d))
        self.cov = torch.nn.Parameter(torch.randn((d, d)))
        self.d = d
        self.epsilon = torch.nn.Parameter(1e-3*torch.eye(d))

    def fit(self, x):
        if x.abs().min() != x.abs().max() != 1:
            raise RuntimeError("`x` must be spin-valued.")
        self.mu.data[:] = x.mean(0)
        self.cov.data[:] = x.mT.cov()

    def sample(self, n):
        c = torch.linalg.cholesky(self.cov + self.epsilon)
        z = torch.randn((n, self.d), device=self.mu.device)
        x = z@c.mT + self.mu
        return bit2spin_soft((x >= 0).float())


class IndependentBernoulli(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.p = torch.nn.Parameter(torch.rand(d))
        self.d = d

    def fit(self, x):
        if x.abs().min() != x.abs().max() != 1:
            raise RuntimeError("`x` must be spin-valued.")
        self.p.data[:] = spin2bit_soft(x.mean(0))

    def sample(self, n):
        if self.p is None:
            raise RuntimeError("Model parameters have not been defined.")
        pp = self.p.unsqueeze(0).repeat(n, 1)
        return bit2spin_soft(pp.bernoulli())
