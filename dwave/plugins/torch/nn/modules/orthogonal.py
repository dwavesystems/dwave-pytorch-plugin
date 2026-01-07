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

from collections import deque

import torch
import torch.nn as nn
from einops import einsum

from dwave.plugins.torch.nn.modules.utils import store_config

__all__ = ["GivensRotationLayer"]


def _get_blocks_edges(n: int) -> list[list[tuple[int, int]]]:
    """Uses the circle method for Round Robin pairing to create blocks of edges for parallel Givens
    rotations.

    A block is a list of pairs of indices indicating which coordinates to rotate together. Pairs
    in the same block can be rotated in parallel since they commute.

    Args:
        n (int): Dimension of the vector space onto which an orthogonal layer will be built.

    Returns:
        list[list[tuple[int, int]]]: Blocks of edges for parallel Givens rotations.

    .. note::
        If n is odd, a dummy dimension is added to make it even. When using the resulting blocks to
        build an orthogonal transformation, rotations involving the dummy dimension should be
        ignored.
    """
    if n % 2 != 0:
        n += 1  # Add a dummy dimension for odd n
        is_odd = True
    else:
        is_odd = False

    def circle_method(sequence):
        seq_first_half = sequence[: len(sequence) // 2]
        seq_second_half = sequence[len(sequence) // 2 :][::-1]
        return list(zip(seq_first_half, seq_second_half))

    blocks = []
    sequence = list(range(n))
    seqdeque = deque(sequence[1:])
    for _ in range(n - 1):
        pairs = circle_method(sequence)
        if is_odd:
            # Remove pairs involving the dummy dimension:
            pairs = [pair for pair in pairs if n - 1 not in pair]
        blocks.append(pairs)
        seqdeque.rotate(1)
        sequence[1:] = list(seqdeque)
    return blocks


class _RoundRobinGivens(torch.autograd.Function):
    """Implements custom forward and backward passes to implement the parallel algorithms in
    https://arxiv.org/abs/2106.00003
    """

    @staticmethod
    def forward(ctx, angles: torch.Tensor, blocks: torch.Tensor, n: int) -> torch.Tensor:
        """Creates a rotation matrix in n dimensions using parallel Givens transformations by
        blocks.

        Args:
            ctx (context): Stores information for backward propagation.
            angles (torch.Tensor): A ``((n - 1) * n // 2,)`` shaped tensor containing all rotations
                between pairs of dimensions.
            blocks (torch.Tensor): A ``(n - 1, n // 2, 2)`` shaped tensor containing the indices
                that specify rotations between pairs of dimensions. Each of the ``n - 1`` blocks
                contains ``n // 2`` pairs of independent rotations.
            n (int): Dimension of the space.

        Returns:
            torch.Tensor: The nxn rotation matrix.
        """
        # Blocks is of shape (n_blocks, n/2, 2) containing indices for angles
        # Within each block, each Givens rotation is commuting, so we can apply them in parallel
        U = torch.eye(n, device=angles.device, dtype=angles.dtype)
        block_size = n // 2
        idx_block = torch.arange(block_size, device=angles.device)
        for b, block in enumerate(blocks):
            # angles is of shape (n_angles,) containing all angles for contiguous blocks.
            angles_in_block = angles[idx_block + b * blocks.size(1)]  # shape (n/2,)
            c = torch.cos(angles_in_block)
            s = torch.sin(angles_in_block)
            i_idx = block[:, 0]
            j_idx = block[:, 1]
            r_i = c.unsqueeze(0) * U[:, i_idx] + s.unsqueeze(0) * U[:, j_idx]
            r_j = -s.unsqueeze(0) * U[:, i_idx] + c.unsqueeze(0) * U[:, j_idx]
            U[:, i_idx] = r_i
            U[:, j_idx] = r_j
        ctx.save_for_backward(angles, blocks, U)
        return U

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        """Computes the VJP needed for backward propagation.

        Args:
            ctx (context): Contains information for backward propagation.
            grad_output (torch.Tensor): A tensor containing the partial derivatives for the loss
                with respect to the output of the forward pass, i.e., dL/dU.

        Returns:
            tuple[torch.Tensor, None, None]: The gradient of the loss with respect to the input
                angles. No calculation of gradients with respect to blocks or n is needed (cf.
                forward method), so None is returned for these.
        """
        angles, blocks, Ufwd_saved = ctx.saved_tensors
        Ufwd = Ufwd_saved.clone()
        M = grad_output.t()  # dL/dU, i.e., grad_output is of shape (n, n)
        n = M.size(1)
        block_size = n // 2
        A = torch.zeros((block_size, n), device=angles.device, dtype=angles.dtype)
        grad_theta = torch.zeros_like(angles, dtype=angles.dtype)
        idx_block = torch.arange(block_size, device=angles.device)
        for b, block in enumerate(blocks):
            i_idx = block[:, 0]
            j_idx = block[:, 1]
            angles_in_block = angles[idx_block + b * block_size]  # shape (n/2,)
            c = torch.cos(angles_in_block)
            s = torch.sin(angles_in_block)
            r_i = c.unsqueeze(1) * Ufwd[i_idx] + s.unsqueeze(1) * Ufwd[j_idx]
            r_j = -s.unsqueeze(1) * Ufwd[i_idx] + c.unsqueeze(1) * Ufwd[j_idx]
            Ufwd[i_idx] = r_i
            Ufwd[j_idx] = r_j
            r_i = c.unsqueeze(0) * M[:, i_idx] + s.unsqueeze(0) * M[:, j_idx]
            r_j = -s.unsqueeze(0) * M[:, i_idx] + c.unsqueeze(0) * M[:, j_idx]
            M[:, i_idx] = r_i
            M[:, j_idx] = r_j
            A[:] = M[:, j_idx].T * Ufwd[i_idx] - M[:, i_idx].T * Ufwd[j_idx]
            grad_theta[idx_block + b * block_size] = A.sum(dim=1)
        return grad_theta, None, None


class GivensRotationLayer(nn.Module):
    """An orthogonal layer implementing a rotation using a sequence of Givens rotations arranged in
    a round-robin fashion.

    Angles are arranged into blocks, where each block references rotations that can be applied in
    parallel because these rotations commute.

    Args:
        n (int): Dimension of the input and output space. Must be at least 2.
        bias (bool): If True, adds a learnable bias to the output. Default: True.
    """

    @store_config
    def __init__(self, n: int, bias: bool = True):
        super().__init__()
        if not isinstance(n, int) or n <= 1:
            raise ValueError(f"n must be an integer greater than 1, {n} was passed")
        if not isinstance(bias, bool):
            raise ValueError(f"bias must be a boolean, {bias} was passed")
        self.n = n
        self.n_angles = n * (n - 1) // 2
        self.angles = nn.Parameter(torch.randn(self.n_angles))
        blocks_edges = _get_blocks_edges(n)
        self.register_buffer(
            "blocks",
            torch.tensor(blocks_edges, dtype=torch.long),
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(n))
        else:
            self.register_parameter("bias", None)

    def _create_rotation_matrix(self) -> torch.Tensor:
        """Computes the Givens rotation matrix."""
        return _RoundRobinGivens.apply(self.angles, self.blocks, self.n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the Givens rotation to the input tensor ``x``.

        Args:
            x (torch.Tensor): Input tensor of shape ``(..., n)``.

        Returns:
            torch.Tensor: Rotated tensor of shape ``(..., n)``.
        """
        unitary = self._create_rotation_matrix()
        rotated_x = einsum(x, unitary, "... i, o i -> ... o")
        if self.bias is not None:
            rotated_x += self.bias
        return rotated_x
