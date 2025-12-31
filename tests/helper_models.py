import torch
import torch.nn as nn
from einops import einsum


class NaiveGivensRotationLayer(nn.Module):
    """Naive implementation of a Givens rotation layer.

    Sequentially applies all Givens rotations to implement an orthogonal transformation in an order
    provided by blocks, which are of shape (n_blocks, n/2, 2), and where usually each block contains
    pairs of indices such that no index appears more than once in a block. However, this
    implementation does not rely on that assumption, so that indeces can appear multiple times in a
    block; however, all pairs of indices must appear exactly once in the entire blocks tensor.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): If True, adds a learnable bias to the output. Default: True.

    Note:
        This layer defines an nxn SO(n) rotation matrix, so in_features must be equal to
        out_features.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        assert in_features == out_features, (
            "This layer defines an nxn SO(n) rotation matrix, so in_features must be equal to "
            "out_features."
        )
        self.n = in_features
        self.angles = nn.Parameter(torch.randn(in_features * (in_features - 1) // 2))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def _create_rotation_matrix(self, angles, blocks: torch.Tensor | None = None):
        """Creates the rotation matrix from the Givens angles by applying the Givens rotations in
        order and sequentially, as specified by blocks.

        Args:
            angles (torch.Tensor): Givens rotation angles.
            blocks (torch.Tensor | None, optional): Blocks specifying the order of rotations. If
                None, all possible pairs of dimensions will be shaped into (n-1, n/2, 2) to create
                the blocks. Defaults to None.

        Returns:
            torch.Tensor: Rotation matrix.
        """
        block_size = self.n // 2
        if blocks is None:
            # Create dummy blocks from triu indices:
            triu_indices = torch.triu_indices(self.n, self.n, offset=1)
            blocks = triu_indices.t().view(-1, block_size, 2)
        U = torch.eye(self.n, dtype=angles.dtype)
        for b, block in enumerate(blocks):
            for k in range(block_size):
                i = block[k, 0].item()
                j = block[k, 1].item()
                angle = angles[b * block_size + k]
                c = torch.cos(angle)
                s = torch.sin(angle)
                Ge = torch.eye(self.n, dtype=angles.dtype)
                Ge[i, i] = c
                Ge[j, j] = c
                Ge[i, j] = -s
                Ge[j, i] = s
                # Explicit Givens rotation
                U = U @ Ge
        return U

    def forward(self, x: torch.Tensor, blocks: torch.Tensor) -> torch.Tensor:
        """Applies the Givens rotation to the input tensor ``x``.

        Args:
            x (torch.Tensor): Input tensor of shape (..., n).
            blocks (torch.Tensor): Blocks specifying the order of rotations.

        Returns:
            torch.Tensor: Rotated tensor of shape (..., n).
        """
        W = self._create_rotation_matrix(self.angles, blocks)
        x = einsum(x, W, "... i, o i -> ... o")
        if self.bias is not None:
            x = x + self.bias
        return x
