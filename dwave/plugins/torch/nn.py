from inspect import signature

import torch
from torch import nn


class Module(nn.Module):

    def __init__(self, obj, kwargs):
        """Base class for all neural network modules that additionally stores its init args in a config.
        """
        super().__init__()
        self._config = {k: kwargs[k] for k in signature(obj.__init__).parameters.keys()}

    @property
    def config(self):
        """The dictionary of arguments used to instantiate the model."""
        return self._config.copy()


class Identity(Module):

    def __init__(self):
        """An identity module.

        This module is useful for handling cases where a neural network module is expected, but no
        effect is desired."""
        super().__init__(self, vars())

    def forward(self, x) -> torch.Tensor:
        """Input

        Args:
            x (torch.Tensor): The input and the output.

        Returns:
            torch.Tensor: The input and the output.
        """
        return x


class SkipLinear(Module):

    def __init__(self, din, dout) -> None:
        """Applies a linear transformation to the incoming data: :math:`y = xA^T`.

        This module is identity when `din == dout`, otherwise, it is a linear transformation, i.e.,
        no bias term.

        Args:
            din (int): Size of each input sample.
            dout (int): Size of each output sample.
        """
        super().__init__(self, vars())
        if din == dout:
            self.linear = Identity()
        else:
            self.linear = nn.Linear(din, dout, bias=False)

    def forward(self, x):
        """Apply a linear transformation to the input variable `x`.

        Args:
            x (torch.Tensor): the input tensor.

        Returns:
            torch.Tensor: the linearly-transformed tensor of `x`.
        """
        return self.linear(x)


class LinearBlock(Module):
    def __init__(self, din, dout, p) -> None:
        """A linear block consisting of normalizations, linear transformations, dropout, relu, and a skip connection.

        The module is composed of (in order):
        1. a first layer norm,
        2. a first linear transformation,
        3. a dropout,
        4. a relu activation,
        5. a second layer norm,
        6. a second linear layer, and, finally,
        7. a skip connection from initial input to output.

        Args:
            din (int): Size of each input sample
            dout (int): Size of each output sample
            p (float): Dropout probability.
        """
        super().__init__(self, vars())
        self.skip = SkipLinear(din, dout)
        linear_1 = nn.Linear(din, dout)
        linear_2 = nn.Linear(dout, dout)
        self.block = nn.Sequential(
            nn.LayerNorm(din),
            linear_1,
            nn.Dropout(p),
            nn.ReLU(),
            nn.LayerNorm(dout),
            linear_2,
        )

    def forward(self, x) -> torch.Tensor:
        """Transforms the input `x` with the modules.

        Args:
            x (torch.Tensor): An input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.block(x) + self.skip(x)
