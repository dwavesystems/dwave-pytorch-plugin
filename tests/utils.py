from inspect import signature

import torch


def model_probably_good(
        model: torch.nn.Module, shape_in: tuple[int, ...], shape_out: tuple[int, ...]
) -> bool:
    """Checks whether the model output has expected shape, is probably unconstrained, and the model
    has its configs stored.

    This function generates dummy data with a padded batch dimension on top of the
    input dimension (so ``shape_in`` should exclude a batch dimension). The data is passed through
    the ``model``. Subsequent tests are described in ``_shapes_match``, ``_probably_unconstrained``,
    and ``_has_correct_config``.

    Args:
        model (torch.nn.Module): The module to be tested.
        shape_in (tuple[int, ...]): Input data shape excluding the batch dimension.
        shape_out (tuple[int, ...]): Output data shape excluding the batch dimension.

    Returns:
        bool: Indicator for whether the model meets the three conditions above.
    """
    bs = 100
    x = torch.randn((bs, ) + shape_in)
    y = model(x)
    padded_out = (bs,)+shape_out
    return (_shapes_match(y, padded_out)
            and _probably_unconstrained(y)
            and _has_correct_config(model))


def _has_correct_config(model: torch.nn.Module) -> bool:
    """Checks whether the model has its initialization arguments stored in a `config` field.

    Args:
        model (torch.nn.Module): The module to be tested.

    Returns:
        bool: Indicator for whether the model has its initialization arguments stored.
    """
    if not hasattr(model, "config"):
        return False
    sig = signature(model.__init__)
    return set(model.config.keys()) == set(sig.parameters.keys()) | {"module_name"}


def _shapes_match(x: torch.Tensor, y: tuple[int, ...]) -> bool:
    """Checks whether `x.shape` is equal to `y`.

    Args:
        x (torch.Tensor): A tensor.
        y (tuple[int, ...]): The expected shape.

    Returns:
        bool: Indicator for whether the shape is as expected.
    """
    return tuple(x.shape) == y


def _are_all_spins(x: torch.Tensor) -> bool:
    """Checks all entries of `x` are one in absolute value.

    Args:
        x (torch.Tensor): A tensor.

    Returns:
        bool: indicator for whether all entries of `x` are in ``{-1, 1}``.
    """
    return (x.float().abs() == 1).all()


def _has_mixed_signs(x: torch.Tensor) -> bool:
    """Checks whether `x` has both positive and negative values.

    Args:
        x (torch.Tensor): A tensor to be cast to type float.

    Returns:
        bool: Indicator for whether `x` consists of both positive and negative values.
    """
    return (((x > 0).float().sum() > 0)
            and
            ((x < 0).float().sum() > 0))


def _has_zeros(x: torch.Tensor) -> bool:
    """Checks whether `x` has exact zeros.

    Args:
        x (torch.Tensor): A tensor.

    Returns:
        bool: Indicator for whether `x` has any zero-valued entries.
    """
    return (x == 0).float().any()


def _bounded_in_plus_minus_one(x: torch.Tensor):
    """Checks whether all entries of `x` are in ``[-1, 1]``.

    Args:
        x (torch.Tensor): A tensor.

    Returns:
        bool: Indicator for whether all values of `x` are in ``[-1, 1]``.
    """
    return (x.abs() <= 1).all()


def _probably_unconstrained(x: torch.Tensor):
    """Checks whether `x` has any activation-like constraints.
    Checks `x` has no exact zeros, not bounded in ``[-1, 1]``, and has both positive and
    negative-valued entries.

    Args:
        x (torch.Tensor): A tensor.

    Returns:
        bool: Indicator for whether `x` passes the "constraints".
    """
    return not _has_zeros(x) and not _bounded_in_plus_minus_one(x) and _has_mixed_signs(x)
