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
from __future__ import annotations
import inspect
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from functools import partial

from functools import wraps
from types import MappingProxyType

__all__ = ["store_config"]


def store_config(fn: Callable) -> partial:
    """A decorator that tracks and stores arguments of methods (excluding ``self``).

    .. note:: 
        If an argument of the function also has a config attribute, then the argument's entry in
        the dictionary will be replaced by the argument's config. For example, an argument ``foo`` has
        a ``config`` attribute, i.e., ``foo.config`` exists, then ``self.config`` will contain the entry
        ``{"foo": foo.config}``. This is motivated by the convenience of storing configs of nested
        modules.

    Args:
        fn (Callable[object, ...]): A method whose arguments will be stored in ``self.config``.

    Returns:
        partial: Wrapper function that stores argument of method.
    """
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        """Store ``args``, ``kwargs``, and ``{"module_name": self.__class__.__name__}`` as a dictionary in ``self.config``.
        """
        sig = inspect.signature(fn)
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()

        config = {k: v for k, v in bound.arguments.items() if v != self}
        config['module_name'] = self.__class__.__name__
        for k, v in config.items():
            if hasattr(v, "config"):
                config[k] = v.config
        self.config = MappingProxyType(config)

        return fn(self, *args, **kwargs)
    return wrapper
