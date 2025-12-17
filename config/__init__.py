"""Config module for GenOmnimatte."""

from config.default_cogvideox import get_config as get_cogvideox_config
from config.default_omnimatte import get_config as get_omnimatte_base_config

__all__ = ["get_cogvideox_config", "get_omnimatte_base_config"]
