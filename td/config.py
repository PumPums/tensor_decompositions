import os
from dataclasses import dataclass
from typing import Callable


def _check_env_var(var_name="IS_DECOMPOSED"):
    decompose_mode = int(os.environ.get(var_name, 0))
    assert decompose_mode in [
        0,
        1,
        2,
        3,
    ], f"Wrong env IS_DECOMPOSED={decompose_mode} value"
    return decompose_mode


def _calc_core_ranks(in_channels, out_channels):
    if in_channels == out_channels:
        return [in_channels // 2, out_channels // 2]
    if in_channels > out_channels:
        return [in_channels // 4, out_channels // 2]
    return [in_channels // 2, out_channels // 4]


def _calc_stick_rank(in_channels, out_channels):
    if in_channels == out_channels:
        return out_channels // 2
    return min(in_channels, out_channels)


@dataclass
class TDConfig:
    """Hidden config for Conv2dTD"""

    # get decomposed_mode from env variable "IS_DECOMPOSED"
    # 1, 2, 3 are modes, 0 means is_decomposed=False
    is_decomposed_mode: bool = _check_env_var()
    # get function for core_ranks calculation
    calc_core_ranks: Callable = _calc_core_ranks
    # get function for stick_rank calculation
    calc_stick_rank: Callable = _calc_stick_rank
