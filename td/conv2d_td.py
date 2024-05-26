from typing import Optional

import numpy as np
import torch

from .config import TDConfig
from .core import partial_tucker, tucker_stick


class Conv2dTD(torch.nn.Conv2d):
    """
    nn.Conv2d class with tensor decompositions techniques
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
        core_ranks=None,  # new arg
        stick_rank=None,  # new arg
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        self.td_cfg = TDConfig()
        self.need_decompose = self._need_decompose()
        self.is_decomposed = self.td_cfg.is_decomposed_mode > 0 and self.need_decompose
        self.decomposed_conv2d = None

        self.core_ranks = core_ranks
        self.stick_rank = stick_rank

        if self.is_decomposed and self.need_decompose:
            if self.core_ranks is None:
                self.core_ranks = self.td_cfg.calc_core_ranks(
                    self.in_channels, self.out_channels
                )
            if self.stick_rank is None:
                self.stick_rank = self.td_cfg.calc_stick_rank(
                    self.in_channels, self.out_channels
                )

            self._init_decomposed_convs(
                core_ranks=self.core_ranks,
                stick_rank=self.stick_rank,
                mode=self.td_cfg.is_decomposed_mode,
                need_bias=True if self.bias is not None else None,
            )

    def _init_decomposed_convs(self, core_ranks, stick_rank, mode, need_bias):
        device = self.weight.device
        new_convs = self._create_new_convs(core_ranks, stick_rank, mode, need_bias)
        self.decomposed_conv2d = torch.nn.Sequential(
            *[new_convs[i] for i in sorted(new_convs)]
        ).to(device)
        self.weight = None
        self.bias = None
        self.is_decomposed = True

    def forward(self, *args, **kwargs):
        return (
            self.decomposed_conv2d(*args, **kwargs)
            if self.is_decomposed
            else super().forward(*args, **kwargs)
        )

    def decompose(
        self,
        core_ranks: Optional[tuple] = None,
        stick_rank: Optional[int] = None,
        mode: int = 2,
        n_iters: int = 100,
        tol: float = 5e-4,
    ):
        """
        Check if this layer is decomposed
        """
        if not self.is_decomposed and self.need_decompose:
            core_ranks = (
                self.core_ranks
                if self.core_ranks is not None
                else (
                    core_ranks
                    if core_ranks is not None
                    else self.td_cfg.calc_core_ranks(
                        self.in_channels, self.out_channels
                    )
                )
            )
            stick_rank = (
                self.stick_rank
                if self.stick_rank is not None
                else (
                    stick_rank
                    if stick_rank is not None
                    else self.td_cfg.calc_stick_rank(
                        self.in_channels, self.out_channels
                    )
                )
            )
            self._decompose(core_ranks, stick_rank, mode, n_iters, tol)

    def _need_decompose(self):
        check_min_ker = min(self.kernel_size) > 1
        check_not_2x2 = not (
            self.kernel_size[0] == self.kernel_size[1] and self.kernel_size[0] == 2
        )
        check_layers = min(self.in_channels, self.out_channels) > 16
        dev_16 = self.in_channels % 16 == 0 and self.out_channels % 16 == 0
        return all([check_min_ker, check_not_2x2, check_layers, dev_16])

    def _decompose(self, core_ranks, stick_rank, mode=2, n_iters=100, tol=5e-4):
        """
        Decompose weights of Conv2d and built new layers

        Args:
            core_ranks: (in_channels, out_channels) in the intermidiate tesor of
                the partial_tucker decomposition (mode=1,3)
            stick_rank: shape of the intermidiate dimetion between tensors of
                the tucker_stick decomposition (mode=2,3)
            mode: choosen mode (1, 2 or 3)
            n_iters: amount of iterations for core approximation
            tol: err for early stop of iterations

        mode=1 (partial_tucker):
            represent conv2d layer as nn.Sequential(
                conv2d(in_channels=in_channels, out_channels=core_ranks[0], kernel_size=(1, 1))
                conv2d(in_channels=core_ranks[0], out_channels=core_ranks[1], kernel_size=(ker_h, ker_w))
                conv2d(in_channels=core_ranks[1], out_channels=out_channels, kernel_size=(1, 1))
            )

        mode=2 (tucker_stick): --- (recommended) ---
            represent conv2d layer as nn.Sequential(
                conv2d(in_channels=in_channels, out_channels=stick_rank, kernel_size=(ker_h, 1))
                conv2d(in_channels=stick_rank, out_channels=out_channels, kernel_size=(1, ker_w))
            )

        mode=3 (partial_tucker + tucker_stick):
            represent conv2d layer as nn.Sequential(
                conv2d(in_channels=in_channels, out_channels=core_ranks[0], kernel_size=(1, 1))
                conv2d(in_channels=core_ranks[0], out_channels=stick_rank, kernel_size=(ker_h, 1))
                conv2d(in_channels=stick_rank, out_channels=core_ranks[1], kernel_size=(1, ker_w))
                conv2d(in_channels=core_ranks[1], out_channels=out_channels, kernel_size=(1, 1))
            )
        """
        torch_dtype, is_grad, device = (
            self.weight.dtype,
            self.weight.requires_grad,
            self.weight.device,
        )

        weights = self.weight.data.detach().cpu().numpy().astype(np.float64)
        bias = self.bias.data if self.bias is not None else None
        assert mode in [1, 2, 3], f"mode={mode} is not exist"
        assert len(core_ranks) == 2, "len(core_ranks) != 2"

        if stick_rank is None:
            stick_rank = min(core_ranks)

        new_convs = self._create_new_convs(
            core_ranks=core_ranks,
            stick_rank=stick_rank,
            mode=mode,
            need_bias=bias is not None,
        )

        decomposed_weights = self._decompose_weights(
            weights=weights,
            core_ranks=core_ranks,
            stick_rank=stick_rank,
            mode=mode,
            n_iters=n_iters,
            tol=tol,
        )
        for name, conv in new_convs.items():
            curr_weights = torch.from_numpy(decomposed_weights[name]).to(torch_dtype)
            if is_grad:
                curr_weights.requires_grad_(True)
            conv.weight.data = torch.from_numpy(decomposed_weights[name]).to(
                torch_dtype
            )

        if bias is not None:
            if mode == 2:
                new_convs[3].bias.data = bias
            else:
                new_convs[4].bias.data = bias

        self.decomposed_conv2d = torch.nn.Sequential(
            *[new_convs[i] for i in sorted(new_convs)]
        ).to(device)
        self.weight = None
        self.bias = None
        self.is_decomposed = True

    def _decompose_weights(
        self, weights, core_ranks, stick_rank, mode=2, n_iters=100, tol=5e-4
    ):
        """
        Decompose weights of Conv2d for the corresponding mode of decomposition
        """
        decomposed_weights = {}
        # partial_tucker
        if mode in [1, 3]:
            assert core_ranks is not None, "core_ranks (for partial_tucker) is None"
            tucker_core, [tucker_last, tucker_first] = partial_tucker(
                weights, dims=[0, 1], ranks=core_ranks, n_iters=n_iters, tol=tol
            )  # TODO dims for other dimensions

            decomposed_weights.update(
                {
                    1: np.expand_dims(tucker_first.transpose(1, 0), axis=[2, 3]),
                    2: tucker_core,
                    4: np.expand_dims(tucker_last, axis=[2, 3]),
                }
            )

        if mode in [2, 3]:
            # tucker_stick
            if mode == 2:
                first_stick, last_stick = tucker_stick(
                    weights, dim=1, rank=stick_rank
                )  # TODO dim for other dimensions
            # partial_tucker + tucker_stick
            elif mode == 3:
                first_stick, last_stick = tucker_stick(
                    tucker_core, dim=1, rank=stick_rank
                )  # TODO dim for other dimensions
            decomposed_weights.update({2: first_stick, 3: last_stick})
        return decomposed_weights

    def _create_new_convs(self, core_ranks, stick_rank, mode, need_bias=True):
        """
        Create new conv layers for the corresponding mode of decomposition
        """
        new_convs = {}
        if mode in [1, 3]:
            # pointwise convolution
            tucker_first_conv = torch.nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=core_ranks[0],
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=self.dilation,
                bias=False,
            )
            # core convolution
            tucker_core_conv = torch.nn.Conv2d(
                in_channels=core_ranks[0],
                out_channels=core_ranks[1],
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                bias=False,
            )
            # pointwise convolution
            tucker_last_conv = torch.nn.Conv2d(
                in_channels=core_ranks[1],
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=self.dilation,
                bias=need_bias,
            )
            new_convs.update(
                {1: tucker_first_conv, 2: tucker_core_conv, 4: tucker_last_conv}
            )
        if mode in [2, 3]:
            assert stick_rank is not None, "stick_rank (for tucker_stick) is None"
            # tucker_stick
            if mode == 2:
                # conv2d(ker_size=(ker_h, 1))
                stick_first_conv = torch.nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=stick_rank,
                    kernel_size=(self.kernel_size[0], 1),
                    stride=(self.stride[0], 1),
                    padding=(self.padding[0], 0),
                    dilation=(self.dilation[0], 1),
                    bias=False,
                )
                # conv2d(ker_size=(1, ker_w))
                stick_last_conv = torch.nn.Conv2d(
                    in_channels=stick_rank,
                    out_channels=self.out_channels,
                    kernel_size=(1, self.kernel_size[1]),
                    stride=(1, self.stride[1]),
                    padding=(0, self.padding[1]),
                    dilation=(1, self.dilation[1]),
                    bias=need_bias,
                )
            # partial_tucker + tucker_stick
            elif mode == 3:
                # conv2d(ker_size=(ker_h, 1))
                stick_first_conv = torch.nn.Conv2d(
                    in_channels=core_ranks[0],
                    out_channels=stick_rank,
                    kernel_size=(self.kernel_size[0], 1),
                    stride=(self.stride[0], 1),
                    padding=(self.padding[0], 0),
                    dilation=(self.dilation[0], 1),
                    bias=False,
                )
                # conv2d(ker_size=(1, ker_w))
                stick_last_conv = torch.nn.Conv2d(
                    in_channels=stick_rank,
                    out_channels=core_ranks[1],
                    kernel_size=(1, self.kernel_size[1]),
                    stride=(1, self.stride[1]),
                    padding=(0, self.padding[1]),
                    dilation=(1, self.dilation[1]),
                    bias=False,
                )
            new_convs.update({2: stick_first_conv, 3: stick_last_conv})
        return new_convs
