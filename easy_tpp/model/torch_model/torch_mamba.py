# Beginning taken from https://github.com/alxndrTL/mamba.py/blob/main/mambapy/mamba.py

import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

"""

An implementation of the parallel scan operation in PyTorch (Blelloch version).
Please see docs/pscan.ipynb for a detailed explanation of what happens here.

"""


def npo2(len):
    """
    Returns the next power of 2 above len
    """

    return 2 ** math.ceil(math.log2(len))


def pad_npo2(X):
    """
    Pads input length dim to the next power of 2

    Args:
        X : (B, L, D, N)

    Returns:
        Y : (B, npo2(L), D, N)
    """

    len_npo2 = npo2(X.size(1))
    pad_tuple = (0, 0, 0, 0, 0, len_npo2 - X.size(1))
    return F.pad(X, pad_tuple, "constant", 0)


class PScan(torch.autograd.Function):
    @staticmethod
    def pscan(A, X):
        # A : (B, D, L, N)
        # X : (B, D, L, N)

        # modifies X in place by doing a parallel scan.
        # more formally, X will be populated by these values :
        # H[t] = A[t] * H[t-1] + X[t] with H[0] = 0
        # which are computed in parallel (2*log2(T) sequential steps (ideally), instead of T sequential steps)

        # only supports L that is a power of two (mainly for a clearer code)

        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        # up sweep (last 2 steps unfolded)
        Aa = A
        Xa = X
        for _ in range(num_steps - 2):
            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
            Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])

            Aa = Aa[:, :, :, 1]
            Xa = Xa[:, :, :, 1]

        # we have only 4, 2 or 1 nodes left
        if Xa.size(2) == 4:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            Aa[:, :, 1].mul_(Aa[:, :, 0])

            Xa[:, :, 3].add_(
                Aa[:, :, 3].mul(Xa[:, :, 2] + Aa[:, :, 2].mul(Xa[:, :, 1]))
            )
        elif Xa.size(2) == 2:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            return
        else:
            return

        # down sweep (first 2 steps unfolded)
        Aa = A[:, :, 2 ** (num_steps - 2) - 1 : L : 2 ** (num_steps - 2)]
        Xa = X[:, :, 2 ** (num_steps - 2) - 1 : L : 2 ** (num_steps - 2)]
        Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 1]))
        Aa[:, :, 2].mul_(Aa[:, :, 1])

        for k in range(num_steps - 3, -1, -1):
            Aa = A[:, :, 2**k - 1 : L : 2**k]
            Xa = X[:, :, 2**k - 1 : L : 2**k]

            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            Xa[:, :, 1:, 0].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1]))
            Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])

    @staticmethod
    def pscan_rev(A, X):
        # A : (B, D, L, N)
        # X : (B, D, L, N)

        # the same function as above, but in reverse
        # (if you flip the input, call pscan, then flip the output, you get what this function outputs)
        # it is used in the backward pass

        # only supports L that is a power of two (mainly for a clearer code)

        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        # up sweep (last 2 steps unfolded)
        Aa = A
        Xa = X
        for _ in range(num_steps - 2):
            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            Xa[:, :, :, 0].add_(Aa[:, :, :, 0].mul(Xa[:, :, :, 1]))
            Aa[:, :, :, 0].mul_(Aa[:, :, :, 1])

            Aa = Aa[:, :, :, 0]
            Xa = Xa[:, :, :, 0]

        # we have only 4, 2 or 1 nodes left
        if Xa.size(2) == 4:
            Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 3]))
            Aa[:, :, 2].mul_(Aa[:, :, 3])

            Xa[:, :, 0].add_(
                Aa[:, :, 0].mul(Xa[:, :, 1].add(Aa[:, :, 1].mul(Xa[:, :, 2])))
            )
        elif Xa.size(2) == 2:
            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1]))
            return
        else:
            return

        # down sweep (first 2 steps unfolded)
        Aa = A[:, :, 0 : L : 2 ** (num_steps - 2)]
        Xa = X[:, :, 0 : L : 2 ** (num_steps - 2)]
        Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 2]))
        Aa[:, :, 1].mul_(Aa[:, :, 2])

        for k in range(num_steps - 3, -1, -1):
            Aa = A[:, :, 0 : L : 2**k]
            Xa = X[:, :, 0 : L : 2**k]

            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)

            Xa[:, :, :-1, 1].add_(Aa[:, :, :-1, 1].mul(Xa[:, :, 1:, 0]))
            Aa[:, :, :-1, 1].mul_(Aa[:, :, 1:, 0])

    @staticmethod
    def forward(ctx, A_in, X_in):
        """
        Applies the parallel scan operation, as defined above. Returns a new tensor.
        If you can, privilege sequence lengths that are powers of two.

        Args:
            A_in : (B, L, D, N)
            X_in : (B, L, D, N)

        Returns:
            H : (B, L, D, N)
        """

        L = X_in.size(1)

        # cloning is requiered because of the in-place ops
        if L == npo2(L):
            A = A_in.clone()
            X = X_in.clone()
        else:
            # pad tensors (and clone btw)
            A = pad_npo2(A_in)  # (B, npo2(L), D, N)
            X = pad_npo2(X_in)  # (B, npo2(L), D, N)

        # prepare tensors
        A = A.transpose(2, 1)  # (B, D, npo2(L), N)
        X = X.transpose(2, 1)  # (B, D, npo2(L), N)

        # parallel scan (modifies X in-place)
        PScan.pscan(A, X)

        ctx.save_for_backward(A_in, X)

        # slice [:, :L] (cut if there was padding)
        return X.transpose(2, 1)[:, :L]

    @staticmethod
    def backward(ctx, grad_output_in):
        """
        Flows the gradient from the output to the input. Returns two new tensors.

        Args:
            ctx : A_in : (B, L, D, N), X : (B, D, L, N)
            grad_output_in : (B, L, D, N)

        Returns:
            gradA : (B, L, D, N), gradX : (B, L, D, N)
        """

        A_in, X = ctx.saved_tensors

        L = grad_output_in.size(1)

        # cloning is requiered because of the in-place ops
        if L == npo2(L):
            grad_output = grad_output_in.clone()
            # the next padding will clone A_in
        else:
            grad_output = pad_npo2(grad_output_in)  # (B, npo2(L), D, N)
            A_in = pad_npo2(A_in)  # (B, npo2(L), D, N)

        # prepare tensors
        grad_output = grad_output.transpose(2, 1)
        A_in = A_in.transpose(2, 1)  # (B, D, npo2(L), N)
        A = torch.nn.functional.pad(
            A_in[:, :, 1:], (0, 0, 0, 1)
        )  # (B, D, npo2(L), N) shift 1 to the left (see hand derivation)

        # reverse parallel scan (modifies grad_output in-place)
        PScan.pscan_rev(A, grad_output)

        Q = torch.zeros_like(X)
        Q[:, :, 1:].add_(X[:, :, :-1] * grad_output[:, :, 1:])

        return Q.transpose(2, 1)[:, :L], grad_output.transpose(2, 1)[:, :L]


pscan = PScan.apply

"""

This file closely follows the mamba_simple.py from the official Mamba implementation, and the mamba-minimal by @johnma2006.
The major differences are :
-the convolution is done with torch.nn.Conv1d
-the selective scan is done in PyTorch

A sequential version of the selective scan is also available for comparison. Also, it is possible to use the official Mamba implementation.

This is the structure of the torch modules :
- A Mamba model is composed of several layers, which are ResidualBlock.
- A ResidualBlock is composed of a MambaBlock, a normalization, and a residual connection : ResidualBlock(x) = mamba(norm(x)) + x
- This leaves us with the MambaBlock : its input x is (B, L, D) and its outputs y is also (B, L, D) (B=batch size, L=seq len, D=model dim).
First, we expand x into (B, L, 2*ED) (where E is usually 2) and split it into x and z, each (B, L, ED).
Then, we apply the short 1d conv to x, followed by an activation function (silu), then the SSM.
We then multiply it by silu(z).
See Figure 3 of the paper (page 8) for a visual representation of a MambaBlock.

"""


@dataclass
class MambaConfig:
    d_model: int  # D
    n_layers: int
    dt_rank: Union[int, str] = "auto"
    d_state: int = 16  # N in paper/comments
    expand_factor: int = 2  # E in paper/comments
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"  # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    rms_norm_eps: float = 1e-5
    base_std: float = 0.02

    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = False  # apply layernorms to internal activations

    mup: bool = False
    mup_base_width: float = 128  # width=d_model

    pscan: bool = True  # use parallel scan mode or sequential mode when training
    use_cuda: bool = False  # use official CUDA implementation when training (not compatible with (b)float16)

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model  # E*D = ED in comments

        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)

        # muP
        if self.mup:
            self.mup_width_mult = self.d_model / self.mup_base_width


class Mamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList(
            [ResidualBlock(config) for _ in range(config.n_layers)]
        )

    def forward(self, x, dt):
        # x : (B, L, D)
        # dt: (B, L)

        # y : (B, L, D)

        for layer in self.layers:
            x = layer(x, dt)

        return x

    def step(self, x, caches):
        raise NotImplementedError
        # # x : (B, L, D)
        # # caches : [cache(layer) for all layers], cache : (h, inputs)

        # # y : (B, L, D)
        # # caches : [cache(layer) for all layers], cache : (h, inputs)

        # for i, layer in enumerate(self.layers):
        #     x, caches[i] = layer.step(x, caches[i])

        # return x, caches


class ResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.mixer = MambaBlock(config)
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps, config.mup)

    def forward(self, x, dt):
        # x : (B, L, D)
        # dt: (B, L)

        # output : (B, L, D)

        output = self.mixer(self.norm(x), dt) + x
        return output

    def step(self, x, cache):
        raise NotImplementedError
        # # x : (B, D)
        # # cache : (h, inputs)
        #         # h : (B, ED, N)
        #         # inputs: (B, ED, d_conv-1)

        # # output : (B, D)
        # # cache : (h, inputs)

        # output, cache = self.mixer.step(self.norm(x), cache)
        # output = output + x
        # return output, cache


class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        # projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        self.conv1d = nn.Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            kernel_size=config.d_conv,
            bias=config.conv_bias,
            groups=config.d_inner,
            padding=config.d_conv - 1,
        )

        # projects x to input-dependent delta, B, C
        self.x_proj = nn.Linear(
            config.d_inner, config.dt_rank + 2 * config.d_state, bias=False
        )

        # projects delta from dt_rank to d_inner
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        # dt initialization
        # dt weights
        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # delta bias
        dt = torch.exp(
            torch.rand(config.d_inner)
            * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(
            -torch.expm1(-dt)
        )  # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # S4D real initialization
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(
            config.d_inner, 1
        )
        self.A_log = nn.Parameter(
            torch.log(A)
        )  # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.D._no_weight_decay = True

        # projects block output from ED back to D
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

        # used in jamba
        if self.config.inner_layernorms:
            self.dt_layernorm = RMSNorm(
                self.config.dt_rank, config.rms_norm_eps, config.mup
            )
            self.B_layernorm = RMSNorm(
                self.config.d_state, config.rms_norm_eps, config.mup
            )
            self.C_layernorm = RMSNorm(
                self.config.d_state, config.rms_norm_eps, config.mup
            )
        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

        if self.config.use_cuda:
            try:
                from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

                self.selective_scan_cuda = selective_scan_fn
            except ImportError:
                print("Failed to import mamba_ssm. Falling back to mamba.py.")
                self.config.use_cuda = False

    def _apply_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def forward(self, x, dt):
        # x : (B, L, D)
        # dt: (B, L)

        # y : (B, L, D)

        _, L, _ = x.shape

        xz = self.in_proj(x)  # (B, L, 2*ED)
        x, z = xz.chunk(2, dim=-1)  # (B, L, ED), (B, L, ED)

        # x branch
        x = x.transpose(1, 2)  # (B, ED, L)
        x = self.conv1d(x)[
            :, :, :L
        ]  # depthwise convolution over time, with a short filter
        x = x.transpose(1, 2)  # (B, L, ED)

        x = F.silu(x)
        y = self.ssm(x, z, dt)

        if self.config.use_cuda:
            output = self.out_proj(y)  # (B, L, D)
            return output  # the rest of the operations are done in the ssm function (fused with the CUDA pscan)

        # z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output)  # (B, L, D)

        return output

    def ssm(self, x, z, dt):
        # x : (B, L, ED)
        # dt: (B, L)

        # y : (B, L, ED)

        A = -torch.exp(self.A_log.float())  # (ED, N)
        D = self.D.float()

        deltaBC = self.x_proj(x)  # (B, L, dt_rank+2*N)
        delta, B, C = torch.split(
            deltaBC,
            [self.config.dt_rank, self.config.d_state, self.config.d_state],
            dim=-1,
        )  # (B, L, dt_rank), (B, L, N), (B, L, N)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = self.dt_proj.weight @ delta.transpose(
            1, 2
        )  # (ED, dt_rank) @ (B, L, dt_rank) -> (B, ED, L)
        # Keep above operations the same even though we will discard later
        delta = dt[:, None, :].expand(-1, delta.shape[1], -1)

        # here we just apply the matrix mul operation of delta = softplus(dt_proj(delta))
        # the rest will be applied later (fused if using cuda)

        # choose which selective_scan function to use, according to config
        if self.config.use_cuda:
            # these are unfortunately needed for the selective_scan_cuda function
            x = x.transpose(1, 2)
            B = B.transpose(1, 2)
            C = C.transpose(1, 2)
            z = z.transpose(1, 2)

            # "softplus" + "bias" + "y * silu(z)" operations are fused
            y = self.selective_scan_cuda(
                x,
                delta,
                A,
                B,
                C,
                D,
                z=z,
                delta_softplus=True,
                delta_bias=self.dt_proj.bias.float(),
            )
            y = y.transpose(1, 2)  # (B, L, ED)

        else:
            delta = delta.transpose(1, 2)
            # Even though we replaced delta with dt, MHP still applies the softplus to it
            delta = F.softplus(delta)  # + self.dt_proj.bias)

            if self.config.pscan:
                y = self.selective_scan(x, delta, A, B, C, D)
            else:
                y = self.selective_scan_seq(x, delta, A, B, C, D)

        return y

    def selective_scan(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  # (B, L, ED, N)

        hs = pscan(deltaA, BX)

        y = (hs @ C.unsqueeze(-1)).squeeze(
            3
        )  # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y

    def selective_scan_seq(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  # (B, L, ED, N)

        h = torch.zeros(
            x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device
        )  # (B, ED, N)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)

        hs = torch.stack(hs, dim=1)  # (B, L, ED, N)

        y = (hs @ C.unsqueeze(-1)).squeeze(
            3
        )  # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y

    # -------------------------- inference -------------------------- #
    """
    Concerning auto-regressive inference

    The cool part of using Mamba : inference is constant wrt to sequence length
    We just have to keep in cache, for each layer, two things :
    - the hidden state h (which is (B, ED, N)), as you typically would when doing inference with a RNN
    - the last d_conv-1 inputs of the layer, to be able to compute the 1D conv which is a convolution over the time dimension
      (d_conv is fixed so this doesn't incur a growing cache as we progress on generating the sequence)
      (and d_conv is usually very small, like 4, so we just have to "remember" the last 3 inputs)

    Concretely, these two quantities are put inside a cache tuple, and are named h and inputs respectively.
    h is (B, ED, N), and inputs is (B, ED, d_conv-1)
    The MambaBlock.step() receives this cache, and, along with outputing the output, alos outputs the updated cache for the next call.

    The cache object is initialized as follows : (None, torch.zeros()).
    When h is None, the selective scan function detects it and start with h=0.
    The torch.zeros() isn't a problem (it's same as just feeding the input, because the conv1d is padded)

    As we need one such cache variable per layer, we store a caches object, which is simply a list of cache object. (See mamba_lm.py)
    """

    def step(self, x, cache):
        raise NotImplementedError

    def ssm_step(self, x, h):
        raise NotImplementedError


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, use_mup: bool = False):
        super().__init__()

        self.use_mup = use_mup
        self.eps = eps

        # https://arxiv.org/abs/2404.05728, RMSNorm gains prevents muTransfer (section 4.2.3)
        if not use_mup:
            self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        if not self.use_mup:
            return output * self.weight
        else:
            return output


from easy_tpp.model.torch_model.torch_baselayer import ScaledSoftplus
from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel


class MHP(TorchBaseModel):
    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.
        """
        super(MHP, self).__init__(model_config)
        self.d_model = model_config.hidden_size

        self.mamba_config = MambaConfig(
            d_model=self.d_model,
            n_layers=model_config.num_layers,
            d_state=16,
            expand_factor=2,
            pscan=True,
            use_cuda=False,  # Cuda is allowed in general, this just disables fused kernels
        )

        self.mamba = Mamba(self.mamba_config)

        self.factor_intensity_base = nn.Parameter(
            torch.empty([1, self.num_event_types], device=self.device)
        )
        self.factor_intensity_decay = nn.Parameter(
            torch.empty([1, self.num_event_types], device=self.device)
        )
        nn.init.xavier_normal_(self.factor_intensity_base)
        nn.init.xavier_normal_(self.factor_intensity_decay)

        # convert hidden vectors into event-type-sized vector
        self.layer_intensity_hidden = nn.Linear(self.d_model, self.num_event_types)
        self.softplus = ScaledSoftplus(self.num_event_types)

    def forward(self, time_delta_seqs, type_seqs):
        """Call the model

        Args:
            time_delta_seqs (tensor): [batch_size, seq_len], timestamp seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.

        Returns:
            tensor: hidden states at event times.
        """
        # [batch_size, seq_len, hidden_size]
        return self.mamba.forward(
            self.layer_type_emb(type_seqs),
            time_delta_seqs,
        )

    def loglike_loss(self, batch, **kwargs):
        """Compute the loglike loss.

        Args:
            batch (tuple, list): batch input.

        Returns:
            tuple: loglike loss, num events.
        """
        _, time_delta_seqs, type_seqs, batch_non_pad_mask, _ = batch

        # 1. compute event-loglik
        # [batch_size, seq_len, hidden_size]
        enc_out = self.forward(time_delta_seqs[:, :-1], type_seqs[:, :-1])

        # [batch_size, seq_len, num_event_types]
        # update time decay based on Equation (6)
        # [1, 1, num_event_types]
        factor_intensity_decay = self.factor_intensity_decay[None, ...]
        factor_intensity_base = self.factor_intensity_base[None, ...]

        # update time decay based on Equation (6)
        # [batch_size, seq_len, num_event_types]
        intensity_states = (
            factor_intensity_decay * time_delta_seqs[:, 1:, None]
            + self.layer_intensity_hidden(enc_out)
            + factor_intensity_base
        )

        lambda_at_event = self.softplus(intensity_states)

        # 2. compute non-event-loglik (using MC sampling to compute integral)
        # 2.1 sample dtimes
        # [batch_size, seq_len, num_sample]
        sample_dtimes = self.make_dtime_loss_samples(time_delta_seqs[:, 1:])

        # 2.2 compute intensities at sampled times
        # [batch_size, num_times = max_len - 1, num_sample, event_num]
        state_t_sample = self.compute_states_at_sample_times(
            event_states=enc_out, sample_dtimes=sample_dtimes
        )
        lambda_t_sample = self.softplus(state_t_sample)

        event_ll, non_event_ll, num_events, mark_ll, time_ll_pos = (
            self.compute_loglikelihood(
                lambda_at_event=lambda_at_event,
                lambdas_loss_samples=lambda_t_sample,
                time_delta_seq=time_delta_seqs[:, 1:],
                seq_mask=batch_non_pad_mask[:, 1:],
                type_seq=type_seqs[:, 1:],
            )
        )
        # lambda_type_mask=type_mask[:, 1:])

        # compute extra statistics
        time_ll = time_ll_pos - non_event_ll

        # compute loss to optimize
        loss = -(event_ll - non_event_ll).sum()

        return_raw_ll = kwargs.get("return_raw_ll", False)
        res_dict = (
            {"non_event_ll": non_event_ll, "mark_intensity": lambda_at_event}
            if return_raw_ll
            else None
        )

        return loss, num_events, mark_ll.sum(), time_ll.sum(), res_dict

    def compute_states_at_sample_times(self, event_states, sample_dtimes):
        """Compute the hidden states at sampled times.

        Args:
            event_states (tensor): [batch_size, seq_len, hidden_size].
            sample_dtimes (tensor): [batch_size, seq_len, num_samples].

        Returns:
            tensor: hidden state at each sampled time.
        """
        # [batch_size, seq_len, 1, hidden_size]
        event_states = event_states[:, :, None, :]

        # [batch_size, seq_len, num_samples, 1]
        sample_dtimes = sample_dtimes[..., None]

        # [1, 1, 1, num_event_types]
        factor_intensity_decay = self.factor_intensity_decay[None, None, ...]
        factor_intensity_base = self.factor_intensity_base[None, None, ...]

        # update time decay based on Equation (6)
        # [batch_size, seq_len, num_samples, num_event_types]
        intensity_states = (
            factor_intensity_decay * sample_dtimes
            + self.layer_intensity_hidden(event_states)
            + factor_intensity_base
        )

        return intensity_states

    def compute_intensities_at_sample_times(
        self, time_seqs, time_delta_seqs, type_seqs, sample_dtimes, **kwargs
    ):
        """Compute hidden states at sampled times.

        Args:
            time_seqs (tensor): [batch_size, seq_len], times seqs.
            time_delta_seqs (tensor): [batch_size, seq_len], time delta seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.
            sample_dtimes (tensor): [batch_size, seq_len, num_samples], sampled inter-event timestamps.

        Returns:
            tensor: [batch_size, seq_len, num_samples, num_event_types], intensity at all sampled times.
        """

        compute_last_step_only = kwargs.get("compute_last_step_only", False)

        # [batch_size, seq_len, num_samples]
        enc_out = self.forward(time_delta_seqs, type_seqs)

        # [batch_size, seq_len, num_samples, hidden_size]
        encoder_output = self.compute_states_at_sample_times(enc_out, sample_dtimes)

        if compute_last_step_only:
            lambdas = self.softplus(encoder_output[:, -1:, :, :])
        else:
            # [batch_size, seq_len, num_samples, num_event_types]
            lambdas = self.softplus(encoder_output)
        return lambdas
