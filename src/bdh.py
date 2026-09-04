"""Minimal BDH-GPU with an explicit, inspectable synaptic state.

Derived from the reference implementation released by Pathway Technology, Inc.
under the MIT licence (github.com/pathwaycom/bdh, bdh.py). See NOTICE.md.

What we changed and why
-----------------------
1. forward_recurrent() materialises the synaptic state rho_{t,l} at every step.
   The reference computes attention in the token-parallel (quadratic) form,
   which is mathematically equivalent but never builds rho. We need rho as a
   concrete object because the whole explainer is about reading and ablating it.
2. u_decay exposes the damping half of U. Definition 4 of the BDH paper
   (arXiv:2509.26507) states U is "a diagonal or block-diagonal matrix
   representing local rotation or damping of state (such as ALiBi or RoPE)".
   The public reference instantiates U as RoPE, i.e. rotation with u = 1.
   Setting u_decay < 1 is the damping case of the same published U.
   DEFAULT IS 1.0, which reproduces the reference exactly.
3. Ablation hooks, so a learner's click can zero real state entries.

Notation map (paper -> this code):
    n (neurons)        -> N * n_head
    d (embedding)      -> n_embd, D
    E   in R^{d x n}   -> decoder (transposed)
    D_x in R^{n x d}   -> encoder
    D_y in R^{n x d}   -> encoder_v
    rho in R^{d x n}   -> state tensor (B, nh, D, N)
    sigma in R^{n x n} -> D_y @ rho, the neuron-neuron synapse matrix (rank <= d)
"""

import dataclasses
import math

import torch
import torch.nn.functional as F
from torch import nn


@dataclasses.dataclass
class BDHConfig:
    n_layer: int = 2
    n_embd: int = 64
    dropout: float = 0.1
    n_head: int = 1
    mlp_internal_dim_multiplier: int = 8
    vocab_size: int = 64
    u_decay: float = 1.0  # 1.0 == reference (pure RoPE rotation, no damping)

    @property
    def N(self) -> int:
        return self.mlp_internal_dim_multiplier * self.n_embd // self.n_head


def get_freqs(n, theta, dtype):
    """RoPE frequencies. Identical to the reference implementation."""

    def quantize(t, q=2):
        return (t / q).floor() * q

    return (
        1.0
        / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
        / (2 * math.pi)
    )


def rope(phases, v):
    """Rotate v by phases. Identical to the reference implementation."""
    v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
    phases = (phases % 1) * (2 * math.pi)
    return (v * torch.cos(phases)).to(v.dtype) + (v_rot * torch.sin(phases)).to(v.dtype)


class BDH(nn.Module):
    def __init__(self, config: BDHConfig):
        super().__init__()
        self.config = config
        nh, D, N = config.n_head, config.n_embd, config.N

        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.lm_head = nn.Parameter(
            torch.zeros((D, config.vocab_size)).normal_(std=0.02)
        )
        self.embed = nn.Embedding(config.vocab_size, D)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.drop = nn.Dropout(config.dropout)
        self.register_buffer(
            "freqs", get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)
        )

    # ------------------------------------------------------------------
    # Token-parallel form: what we train with. Reference-equivalent.
    # ------------------------------------------------------------------
    def forward(self, idx, targets=None):
        C = self.config
        B, T = idx.size()
        u = C.u_decay

        x = self.ln(self.embed(idx).unsqueeze(1))  # B, 1, T, D

        pos = torch.arange(0, T, device=idx.device, dtype=torch.float32)
        r_phases = pos.view(1, 1, -1, 1) * self.freqs

        # decay factor u^(t - tau), folded into the score matrix
        decay = None
        if u != 1.0:
            dt = pos.view(T, 1) - pos.view(1, T)  # t - tau
            decay = torch.where(
                dt > 0, u ** dt.clamp(min=0.0), torch.zeros_like(dt)
            ).view(1, 1, T, T)

        for _ in range(C.n_layer):
            x_sparse = F.relu(x @ self.encoder)  # B, nh, T, N
            QR = rope(r_phases, x_sparse)

            scores = (QR @ QR.mT).tril(diagonal=-1)  # strictly causal: tau < t
            if decay is not None:
                scores = scores * decay
            yKV = self.ln(scores @ x)  # B, nh, T, D

            y_sparse = F.relu(yKV @ self.encoder_v)
            xy_sparse = self.drop(x_sparse * y_sparse)

            yMLP = (
                xy_sparse.transpose(1, 2).reshape(B, 1, T, C.N * C.n_head)
                @ self.decoder
            )
            x = self.ln(x + self.ln(yMLP))

        logits = x.view(B, T, C.n_embd) @ self.lm_head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    # ------------------------------------------------------------------
    # Recurrent form: same maths, but rho is a real tensor we can touch.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward_recurrent(self, idx, ablate=None, record=False):
        """Step through the sequence carrying rho explicitly.

        ablate: optional callable (layer, t, rho) -> rho, applied to the state
                after the write at step t and before the read at step t+1.
        record: if True, also return per-step diagnostics.

        Returns (logits, info); info["rho"][l] is the final state of layer l,
        shape (B, nh, D, N).
        """
        C = self.config
        B, T = idx.size()
        nh, D, N = C.n_head, C.n_embd, C.N
        u = C.u_decay
        dev = idx.device

        was_training = self.training
        self.eval()

        x_layer_in = self.ln(self.embed(idx).unsqueeze(1))  # B,1,T,D
        rho = [torch.zeros(B, nh, D, N, device=dev) for _ in range(C.n_layer)]
        info = {"rho": rho, "x_sparse": [], "delta": [], "read": []}

        # Layer l consumes the whole output of layer l-1, so we loop layers
        # outside and tokens inside.
        for l in range(C.n_layer):
            x_sparse_all = F.relu(x_layer_in @ self.encoder)  # B,nh,T,N
            yKV_steps = []
            per_step_delta = []
            for t in range(T):
                phase = torch.full((1, 1, 1, 1), float(t), device=dev) * self.freqs
                xs_t = x_sparse_all[:, :, t : t + 1, :]  # B,nh,1,N
                qr_t = rope(phase, xs_t)[:, :, 0, :]  # B,nh,N

                # READ from state written by tau < t
                a_t = torch.einsum("bhdn,bhn->bhd", rho[l], qr_t)
                yKV_steps.append(a_t.unsqueeze(2))

                # WRITE: rank-1 Hebbian outer product, then apply U (damping)
                v_t = x_layer_in[:, 0, t, :]  # B,D
                delta = torch.einsum("bd,bhn->bhdn", v_t, qr_t)
                rho[l] = (rho[l] + delta) * u
                if record:
                    per_step_delta.append(delta)
                if ablate is not None:
                    rho[l] = ablate(l, t, rho[l])

            yKV = self.ln(torch.cat(yKV_steps, dim=2))
            y_sparse = F.relu(yKV @ self.encoder_v)
            xy_sparse = x_sparse_all * y_sparse
            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            x_layer_in = self.ln(x_layer_in + self.ln(yMLP))

            if record:
                info["x_sparse"].append(x_sparse_all)
                info["delta"].append(per_step_delta)
                info["read"].append(yKV)

        logits = x_layer_in.view(B, T, D) @ self.lm_head
        if was_training:
            self.train()
        return logits, info

    def sigma(self, rho_layer):
        """Neuron-neuron synapse matrix sigma = D_y @ rho (n x n, rank <= d).

        This is the object Eq. (6) of the paper calls sigma; BDH-GPU carries it
        in the factorised form rho (d x n). Materialising it is only sensible
        for small N -- it is (N*nh) square.
        """
        # encoder_v is (nh, D, N), i.e. D_y transposed per head
        return torch.einsum("hdm,bhdn->bhmn", self.encoder_v, rho_layer)
