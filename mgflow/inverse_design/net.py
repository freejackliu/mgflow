"""Generator network for inverse design.

Outputs element-wise scores in [0, 1] (sigmoid activated) which are
converted to compositions in extract.py by threshold + renormalization.
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """Generator: latent noise → element-score vector.

    Parameters
    ----------
    n_elements : int
        Number of candidate elements (output dimension).
    latent_dim : int
        Dimension of the latent noise vector (default 100).
    hidden_dims : list of int
        Widths of hidden fully-connected layers.
    dropout : float
        Dropout probability (disabled by default in generation mode).
    leaky_slope : float
        Negative slope for LeakyReLU.
    """

    def __init__(
        self,
        n_elements: int,
        latent_dim: int = 100,
        hidden_dims: list = None,
        dropout: float = 0.0,
        leaky_slope: float = 0.2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        self.n_elements = n_elements
        self.latent_dim = latent_dim

        layers = []
        prev = latent_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.LeakyReLU(leaky_slope, inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, n_elements))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        z : (batch_size, latent_dim) or (latent_dim,)
            Noise vectors sampled from uniform or normal.

        Returns
        -------
        scores : (batch_size, n_elements)
            Element scores in [0, 1].
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return self.net(z)

    @torch.no_grad()
    def generate_batch(self, batch_size: int = 100) -> torch.Tensor:
        """Generate a batch of element-score vectors.

        Noise is sampled from Uniform(-1, 1).

        Returns
        -------
        scores : (batch_size, n_elements)
        """
        device = next(self.parameters()).device
        z = torch.empty(batch_size, self.latent_dim, device=device).uniform_(-1, 1)
        return self.forward(z)

    def __repr__(self):
        return (
            f"Generator(n_elements={self.n_elements}, "
            f"latent_dim={self.latent_dim}, "
            f"params={sum(p.numel() for p in self.parameters())})"
        )
