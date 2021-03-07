import numpy as np
import torch
import torch.nn as nn


class GaussianKernel(nn.Module):
    def __init__(self, logs2):
        super(GaussianKernel, self).__init__()
        self.logs2 = nn.Parameter(logs2)

    def forward(self, x, y):
        return torch.exp(-(0.5 / torch.exp(self.logs2)) * torch.cdist(x, y) ** 2)

    def __call__(self, x, y):
        return self.forward(x, y)


class LaplaceKernel(nn.Module):
    def __init__(self, logs):
        super(LaplaceKernel, self).__init__()
        self.logs = nn.Parameter(logs)

    def forward(self, x, y):
        return torch.exp(-torch.exp(self.logs) * torch.cdist(x, y))

    def __call__(self, x, y):
        return self.forward(x, y)


class CosGaussianKernelDiagonal(nn.Module):
    def __init__(self, mu, logs2diag):
        super(CosGaussianKernelDiagonal, self).__init__()
        self.mu = nn.Parameter(mu)
        self.logs2diag = nn.Parameter(logs2diag)

    def forward(self, x, y):
        S_sqrt = torch.diag(torch.exp(self.logs2diag) ** 0.5)
        x_, y_ = torch.matmul(x, S_sqrt), torch.matmul(y, S_sqrt)
        m_x = torch.matmul(x, self.mu)
        m_y = torch.matmul(y, self.mu)
        M = m_x - m_y.transpose(2, 1)
        return torch.cos(M) * torch.exp(-0.5 * torch.cdist(x_, y_) ** 2)

    def __call__(self, x, y):
        return self.forward(x, y)


class CosKernel(nn.Module):
    def __init__(self, omega):
        super(CosKernel, self).__init__()
        self.omega = nn.Parameter(omega)

    def forward(self, x, y):
        m_x = torch.matmul(x, self.omega)
        m_y = torch.matmul(y, self.omega).transpose(-1, -2)
        M = m_x - m_y
        return torch.cos(M)

    def __call__(self, x, y):
        return self.forward(x, y)


class PolynomialKernel(nn.Module):
    """Polynomial kernel (gamma x.T y + c)^d"""

    def __init__(self, gamma, c, d):
        super(PolynomialKernel, self).__init__()
        self.gamma = nn.Parameter(gamma)
        self.c = nn.Parameter(c)
        self.d = d

    def forward(self, x, y):
        K = (self.gamma * torch.matmul(x, y.T) + self.c) ** self.d
        return K

    def __call__(self, x, y):
        return self.forward(x, y)


class LinearKernel(nn.Module):
    """Linear kernel (identity) x.T y"""

    def __init__(self):
        super(LinearKernel, self).__init__()

    def forward(self, x, y):
        K = torch.matmul(x, y.transpose(-1, -2))  # If something breaks switch to (1, 2)
        return K

    def __call__(self, x, y):
        return self.forward(x, y)


class BochnerKernel(nn.Module):
    def __init__(
        self, latent_d, latent_dist, pf_map, device=None, sample_automatically=None,
    ):
        super(BochnerKernel, self).__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_d = latent_d
        # latent has to implement a "sample" method taking n and d as inputs
        # see https://pytorch.org/docs/stable/distributions.html
        self.latent_dist = latent_dist
        # TODO: there are two established strategies
        # concatenating cos and sin vs cos + a uniform offset
        # cos_sin vs cos_unif, we only use cos_unif here
        # NOTE: To make distribution live on GPU we need to initialise
        # it with tensors that already live there
        self.b_dist = torch.distributions.Uniform(
            torch.tensor(0.0).to(device), torch.tensor(2.0 * np.pi).to(device)
        )

        # pf is for pushforward and is a torch trainable function
        self.pf_map = pf_map
        self.device = device
        self.sample_automatically = sample_automatically

    def sample_features(self, D, B=1):
        self.latents = self.latent_dist.sample((B, D, self.latent_d)).to(
            self.device
        )  # B x D x latent_d
        self.omegas = self.pf_map(self.latents)  # B x D x d
        self.bs = self.b_dist.sample((B, D, 1))  # B x D x 1
        self.D = D

    def random_feature_map(self, x):
        if self.sample_automatically is not None:
            self.sample_features(self.sample_automatically)
        t = torch.matmul(x, self.omegas.transpose(-1, -2)) + self.bs.transpose(
            -1, -2
        )  # b x n x D
        return ((2.0 / self.D) ** 0.5) * torch.cos(t)

    def random_kernel(self, x, y):
        # TODO: There's a closed form for this but we use the feature map
        # see cite:sutherland15_error_random_fourier_featur
        Z_x = self.random_feature_map(x)  # b x n x D
        Z_y = self.random_feature_map(y)  # b x m x D
        # Symmetrise, possibly add ridge to make psd
        K = torch.matmul(Z_x, Z_y.transpose(1, 2)) + 1e-8
        return K

    def forward(self, x, y):
        return self.random_kernel(x, y)

    def __call__(self, x, y):
        return self.forward(x, y)
