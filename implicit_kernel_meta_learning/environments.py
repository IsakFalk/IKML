from functools import reduce

import numpy as np
import torch

from .kernels import GaussianKernel


class KernelRegressionEnvironment:
    def __init__(self, num_basis_elems, d, kernel, noise=0.1, device="cpu"):
        self.num_basis_elems = num_basis_elems
        self.d = d
        # We don't want to training the environment
        for param in kernel.parameters():
            param.requires_grad = False
        self.kernel = kernel
        self.noise = noise
        self.device = device

    def generate_task(self, n_tr, n_val):
        X_bases = torch.rand(self.num_basis_elems, self.d).to(self.device)
        alphas = 2 * torch.randn(self.num_basis_elems).to(self.device)

        def f_(X):
            return self.kernel(X, X_bases) @ alphas

        X_tr = torch.rand(n_tr, self.d).to(self.device)
        X_val = torch.rand(n_val, self.d).to(self.device)

        Y_tr = f_(X_tr) + self.noise * torch.randn(n_tr).to(self.device)
        Y_val = f_(X_val) + self.noise * torch.randn(n_val).to(self.device)
        return X_tr, X_val, Y_tr, Y_val, f_


class GaussianRegressionEnvironment(KernelRegressionEnvironment):
    def __init__(self, num_basis_elems, d, s2, noise=0.1):
        super().__init__(num_basis_elems, d, GaussianKernel(s2), noise)


class MixedKernelRegressionEnvironment:
    """Gives a mixture environment define by the kernels passed as __init__ arguments"""

    def __init__(self, num_basis_elems_per_kernel, d, kernels, noise=0.1):
        self.num_basis_elems_per_kernel = num_basis_elems_per_kernel
        self.d = d
        for kernel in kernels:
            for param in kernel.parameters():
                param.requires_grad = False
        self.kernels = kernels
        self.noise = noise

        self.marginal_dist = lambda *n: 10 * torch.rand(*n) - 5
        self.base_inputs_dist = lambda *n: 10 * torch.rand(*n) - 5
        self.alpha_dist = lambda n: 2 * torch.randn(n)

    def generate_task(self, n_tr, n_val):
        # The below generates outputs for each kernel
        # then combines them using reduce which simply sums the outputs
        X_bases_list = [
            self.base_inputs_dist(self.num_basis_elems_per_kernel, self.d)
            for _ in self.kernels
        ]
        alphas_list = [
            2 * self.alpha_dist(self.num_basis_elems_per_kernel) for _ in self.kernels
        ]
        zip(X_bases_list, alphas_list, self.kernels)
        # num_bases = self.num_basis_elems_per_kernel * len(self.kernels)

        def f_(X):
            zipped = zip(X_bases_list, alphas_list, self.kernels)
            outputs = [
                (kernel(X, X_bases) @ alphas) for (X_bases, alphas, kernel) in zipped
            ]
            return reduce(lambda x, y: x + y, outputs)

        X_tr = self.marginal_dist(n_tr, self.d)
        X_val = self.marginal_dist(n_val, self.d)

        Y_tr = f_(X_tr) + self.noise * torch.randn(n_tr)
        Y_val = f_(X_val) + self.noise * torch.randn(n_val)

        return X_tr, X_val, Y_tr, Y_val, f_


class SinusoidRegressionEnvironment:
    def __init__(self, noise=0.0):
        self.noise = noise
        self.amp_dist = torch.distributions.Uniform(0.1, 5.0)
        self.phase_dist = torch.distributions.Uniform(0.0, np.pi)
        self.marginal_dist = torch.distributions.Uniform(-5.0, 5.0)

    def generate_task(self, n_tr, n_val):
        # Amplitude and phase of sinusoid in task
        A = self.amp_dist.sample((1,)).squeeze()
        phi = self.phase_dist.sample((1,)).squeeze()

        def f_(X):
            return A * torch.sin(X + phi).squeeze()

        # Sample data
        X_tr = self.marginal_dist.sample((n_tr, 1))
        Y_tr = f_(X_tr) + self.noise * torch.randn(n_tr)

        X_val = self.marginal_dist.sample((n_val, 1))
        Y_val = f_(X_val) + self.noise * torch.randn(n_val)
        return X_tr, X_val, Y_tr, Y_val, f_
