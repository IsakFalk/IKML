import torch
import torch.nn as nn
import torch.nn.functional as F


class RidgeRegression(nn.Module):
    def __init__(self, lam, kernel, device=None):
        super(RidgeRegression, self).__init__()
        self.lam = lam
        self.kernel = kernel
        self.alphas = None
        self.X_tr = None
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def fit(self, X, Y):
        if len(X.size()) == 3:
            b, n, d = X.size()
            b, m, l = Y.size()
        elif len(X.size()) == 2:
            n, d = X.size()
            m, l = Y.size()
        assert (
            n == m
        ), "Tensors need to have same dimension, dimensions are {} and {}".format(n, m)

        self.K = self.kernel(X, X)
        K_nl = self.K + self.lam * n * torch.eye(n).to(self.device)
        # To use solve we need to make sure Y is a float
        # and not an int
        self.alphas, _ = torch.solve(Y.float(), K_nl)
        self.X_tr = X

    def predict(self, X):
        return torch.matmul(self.kernel(X, self.X_tr), self.alphas)


class LearnedBiasRidgeRegression(nn.Module):
    def __init__(self, d, log_lam, device=None):
        super(LearnedBiasRidgeRegression, self).__init__()
        self.log_lam = nn.Parameter(torch.tensor(log_lam))
        self.bias = nn.Parameter(torch.tensor(torch.zeros(d)).reshape(-1, 1))
        self.w = None
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def fit(self, X, Y):
        if len(X.size()) == 3:
            b, n, d = X.size()
            b, m, l = Y.size()
        elif len(X.size()) == 2:
            n, d = X.size()
            m, l = Y.size()
        assert (
            n == m
        ), "Tensors need to have same dimension, dimensions are {} and {}".format(n, m)

        C = X.transpose(-2, -1).matmul(X) + n * torch.exp(self.log_lam) * torch.eye(
            d
        ).to(self.device)
        a = X.transpose(-2, -1).matmul(Y) + self.bias
        self.w, _ = torch.solve(a, C)

    def predict(self, X):
        return torch.matmul(X, self.w)


class FeatureMapRidgeRegression(nn.Module):
    """Like RidgeRegression but with an additional feature map phi: X \to Phi

    feature_map is a torch module which is learned together with the rest of the parameters

    TODO: Log is wrong since it can go negative, fix"""

    def __init__(self, lam, kernel, feature_map, normalize_features=False, device=None):
        super(FeatureMapRidgeRegression, self).__init__()
        self.lam = nn.Parameter(torch.tensor(lam))
        self.kernel = kernel
        self.feature_map = feature_map
        self.normalize_features = normalize_features
        self.alphas = None
        self.Phi_tr = None
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def fit(self, X, Y):
        Y = F.one_hot(Y)
        n = X.size()[0]

        # Normalize features
        Phi = self.feature_map(X)  # B x N x D
        if self.normalize_features:
            Phi = F.normalize(Phi, dim=-1)

        self.K = self.kernel(Phi, Phi)
        K_nl = self.K + self.lam * n * torch.eye(n).to(self.device)
        # To use solve we need to make sure nY is a float
        # and not an int
        self.alphas, _ = torch.solve(Y.float(), K_nl)
        self.Phi_tr = Phi

    def predict(self, X):
        return torch.matmul(self.kernel(self.feature_map(X), self.Phi_tr), self.alphas)
