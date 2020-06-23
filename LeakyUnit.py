import torch
import torch.nn as nn
import torch.nn.functional as F


class LeakyUnit(nn.Module):
    def __init__(self, n_features):
        super(LeakyUnit, self).__init__()
        self.W_r = nn.Conv2d(2*n_features, n_features, kernel_size=3, padding=1, stride=1, bias=False)
        self.W = nn.Conv2d(n_features, n_features, kernel_size=3, padding=1, stride=1, bias=False)
        self.U = nn.Conv2d(n_features, n_features, kernel_size=3, padding=1, stride=1, bias=False)
        self.W_z = nn.Conv2d(2*n_features, n_features, kernel_size=3, padding=1, stride=1, bias=False)
        self.sigma = nn.Sigmoid()

    def forward(self, f_m, f_n):
        r_mn = self.sigma(self.W_r(torch.cat((f_m, f_n), dim=1)))
        f_mn_hat = torch.tanh(self.U(f_m) + self.W(r_mn * f_n))
        z_mn = self.sigma(self.W_z(torch.cat((f_m, f_n), dim=1)))
        f_m_out = z_mn * f_m + (1 - z_mn) * f_mn_hat
        # f_n_out = (1 - r_mn) * f_n

        return f_m_out, r_mn, z_mn
