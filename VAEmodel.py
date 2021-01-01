"""
Author: Navid Shervani-Tabar
"""
import torch

from torch import nn
from torch.autograd import Variable

from filter import scattering


class VAEmod(nn.Module):
    def __init__(self, args):
        super(VAEmod, self).__init__()

        # -- training parameters
        self.device = args.device

        # -- graph parameters
        self.n_max_atom = args.n_node
        self.n_type_bond = args.n_bond_type
        self.n_atom_features = args.n_atom_type
        self.n_scat_atom_features = args.n_scat_atom_features

        # -- scattering parameters
        self.scat = scattering(args).to(self.device)
        self.sdim = args.sdim

        # -- network parameters
        self.leaky = nn.LeakyReLU(0.01, inplace=False)
        self.relu = nn.ReLU()

        self.dim_interm = 8
        self.z_dim = args.z_dim
        scat_dim = self.sdim * self.n_scat_atom_features
        enc_dim = 400
        h_1_dim = 2 * self.z_dim
        h_2_dim = 4 * self.z_dim
        h_3_dim = 8 * self.z_dim
        h_4_dim = self.n_max_atom * self.n_type_bond * self.dim_interm
        h_6_dim = self.n_max_atom * self.n_atom_features

        # -- encoder
        self.bn_1     = nn.BatchNorm1d(scat_dim)
        self.enc_fc_1 = nn.Linear(scat_dim, enc_dim)
        self.bn_2     = nn.BatchNorm1d(enc_dim)
        self.enc_fc_2 = nn.Linear(enc_dim, self.z_dim)
        self.enc_fc_3 = nn.Linear(enc_dim, self.z_dim)

        # -- weight network
        if bool(args.y_target):
            self.dec_fc_1 = nn.Linear(self.z_dim + 3, h_1_dim)
        else:
            self.dec_fc_1 = nn.Linear(self.z_dim, h_1_dim)
        self.dec_fc_2 = nn.Linear(h_1_dim, h_2_dim)
        self.dec_fc_3 = nn.Linear(h_2_dim, h_3_dim)
        self.dec_fc_4 = nn.Linear(h_3_dim, h_4_dim)

        # -- signal network
        self.SM = nn.Softmax(dim=3)
        if bool(args.y_target):
            self.dec_fc_5 = nn.Linear(self.n_max_atom * self.n_type_bond * self.n_max_atom + self.z_dim + 3, h_6_dim)
        else:
            self.dec_fc_5 = nn.Linear(self.n_max_atom * self.n_type_bond * self.n_max_atom + self.z_dim, h_6_dim)

    def encode(self, x):

        h_1 = self.bn_1(x)
        h_2 = self.relu(self.bn_2(self.enc_fc_1(h_1)))

        return self.enc_fc_2(h_2), self.enc_fc_3(h_2)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode_W(self, z):

        # -- adjacency network (shared)
        h_1 = self.leaky((self.dec_fc_1(z)))
        h_2 = self.leaky((self.dec_fc_2(h_1)))
        h_3 = self.leaky((self.dec_fc_3(h_2)))
        h_4 = self.leaky((self.dec_fc_4(h_3)))
        h_4 = h_4.view(-1, self.n_max_atom, self.n_type_bond, self.dim_interm)
        h_4 = self.leaky(torch.matmul(h_4.permute(0, 2, 1, 3), h_4.permute(0, 2, 3, 1)))
        h_4 = h_4.permute(0, 2, 3, 1)

        return h_4

    def decode_f(self, z, W):

        W = self.SM(W)

        # -- node network
        h_5 = W.reshape(-1, self.n_max_atom * self.n_max_atom * self.n_type_bond)
        h_5 = torch.cat((z, h_5), dim=1)
        h_5 = self.leaky((self.dec_fc_5(h_5)))
        h_5 = h_5.view(-1, self.n_max_atom, self.n_atom_features)

        return h_5

    def decode(self, z):

        W = self.decode_W(z)
        f = self.decode_f(z, W)

        return [f, W]

    def forward(self, signal, adjacency, props):

        signal_in = torch.transpose(signal.reshape(-1, self.n_max_atom, self.n_atom_features), 2, 1)

        if props is not None:
            signal_in = torch.cat((signal_in, props.unsqueeze(2).repeat(1, 1, 9)), dim=1)
        mu, logvar = self.encode(self.scat(adjacency, signal_in).reshape(-1, self.sdim * self.n_scat_atom_features))
        z = self.reparameterize(mu, logvar)

        # -- for constraint regularization
        z_prior = self.reparameterize(torch.zeros(mu.size(), device=self.device), torch.zeros(mu.size(), device=self.device))

        if props is not None:
            return self.decode(torch.cat((z, props), dim=1)), mu, logvar, self.decode(torch.cat((z_prior, props), dim=1))
        else:
            return self.decode(z), mu, logvar, self.decode(z_prior)
