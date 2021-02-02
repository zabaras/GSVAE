"""
Author: Navid Shervani-Tabar
"""
import os
import torch
import pickle
import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from rdkit import Chem
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
from rdkit.Chem import Descriptors

class kernel:
    def __init__(self, K, R, d, M, gamma, omega=None):
        self.R = float(R)
        self.M = M
        self.K = K
        self.d = d
        self.gamma = gamma
        self.relu = nn.ReLU()

        if omega is not None:
            self.omega = omega
            self.gamma = self.omega(self.gamma.float())

        # -- Half-Cosine kernel
        self.tmp = self.R * self.gamma / (self.M - self.R + 1)
        self.g_U = lambda lamb: sum(
            [self.d[k] * torch.cos(2 * np.pi * (lamb / self.tmp - 0.5) * k) for k in range(self.K + 1)]) * (
                                lamb >= 0) * (lamb <= self.tmp)

    def uniform_translates(self, lamb, m):
        """
            constructs filter-bank of uniform translates.
        :param lamb: eigenvalue (analogue of frequency).
        :param m: number of filters in the filter bank.
        :return: filter response to input eigenvalues.
        """

        return self.g_U(lamb - self.tmp / self.R * (m - self.R + 1))

    def warped_filter(self, lamb, m):
        """
            filter-bank of warped kernels.
        :param lamb: eigenvalue (analogue of frequency).
        :param m: number of filters in the filter bank.
        :return: filter response to input eigenvalues.
        """

        return self.uniform_translates(self.omega(lamb), m)


class scattering(nn.Module):
    def __init__(self, args):
        super(scattering, self).__init__()

        # -- training parameters
        self.device = args.device
        self.data_dir = args.data_dir
        self.N = args.N

        # -- graph parameters
        self.n_node = args.n_node
        self.n_atom_features = args.n_scat_atom_features

        # -- scattering parameters
        self.Ns = args.wlt_scales
        self.Nl = args.scat_layers
        self.spectrum, _ = torch.sort(self.compute_spectrum(self.load_data()).reshape(-1))
        self.cdf_warp = self.warp_func

    def load_data(self):
        """
            Loads training data
        :return: tensor of adjacency matrices of graphs in the training set.
        """

        with open(self.data_dir, 'rb') as f:
            smiles = pickle.load(f)[:self.N]
            signal = torch.Tensor(pickle.load(f)[:self.N])
            adjacency = torch.Tensor(pickle.load(f)[:self.N])

        return adjacency

    def compute_spectrum(self, W):
        """
            Computes eigenvalues of normalized graph Laplacian.
        :param W: tensor of graph adjacency matrices.
        :return: eigenvalues of normalized graph Laplacian
        """

        # -- computing Laplacian
        L = torch.diag_embed(W.sum(1)) - W

        # -- normalize
        if True:
            diag = W.sum(1)
            dhalf = torch.diag_embed(1. / torch.sqrt(torch.max(torch.ones(diag.size()), diag)))
            L = dhalf.matmul(L).matmul(dhalf)

        # -- eig decomposition
        self.E, self.V = torch.symeig(L, eigenvectors=True)

        return self.E

    @property
    def warp_func(self):
        """
            Construct warping function based on empirical spectral cumulative distribution.
        :return: CDF based warping function
        """

        self.gamma = self.spectrum.max()

        # -- estimate CDF
        cdf_ = torch.arange(0, len(self.spectrum)) / (len(self.spectrum) - 1.)
        intvl = int(len(self.spectrum)/5-1)

        if True:
            return interp1d(self.spectrum[0::intvl], cdf_[0::intvl], fill_value='extrapolate')
        else:
            return interp1d(self.spectrum, cdf_, fill_value='extrapolate')

    def get_kernel(self):
        """
            Compute adapted graph wavelets and scaling kernels
        :return: a new instance of "kernel" class
        """

        # -- kernel properties
        K = 1
        d = torch.tensor([0.5, 0.5])
        R = 3

        # -- define warping function
        omega = lambda lamb: torch.tensor(self.cdf_warp(lamb.numpy()))

        return kernel(K, R, d, self.Ns, self.gamma, omega)

    def compute_frame(self):
        """
            Compute filter-bank frame for spectral filters
        :return: a collection of frames of each wavelet kernel in the filter-bank.
        """

        my_kernel = self.get_kernel()
        VT = torch.transpose(torch.Tensor(self.V), 2, 1)

        # -- wavelet filter-bank frame
        fb_frame = torch.empty(self.V.shape[0], 0, self.n_node, self.n_node)
        for j in range(self.Ns):

            # -- kernel frame
            filtering_matrix = torch.diag_embed(my_kernel.warped_filter(self.E, j+1))
            kernel_frame = self.V.matmul(filtering_matrix).matmul(VT).unsqueeze(1)

            fb_frame = torch.cat((fb_frame, kernel_frame), dim=1)

        return fb_frame

    def filters(self, W):
        """
            Compute low-pass (average pooling) and band-pass (wavelet kernels) filtering matrices.
        :param W: tensor of graph adjacency matrices.
        :return: average pooling filter and filtering matrix of wavelets filter-bank.
        """

        # -- eig decomposition
        self.compute_spectrum(W.cpu())

        # -- average pooling operator
        mu = (1 / W.shape[1]) * torch.ones(W.shape[0], W.shape[1], device=self.device)

        # -- wavelet filters
        gHat = self.compute_frame()

        return mu, gHat.to(self.device)

    def forward(self, W, f):
        """
            Perform wavelet scattering transform
        :param W: tensor of graph adjacency matrices.
        :param f: tensor of graph signal vectors.
        :return: wavelet scattering coefficients
        """

        mu, gHat = self.filters(W)
        U = f.unsqueeze(1)  # U_0

        # -- first layer
        S = torch.bmm(f, mu.unsqueeze(2))  # S_0

        # -- deeper layers
        mu = mu.unsqueeze(1).unsqueeze(3).repeat(1, self.Ns, 1, 1)

        for l in range(1, self.Nl):
            rho_gHat_f = torch.empty([f.shape[0], 0, self.n_atom_features, self.n_node]).to(self.device)

            for j in range(self.Ns ** (l - 1)):
                fj = U[:, j, :, :].unsqueeze(1)
                gHat_fj = fj @ gHat

                # -- non-linearity
                rho_gHat_fj = torch.abs(gHat_fj)
                rho_gHat_f = torch.cat((rho_gHat_f, rho_gHat_fj), dim=1)

                # -- average pooling
                S_j = torch.transpose((rho_gHat_fj @ mu).squeeze(3), 2, 1)

                # -- S_{m-1}
                S = torch.cat((S, S_j), dim=2)

            # -- U_m
            U = rho_gHat_f.clone()

        return S

def parse_args():
    desc = "Visualization of adaptive graph wavelet filters."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gpu_mode', type=int, default=1, help='Accelerate the script using GPU.')
    parser.add_argument('--wlt_scales', type=int, default=12, help='Number of filters in the spectral domain.')
    parser.add_argument('--scat_layers', type=int, default=4, help='Number of layers in the scattering network.')
    parser.add_argument('--N', type=int, default=600, help='Number of training data.')
    parser.add_argument('--database', type=str, default='QM9', help='Training database name.')

    args = parser.parse_args()

    args.loadtrainedmodel = args.n_samples = 0

    # -- scattering
    args.sdim = 0
    for l in range(args.scat_layers):
        args.sdim += args.wlt_scales ** l

    # -- storage settings
    dir_ = os.getcwd()
    res_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.res_dir = os.path.join(dir_, 'results/', res_name)
    os.makedirs(args.res_dir)

    # -- dataset specification
    args.atom_dict = {0: 'C', 1: 'O', 2: 'N', 3: 'F', 4: 'H'}
    args.n_node = 9
    args.n_atom_type = 5

    args.n_scat_atom_features = args.n_atom_type

    args.data_dir = os.path.join(dir_, 'data/' + args.database + '_0.data')
    args.n_bond_type = 4

    # -- GPU settings
    args.device = torch.device('cuda' if (bool(args.gpu_mode) and torch.cuda.is_available()) else 'cpu')
    if bool(args.gpu_mode) and not torch.cuda.is_available():
        print('No GPUs on this device! Using CPU instead.')

    return args

def main():
    """
        perform scattering transform
    """

    # -- initialize
    args = parse_args()
    scat = scattering(args)
    func = scat.warp_func
    x = torch.linspace(0, scat.gamma, 200)

    # -- plot empirical cdf
    plt.figure(1)
    f, ax = plt.subplots()
    ax.set_ylim([-.01, 1.01])
    ax.set_xlim([-.01, 2.005])
    ax.grid(ls='dashed')
    ax.set_axisbelow(True)
    plt.plot(scat.spectrum, torch.arange(0,len(scat.spectrum))/(len(scat.spectrum)-1.), label='Empircal CDF')
    plt.plot(x, func(x), label='Smooth apprx CDF')
    plt.scatter(scat.spectrum, torch.zeros(scat.spectrum.shape), s=2, c='r', marker='.')
    plt.legend()
    plt.title('Empirical CDF')
    plt.savefig(args.res_dir + '/empirical_cdf', bbox_inches='tight')
    plt.close()

    # -- plot histogram
    plt.figure(1)
    f, ax = plt.subplots()
    ax.set_xlim([-.01, 2.005])
    plt.hist(scat.spectrum, bins=30, rwidth=0.8)
    ax.grid(ls='dashed')
    ax.set_axisbelow(True)
    plt.title('Spectral histogram')
    plt.savefig(args.res_dir + '/spectra_hist', bbox_inches='tight')
    plt.close

    # -- uniform translates
    plt.figure(1)
    my_kernel = kernel(1, 3, torch.tensor([0.5, 0.5]), args.wlt_scales, 2)
    for j in range(args.wlt_scales):
        psi = my_kernel.uniform_translates(x, j)
        plt.plot(x, psi)
    plt.title('Uniform translate filters')
    plt.savefig(args.res_dir + '/uniform_translates', bbox_inches='tight')
    plt.close()

    # -- adaptive filter-bank
    plt.figure(1)
    f, ax = plt.subplots()
    ax.set_ylim([-.01, 1.01])
    ax.set_xlim([-.01, 2.005])
    my_kernel = scat.get_kernel()
    for j in range(args.wlt_scales):
        psi = my_kernel.warped_filter(x, j)
        plt.plot(x, psi)
    plt.title('Adaptive filter-bank')
    plt.savefig(args.res_dir + '/adaptive_filter_bank', bbox_inches='tight')
    plt.close()

    # -- perform scattering transform
    N = 5000
    prp = {}
    with open(args.data_dir, 'rb') as f:
        smiles = pickle.load(f)[:N]
        signal = torch.Tensor(pickle.load(f)[:N])
        adjacency = torch.Tensor(pickle.load(f)[:N])
        for i in range(3):
            prp[str(i)] = torch.Tensor(pickle.load(f)[:N])

    signal_in = torch.transpose(signal.reshape(-1, args.n_node, args.n_scat_atom_features), 2, 1)
    scat_out = scat(adjacency.to(args.device), signal_in.to(args.device)).reshape(-1, args.sdim * args.n_scat_atom_features)

    # -- store coefficients
    with open(args.res_dir + '/scat_out.data', 'wb') as f:
        pickle.dump(smiles, f)
        pickle.dump(signal, f)
        pickle.dump(adjacency, f)
        for i in range(3):
            pickle.dump(prp[str(i)], f)
        pickle.dump(scat_out, f)

    props = {'Rings': [], 'SP3': [], 'PSA': [], 'MolWt': [], 'HBA':[]}
    for idx, sml in enumerate(smiles):
        mol = Chem.MolFromSmiles(sml)
        props['Rings'].append(Chem.rdMolDescriptors.CalcNumRings(mol))
        props['SP3'].append(Chem.rdMolDescriptors.CalcFractionCSP3(mol))
        props['PSA'].append(Descriptors.TPSA(mol))
        props['MolWt'].append(Descriptors.MolWt(mol))
        props['HBA'].append(Chem.rdMolDescriptors.CalcNumHBA(mol))

    # -- plot scattering latent space
    pca = PCA(n_components=2)
    latent_scat = pca.fit_transform(scat_out.cpu().detach().numpy())

    for idx, prp_ in enumerate(list(props.keys())):
        plt.figure(i)
        f, ax = plt.subplots()
        plt.gca().set_aspect('equal', adjustable='box')
        ax.grid(ls='dashed')
        ax.set_axisbelow(True)
        if prp_ == 'Rings':
            plt.scatter(latent_scat[:,0], latent_scat[:,1], c=props[prp_], s=5, vmin=0, vmax=4)
        else:
            plt.scatter(latent_scat[:,0], latent_scat[:,1], c=props[prp_], s=5)
        plt.title(prp_)
        plt.savefig(args.res_dir + '/latent_scat_' + prp_, bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    main()
