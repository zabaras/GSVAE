"""
Author: Navid Shervani-Tabar
"""
import os
import sys
import torch
import argparse
import datetime
import time

from VAEtrain import VAEgraph
from utils import chemf


def parse_args():
    desc = "PyTorch implementation of Molecular Graph Latent Space Discovery with adaptive wavelet graph Scattering VAE."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--epochs', type=int, default=800, help='The number of epochs to run.')
    parser.add_argument('--batch_size', type=int, default=32, help='The size of each batch.')
    parser.add_argument('--gpu_mode', type=int, default=1, help='Accelerate the script using GPU.')
    parser.add_argument('--z_dim', type=int, default=30, help='Latent space dimensionality')
    parser.add_argument('--seed', type=int, default=1400, help='Random seed.')
    parser.add_argument('--loadtrainedmodel', type=str, default='', help='Path to a trained model.')
    parser.add_argument('--mu_reg_1', type=float, default=0., help='Regularization parameter for ghost nodes and valence constraint.')
    parser.add_argument('--mu_reg_2', type=float, default=0., help='Regularization parameter for connectivity constraint.')
    parser.add_argument('--N_vis', type=int, default=3000, help='Number of test data for visualization.')
    parser.add_argument('--log_interval', type=int, default=200, help='Number of epochs between visualizations.')
    parser.add_argument('--n_samples', type=int, default=10000, help='Number of generated samples from molecular space.')
    parser.add_argument('--wlt_scales', type=int, default=8, help='Number of filters in the spectral domain.')
    parser.add_argument('--scat_layers', type=int, default=3, help='Number of layers in the scattering network.')
    parser.add_argument('--database', type=str, default='QM9', help='Name of the training database.')
    parser.add_argument('--datafile', type=str, default='', help='Name of the training file.')
    parser.add_argument('--BB_samples', type=int, default=0, help='Index for Bayesian bootstrap sample.')
    parser.add_argument('--N', type=int, default=600, help='Number of training data.')
    parser.add_argument('--res', type=str, default='results/', help='Path for storing the results.')

    args = parser.parse_args()

    # -- scattering
    args.sdim = 0
    for l in range(args.scat_layers):
        args.sdim += args.wlt_scales ** l

    # -- storage settings
    dir = os.getcwd()
    res_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.res_dir = os.path.join(dir, args.res, res_name)
    os.makedirs(args.res_dir)

    args.vis = bool(args.log_interval)

    # -- dataset specification
    if args.database == 'QM9':
        args.atom_dict = {0: 'C', 1: 'O', 2: 'N', 3: 'F', 4: 'H'}
        args.n_node = 9
        args.n_atom_type = 5

    if args.datafile == '':
        args.data_dir = os.path.join(dir, 'data/' + args.database + '_0.data')
    else:
        args.data_dir = os.path.join(dir, 'data/' + args.datafile)

    args.n_bond_type = 4

    # -- GPU settings
    args.device = torch.device('cuda' if (bool(args.gpu_mode) and torch.cuda.is_available()) else 'cpu')

    return check_args(args)

def check_args(args):
    """
        Check input arguments
    :param args: input arguments
    :return: input arguments
    """

    if args.batch_size < 1:
        sys.tracebacklimit = 0
        raise ValueError('Number of epochs must be larger than or equal to one.')

    if args.batch_size < 1:
        sys.tracebacklimit = 0
        raise ValueError('Batch size must be larger than or equal to one.')

    if args.wlt_scales <= 2:
        sys.tracebacklimit = 0
        raise ValueError('The number of wavelet filters must be higher than 2.')

    if not os.path.isfile(args.data_dir):
        sys.tracebacklimit = 0
        raise OSError('Training data not available. Run data_gen.py at ./data.')

    if bool(args.gpu_mode) and not torch.cuda.is_available():
        print('No GPUs on this device! Running on CPU.')

    return args

def weights_init(m):
    """
        Initialize model parameters for NN layers.
    :param m: module
    :return: initializes weights and biases for linear and batch-normalization layers in the model.
    """

    classname = m.__class__.__name__
    if classname.find('Linear') != -1:

        # -- weights
        init_range = torch.sqrt(torch.tensor(6.0 / (m.in_features + m.out_features)))
        m.weight.data.uniform_(-init_range, init_range)

        # -- bias
        if m.bias is not None:
            m.bias.data.uniform_(-init_range, init_range)

    if classname.find('BatchNorm1d') != -1:

        # -- weights
        m.weight.data.fill_(1)

        # -- bias
        if m.bias is not None:
            m.bias.data.zero_()

def main():
    args = parse_args()
    if args is None:
        exit()

    chem = chemf(args)

    # -- train model
    model = VAEgraph(args)
    model.model.apply(weights_init)

    print('device_count()', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print('get_device_name', torch.cuda.get_device_name(i))

    # -- assign training weights
    weight_posterior = torch.distributions.dirichlet.Dirichlet(torch.tensor([1.] * args.N))
    training_weights = torch.ones(args.N) / args.N
    for i in range(args.BB_samples):
        training_weights = weight_posterior.sample()

    # -- MWLE
    t = time.time()
    model.train(training_weights)
    print('elapsed:', time.time() - t)

    # -- sampling from trained model
    sig, adj, sample_z = model.get_samples(sample_name = '/samples_'+str(args.BB_samples)+'.data')
    if not bool(args.BB_samples):
        mols = chem.MolFromSample(sig, adj)
        chem.draw(mols)
        valid_mol, valid_z = chem.QualityMetrics(mols, sample_z, verbose=True)

        # -- plot property maps
        chem.LatentMap()
        chem.ChemSpace(valid_mol)

if __name__ == '__main__':
    main()
