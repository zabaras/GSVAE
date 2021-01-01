"""
Author: Navid Shervani-Tabar
"""
import os
import sys
import torch
import pickle
import datetime
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from itertools import product
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
from rdkit import Chem, DataStructs, RDLogger
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import Matern
from rdkit.Chem import AllChem, Descriptors, Crippen, Draw
from sklearn.gaussian_process import GaussianProcessRegressor

from filter import scattering

RDLogger.DisableLog('rdApp.*')


class MolecularGraphDataset(Dataset):
    def __init__(self, dataset_name, size, idx_0):
        with open(dataset_name, 'rb') as f:
            self.smiles = pickle.load(f)[idx_0:idx_0 + size]
            self.sig = torch.Tensor(pickle.load(f))[idx_0:idx_0 + size]
            self.adj = torch.Tensor(pickle.load(f))[idx_0:idx_0 + size]

            prp_1 = torch.Tensor(pickle.load(f))[idx_0:idx_0 + size]
            prp_2 = torch.Tensor(pickle.load(f))[idx_0:idx_0 + size]
            prp_3 = torch.Tensor(pickle.load(f))[idx_0:idx_0 + size]

        props = np.vstack((prp_1, prp_2, prp_3)).T

        # -- normalize data
        scaler = StandardScaler()
        scaler.fit(props)
        self.props = torch.Tensor(scaler.transform(props))
        self.mean = scaler.mean_
        self.var = scaler.var_

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        sample = {'smiles': self.smiles[idx], 'signal': self.sig[idx], 'adjacency': self.adj[idx],
                  'properties': self.props[idx]}

        return sample


class tools:
    def __init__(self, args):

        # -- training parameters
        self.device = args.device
        self.reg_flag = [bool(i != 0) for i in args.reg_vec]

        # -- graph parameters
        self.n_max_atom = args.n_node
        self.n_type_bond = args.n_bond_type
        self.n_atom_features = args.n_atom_type

        # -- scattering parameters
        self.scat = scattering(args)
        self.z_dim = args.z_dim
        self.sdim = args.sdim
        self.data_dir = args.data_dir
        self.cond_dsgn = bool(args.y_target)
        self.n_scat_atom_features = args.n_scat_atom_features

        self.res_dir = args.res_dir
        self.chem = chemf(args)
        self.epochs = args.epochs

    def visLatent(self, VisulData, model, epoch, TrainData=None, EndPts=None):
        """
            Visualize the latent space.
        :param VisulData: visualizing dataset
        :param model: trained model
        :param epoch: current epoch
        :param TrainData: training dataset
        :param EndPts: 2 molecules from visualizaton dataset which are used for interpolation
        :return: stores plots of latent space colored by different properties.
        """

        # -- extract visualization data
        N_vis = len(VisulData.dataset)
        signal = VisulData.dataset[:]['signal'].to(self.device)
        adjacency = VisulData.dataset[:]['adjacency'].to(self.device)
        properties = VisulData.dataset[:]['properties'].to(self.device)

        my_properties = {'Rings': [], 'SP3': [], 'PSA': [], 'MolWt': []}
        for smiles in VisulData.dataset[:]['smiles']:
            mol = Chem.MolFromSmiles(smiles)
            my_properties['Rings'].append(Chem.rdMolDescriptors.CalcNumRings(mol))
            my_properties['SP3'].append(Chem.rdMolDescriptors.CalcFractionCSP3(mol))
            my_properties['PSA'].append(Descriptors.TPSA(mol))
            my_properties['MolWt'].append(Descriptors.MolWt(mol))

        steps = 10

        # -- concat training data
        if TrainData is not None:
            N_train = len(TrainData.dataset)
            signal = torch.cat((signal, TrainData.dataset[:]['signal'].to(self.device)))
            adjacency = torch.cat((adjacency, TrainData.dataset[:]['adjacency'].to(self.device)))

        # -- encode inputs
        signal = signal.reshape(-1, self.n_max_atom, self.n_atom_features)
        signal_in = torch.transpose(signal, 2, 1)
        if self.cond_dsgn:
            signal_in = torch.cat((signal_in, properties.unsqueeze(2).repeat(1, 1, 9)), dim=1)
        mu, logvar = model.encode(self.scat(adjacency, signal_in).reshape(-1, self.sdim * self.n_scat_atom_features))

        # -- project latent space to 2D
        pca = PCA(n_components=2)
        pca.fit(mu.cpu().detach().numpy())
        latent_2D = pca.transform(mu.cpu().detach().numpy())

        # -- plot settings
        plt.figure(1)
        f, ax = plt.subplots()
        plt.gca().set_aspect('equal', adjustable='box')
        ax.set_ylim([-4, 4])
        ax.set_xlim([-4, 4])
        ax.grid(ls='dashed')
        ax.set_axisbelow(True)

        # -- plot 2D latent for epochs
        plt.scatter(latent_2D[:N_vis, 0], latent_2D[:N_vis, 1], s=5, c=np.array(my_properties['MolWt']))
        f.savefig(self.res_dir + '/latent_' + str(epoch), bbox_inches='tight')

        # -- plot final 2D latent with training data
        if TrainData is not None:
            res = plt.scatter(latent_2D[N_vis:(N_vis + N_train), 0], latent_2D[N_vis:(N_vis + N_train), 1], s=5,
                              marker='*', c='m', alpha=0.3)
            f.savefig(self.res_dir + '/latent_train', bbox_inches='tight')
            res.remove()

        plt.close()

        # -- plot final 2D latent with each property
        if epoch == self.epochs:
            for i, prp in enumerate(list(my_properties.keys())):
                # -- plot settings
                plt.figure(i + 2)
                f, ax = plt.subplots()
                plt.gca().set_aspect('equal', adjustable='box')
                ax.set_ylim([-4, 4])
                ax.set_xlim([-4, 4])
                ax.grid(ls='dashed')
                ax.set_axisbelow(True)

                # -- plot final 2D
                if prp == 'Rings':
                    plt.scatter(latent_2D[:N_vis, 0], latent_2D[:N_vis, 1], s=5, c=np.array(my_properties[prp][:N_vis]), vmin=0, vmax=4)
                else:
                    plt.scatter(latent_2D[:N_vis, 0], latent_2D[:N_vis, 1], s=5, c=np.array(my_properties[prp][:N_vis]))
                f.savefig(self.res_dir + '/latent_' + prp, bbox_inches='tight')

                plt.close()

    def pltIter(self, name, stat, ylim=None):
        """
            help function to plot statistics vs iteration.
        :param name: name of the image for saving
        :param stat: statistic data to be plotted
        :param ylim: y-axis limits of the plot (stat range)
        :return: stores statistic vs iteration plots
        """

        plt.figure(1)
        f, ax = plt.subplots()
        ax.plot(range(len(stat)), stat)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.grid(True)
        f.tight_layout()
        f.savefig(self.res_dir + name, bbox_inches='tight')
        plt.close()

    def pltLoss(self, train_hist, idx):
        """
            plot loss statistics vs iteration.
        :param train_hist: dictionary of loss statistic data to be plotted
        :param idx: iteration inedx for naming stored file
        :return: stores statistic vs iteration plots through "pltIter" help function
        """

        self.pltIter('/loss_tot_' + str(idx), train_hist['Tl'], [-0.5, 150])
        self.pltIter('/loss_rec_' + str(idx), train_hist['RC'])
        self.pltIter('/loss_kld_' + str(idx), train_hist['KL'])
        if self.reg_flag[0]:
            self.pltIter('/loss_rg1_' + str(idx), train_hist['R1'], [-0.5, 10])
        if self.reg_flag[1]:
            self.pltIter('/loss_rg2_' + str(idx), train_hist['R2'], [-0.5, 10])
        if self.reg_flag[2]:
            self.pltIter('/loss_rg3_' + str(idx), train_hist['R3'], [-0.5, 10])
        if self.reg_flag[3]:
            self.pltIter('/loss_rg4_' + str(idx), train_hist['R4'], [-0.5, 10])

        self.pltIter('/l_ELBO' + str(idx), - (np.array(train_hist['RC']) + np.array(train_hist['KL'])))

    def HistStat(self, mol_lis, n_bins=15):
        """
            Computes histogram statistics for samples.
        :param mol_lis: list of sample sets
        :return: dictionary of set of probability vectors, range of bins, and range of values for different properties
        """

        bins_all = []
        prob_all = {}
        for mols in mol_lis:

            # -- compute physicochemical properties
            props, bounds = self.chem.ChemProperty(mols)
            for i, (prop, bound) in enumerate(zip(props, bounds)):

                # -- plot histogram for each set of samples
                weights = np.ones_like(prop) / float(len(prop))
                my_plot = plt.figure(12)
                prob, bins, _ = plt.hist(prop, n_bins, bound, density=False, histtype='step', color='red', weights=weights)
                plt.close(my_plot)

                # -- concatenate bin probabilities of each property for different sample sets
                try:
                    prob_all[str(i)] = np.concatenate(([prob_all[str(i)], prob.reshape(-1, 1)]), axis=1)
                except:
                    prob_all[str(i)] = prob.reshape(-1, 1)
                    bins_all.append(bins)

        return prob_all, bins_all, bounds

    def ErrorBars(self, quants, bins_all, fill_c, fill_hatch=None, label_=None):
        """
            Plot error bars over property distribution.
        :param quants: computed quantiles of the data, including 50%, upper, and lower quantiles.
        :param bins_all: a vector of start and end points of all the bins.
        :param fill_c: color for error bar's shaded region.
        :param fill_hatch: hatch type for error bar's shaded region.
        :return: plots upper, lower, and 50 quantiles and shades the region in between.
        """

        # -- shade between quantiles
        for j in range(quants.shape[1]):
            x = np.arange(bins_all[j], bins_all[j + 1], 0.0001)
            y1 = quants[0, j]
            y2 = quants[1, j]

            if j>0:
                label_ = None
            plt.fill_between(x, y1, y2, facecolor=fill_c, alpha=0.2, hatch=fill_hatch, label=label_)

        # -- plot quantiles
        self.pltHist(bins_all, quants[0], clr='gray', alph=.3, lw=1.)
        self.pltHist(bins_all, quants[1], clr='gray', alph=.3, lw=1.)

    def pltHist(self, bins, prob, clr, alph, lw, fmt='-'):
        """
            plot histogram from bins and probabilities
        :param bins: start and end points of histogram bins.
        :param prob: vector of bin probabilities.
        :param clr: color of plot line.
        :param alph: transparency of plot line.
        :param lw: width of plot line.
        :return: plots histogram (neither show() or savefig())
        """

        n = len(prob)
        for j in range(n):
            plt.plot([bins[j], bins[j + 1]], [prob[j], prob[j]], fmt, color=clr, linewidth=lw, alpha=alph)

        plt.plot([bins[0], bins[0]], [0., prob[0]], fmt, color=clr, linewidth=lw, alpha=alph)
        plt.plot([bins[n], bins[n]], [prob[n - 1], 0.], fmt, color=clr, linewidth=lw, alpha=alph)
        for j in range(n - 1):
            plt.plot([bins[j + 1], bins[j + 1]], [prob[j], prob[j + 1]], fmt, color=clr, linewidth=lw, alpha=alph)


class chemf:
    def __init__(self, args):

        # -- training parameters
        self.device = args.device
        self.n_samples = args.n_samples
        self.N = args.N
        self.res_dir = args.res_dir
        self.database = args.database
        self.atom_dict = args.atom_dict
        self.data_dir = args.data_dir
        self.LoadData()
        self.loadmodel = bool(args.loadtrainedmodel)
        self.sdim = args.sdim

        # -- graph parameters
        self.n_max_atom = args.n_node
        self.n_type_bond = args.n_bond_type
        self.n_atom_features = args.n_atom_type
        self.scat = scattering(args)

        # -- model
        if self.loadmodel:
            self.filemodel = args.loadtrainedmodel
        else:
            self.filemodel = args.res_dir

    def LoadData(self):
        """
            Loads the training data. Used by QualityMetrics to compute the novelty of the generated samples.
        :return: Stores the SMILES representation from training set as an instance variable.
        """
        with open(self.data_dir, 'rb') as f:
            self.train_data = pickle.load(f)[:self.N]
            self.sig = torch.Tensor(pickle.load(f))[:self.N].to(self.device)
            self.adj = torch.Tensor(pickle.load(f))[:self.N].to(self.device)

    def ValidityFilters(self):
        """
            Defines validity filters for filtering mols.
        :return: connectivity, valency, and combined validity filters
        """
        valid_con = lambda x: Chem.MolToSmiles(x).count('[HH]') == Chem.MolToSmiles(x).count('.')
        valid_val = lambda x: Chem.MolFromSmiles(
            Chem.MolToSmiles(x)) is not None and not '[H]' in Chem.MolToSmiles(x)
        valid_all = lambda x: valid_val(x) and valid_con(x)

        return valid_con, valid_val, valid_all

    def ConstraintStat(self, adj):
        """
            Counts the number of 3-member cycles and cycles with triple bond
        :param adj: generated weighted adjacecny matrix
        :return: number of 3-member cycles and cycles with triple bond
        """

        # -- 3-member cycles
        with torch.no_grad():
            W = adj.clone().float()

        W[W == 2] = 1
        W[W == 3] = 1

        A_i = W.clone()
        for i in range(2):
            A_i = torch.bmm(W, A_i)

        res_1 = torch.einsum('bii->b', A_i) / 6.

        # -- cycles with triple bond
        with torch.no_grad():
            A = adj.clone()
            D = adj.clone()

        b_dim = A.shape[0]
        D[D == 1] = D[D == 2] = 0

        C = torch.empty(b_dim, self.n_max_atom, self.n_max_atom, device=self.device)
        nI = self.n_max_atom * torch.eye(self.n_max_atom, device=self.device).unsqueeze(0).repeat(b_dim, 1, 1)
        for i in range(self.n_max_atom):
            for j in range(i, self.n_max_atom):
                B = A.clone()
                B[:, i, j] = B[:, j, i] = 0
                C[:, i, j] = C[:, j, i] = torch.inverse(nI - B)[:, i, j]

        res_2 = torch.sum(torch.einsum('bij,bij->bij', D, C), (2, 1))

        return np.count_nonzero(res_1.cpu().numpy()), np.count_nonzero(res_2.cpu().numpy())

    def StructStat(self, dataset_smiles):
        """
            Computes the percentage of various functional groups in the data.
        :param dataset_smiles: list of SMILES representations of the data.
        :return:
        """

        FG = {
            'Acetylenic carbon': '[$([CX2]#C)]',
            'Aldehyde         ': '[CX3H1](=O)[#6]',
            'Alkyl carbon     ': '[CX4]',
            'Amide            ': '[NX3][CX3](=[OX1])[#6]',
            'Amino acid       ': '[NX3,NX4+][CX4H]([*])[CX3](=[OX1])[O,N]',
            'Carbonyl         ': '[$([CX3]=[OX1]),$([CX3+]-[OX1-])]',
            'Carboxylic acid  ': '[CX3](=O)[OX1H0-,OX2H1]',
            'Ester            ': '[#6][CX3](=O)[OX2H0][#6]',
            'Ether            ': '[OD2]([#6])[#6]',
            'Halide           ': '[#6][F,Cl,Br,I]',
            'Hydrazone        ': '[NX3][NX2]=[*]',
            'Hydroxyl         ': '[OX2H]',
            'Ketone           ': '[#6][CX3](=O)[#6]',
            'Nitrile          ': '[NX1]#[CX2]',
            'Primary amines   ': '[NH2,nH2]',
            'Secondary amines ': '[NH1,nH1]',
        }
        res = {}
        for group in list(FG.keys()):
            res[group] = []

        for sml in dataset_smiles:
            mol = Chem.MolFromSmiles(sml)

            for (group, SMARTS) in FG.items():
                res[group].append(int(len(mol.GetSubstructMatches(Chem.MolFromSmarts(SMARTS))) > 0))

        lines = ['Functional Group frequencies (percentage):\n']
        for (group, _) in FG.items():
            lines.append(group + ': {:.2f}'.format(np.array(res[group]).mean() * 100) + '\n')

        return lines

    def QualityMetrics(self, mols, z, adj, verbose=False):
        """
            Evaluates quality metrics for the generated molecules including validity, uniqueness, and novelty of the
            sample.
        :param mols: Mol objects of the generated molecules.
        :param z: Latent space representation of the generated molecules.
        :param verbose: Flag indicating whether or not to save and display the metrics.
        :return: Valid mol objects and the corresponding latent variable.
        """

        # -- define validity filters
        valid_con, valid_val, valid_all = self.ValidityFilters()

        assert (len(mols) == len(z))

        valid_mols = list(filter(valid_all, mols))
        valid_set = set(map(lambda x: Chem.MolToSmiles(x), valid_mols))

        # -- store quality metrics
        line1 = "Validity is {:.2%}: valency issue: {:.2%}, connectivity issue: {:.2%}. \n".format(
            np.array(list(map(valid_all, mols))).mean(), 1. - np.array(list(map(valid_val, mols))).mean(),
                                                         1. - np.array(list(map(valid_con, mols))).mean())
        line2 = "Uniqueness is {:.2%}. \n".format(0 if len(valid_mols) == 0 else len(valid_set) / len(valid_mols))
        line3 = "Novelty is {:.2%}. \n".format(
            np.array(list(map(lambda x: Chem.MolToSmiles(x) not in self.train_data, valid_mols))).mean())

        # -- physical constraints stats
        n_three_cycle, n_triple_cycle = self.ConstraintStat(adj)
        line4 = "Number of 3-member cycles is: {}. \n".format(n_three_cycle)
        line5 = "Number of cycles with triple bonds is: {}. \n".format(n_triple_cycle)

        if verbose:
            lines = self.StructStat(valid_set)
            print('\n', line1, line2, line3, line4, line5)
            print(*lines)
            file1 = open(self.res_dir + '/quality.txt', 'a')
            file1.writelines([line1, line2, line3, line4, line5])
            file1.writelines(lines)
            file1.close()

        # -- filter valid latent representations and sanitize
        valid_z = []
        for idx, mol in enumerate(mols):
            if valid_all(mol):
                Chem.rdmolops.SanitizeMol(mol)
                valid_z.append(z[idx])

        return valid_mols, torch.stack(valid_z)

    def MolFromSample(self, signal, adjacency):
        """
            Converts a batch of samples to RDKit's mol objects.
        :param signal: A batch of graph signals.
        :param adjacency: A batch of graph weight matrices.
        :return: A batch of RDKit's mol objects.
        """

        mols = []

        # -- convert graphs to mol objects
        for (f, W) in zip(signal, adjacency):
            atom_vector = []

            # -- construct atom vector from labels
            for atom in f:
                atom_vector.append(self.atom_dict[atom.item()])

            mol = self.MolFromGraph(atom_vector, W)
            mols.append(mol)

        return mols

    def MolFromGraph(self, atom_vector, weight_matrix):
        """
            Converts a molecular graph to RDKit's mol object.
        :param atom_vector: List of the atoms in the molecular graph.
        :param weight_matrix: Weighted ajacency matrix of the molecular graph.
        :return: RDKit's mol object.
        """

        # -- initiate RWMol object
        mol = Chem.RWMol()

        # -- add atoms
        node_idx = {}
        for idx, node in enumerate(atom_vector):
            node_idx[idx] = mol.AddAtom(Chem.Atom(node))

        # -- add bonds
        for i, row in enumerate(weight_matrix):
            for j, bond in enumerate(row):

                if j <= i:
                    continue

                if bond == 0:
                    continue
                elif bond == 1:
                    mol.AddBond(node_idx[i], node_idx[j], Chem.rdchem.BondType.SINGLE)
                elif bond == 2:
                    mol.AddBond(node_idx[i], node_idx[j], Chem.rdchem.BondType.DOUBLE)
                elif bond == 3:
                    mol.AddBond(node_idx[i], node_idx[j], Chem.rdchem.BondType.TRIPLE)
                else:
                    raise Exception('Bond type not supported!')

        return mol.GetMol()

    def draw(self, mols, name='/sample_', path=False):
        """
            Draw molecules on a grid based on CPK coloring convention.
        :param mols: mol objects of the molecules to be drawn.
        :param name: file name for storage.
        :param path: whether to draw a large batch of generated moles or a few mols on an interpolated path.
        :return: stores images of generated molecules.
        """

        n_molpage = 500
        if path == False:
            for i in range(int(self.n_samples / 500)):
                img = Draw.MolsToGridImage(mols[i * n_molpage:min((i + 1) * n_molpage, self.n_samples)], molsPerRow=5)
                img.save(self.res_dir + name + str(i) + '.png')
        else:
            img = Draw.MolsToGridImage(mols, molsPerRow=5)
            img.save(self.res_dir + name + '.png')

    def ChemProperty(self, mols):
        """
            Computes the physicochemical properties of the input molecules.
        :param mols: Valid RDKit mol objects.
        :return: molecular properties and the bounds of their intervals.
        """

        prop_1 = []
        prop_2 = []
        prop_3 = []
        for mol in mols:
            prop_1.append(Descriptors.TPSA(mol))
            prop_2.append(Descriptors.MolWt(mol))
            prop_3.append(Crippen.MolLogP(mol))

        return [prop_1, prop_2, prop_3], [[0, 130.12], [16, 152.04], [-4.91, 3.76]]

    def GPRegress(self, x_, y_, z_):
        """
            Gaussian process regression.
        :param x_: x dimension of grid points.
        :param y_: y dimension of grid points.
        :param z_:  property value at grid points.
        :return: new NxN grid with the corresponding property values.
        """

        # -- construct new grid
        N = 50
        x1x2 = np.array(list(product(np.linspace(x_.min(), x_.max(), N), np.linspace(y_.min(), y_.max(), N))))

        # -- fit data
        gp = GaussianProcessRegressor(kernel=Matern(nu=0.01), n_restarts_optimizer=15)
        gp.fit(np.stack((x_, y_)).T, z_)

        # -- predict
        y_pred = gp.predict(x1x2)

        return x1x2[:, 0].reshape(N, N), x1x2[:, 1].reshape(N, N), np.reshape(y_pred, (N, N))

    def LatentMap(self, model_name='/model.pth'):
        """
            Constructing a map of physicochemical properties for the latent space.
        :param valid_z: samples from z space corresponding to valid molecular graphs.
        :param model_name: path to the saved trained model.
        :return: stores latent space map of physicochemical properties.
        """

        # -- load trained model
        model = torch.load(self.filemodel + model_name)

        # -- encode inputs
        signal = self.sig.reshape(-1, self.n_max_atom, self.n_atom_features)
        signal_in = torch.transpose(signal, 2, 1)
        mu, logvar = model.encode(self.scat(self.adj, signal_in).reshape(-1, self.sdim * self.n_atom_features))

        # -- compute principle axis
        pca = PCA(n_components=2)
        pca.fit(mu.cpu().detach().numpy())

        # -- construct grid
        tmp = torch.linspace(-4, 4, steps=70)
        xv, yv = torch.meshgrid([tmp, tmp])
        grid_2D = torch.cat((xv.reshape(-1, 1), yv.reshape(-1, 1)), dim=1)

        # -- map grid to molecular space
        grid_samples = model.decode(
            torch.tensor(pca.inverse_transform(grid_2D)).to(self.device).float())  # Whole grid in graph space

        grid_sig = torch.argmax(grid_samples[0], dim=2)
        grid_adj = torch.argmax(grid_samples[1], dim=3)
        grid_adj = grid_adj - torch.diag_embed(torch.einsum('...ii->...i', grid_adj))

        with torch.no_grad():
            with open(self.res_dir + '/grid.data', 'wb') as f:
                pickle.dump(grid_sig.clone(), f)
                pickle.dump(grid_adj.clone(), f)
                pickle.dump(grid_2D.clone(), f)

        grid_mols = self.MolFromSample(grid_sig, grid_adj)  # valid grid mol objects
        valid_grid_mols, valid_grid_2D = self.QualityMetrics(grid_mols, grid_2D, grid_adj)
        props = {'Rings':[], 'SP3':[], 'PSA':[], 'MolWt':[]}
        for mol in valid_grid_mols:
            props['Rings'].append(Chem.rdMolDescriptors.CalcNumRings(mol))
            props['SP3'].append(Chem.rdMolDescriptors.CalcFractionCSP3(mol))
            props['PSA'].append(Descriptors.TPSA(mol))
            props['MolWt'].append(Descriptors.MolWt(mol))

        # -- plot property maps
        for idx, prp in enumerate(list(props.keys())):
            plt.figure(idx)
            f, ax = plt.subplots()

            # -- make grid with GP interpolation
            X0p, X1p, Zp = self.GPRegress(valid_grid_2D.T[0], valid_grid_2D.T[1], props[prp])

            im = plt.contour(X0p, X1p, Zp)
            im2 = plt.contourf(X0p, X1p, Zp, alpha=0.3)

            plt.clabel(im, inline=1, fontsize=10, fmt='%1.0f')
            plt.colorbar()
            f.tight_layout()
            f.savefig(self.res_dir + '/latn_comp_smooth_' + prp, bbox_inches='tight')
            plt.close()

    def ChemSpace(self, valid_mol):
        """
            Plot 2-dimensional histogram of chemical spaces.
        :param valid_mol: collection of input valid mol objects.
        :return: stores chemical spaces defined by physicochemical properties.
        """

        # -- filter repetitive moles
        valid_set = set(map(lambda x: Chem.MolToSmiles(x), valid_mol))
        Valid_mol_set = list(map(lambda x: Chem.MolFromSmiles(x), valid_set))

        # -- compute properties from mol objects
        props, bounds = self.ChemProperty(Valid_mol_set)

        # -- plot LogP vs MolWt chemical space
        plt.figure(2)
        f, ax = plt.subplots()
        plt.hist2d(props[1], props[2], range=[bounds[1], bounds[2]], bins=60, cmap=plt.cm.jet)
        f.tight_layout()
        plt.xlabel('MolWt')
        plt.ylabel('LogP')
        f.savefig(self.res_dir + '/prop_joint', bbox_inches='tight')
        plt.close()

    def LoadMols(self, name):
        """
            loading unique and valid mol objects from tensors of adjacency matrix and signals
        :param name: name of the samples file.
        :return: list of unique and valid mol objects,
        """

        # -- define validity filters
        valid_con, valid_val, valid_all = self.ValidityFilters()

        # -- load samples
        with open(name, 'rb') as f:
            sig = pickle.load(f)
            adj = pickle.load(f)

        # -- convert to mol object
        mols = self.MolFromSample(sig, adj)

        # -- filter valid and unique mol objects
        valid_mols = list(filter(valid_all, mols))
        valid_sml_set = set(map(lambda x: Chem.MolToSmiles(x), valid_mols))

        return list(map(lambda x: Chem.MolFromSmiles(x), valid_sml_set))


def parse_args():
    desc = "Uncertainty quantification of adaptive kernel Graph Scattering VAE (GSVAE) using predictive Bayesian bootstrap."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--BB_samples', type=int, default=25, help='Number of Bayesian bootstrap samples.')
    parser.add_argument('--N', type=int, default=600, help='Number of training data for MWLE estimate.')
    parser.add_argument('--database', type=str, default='QM9', help='Training database name.')
    parser.add_argument('--sample_file', type=str, default='BB_600', help='Predictive samples directory.')
    parser.add_argument('--gpu_mode', type=int, default=0, help='Accelerate the script using GPU.')

    args = parser.parse_args()

    args.scat_layers = args.n_samples = args.loadtrainedmodel = args.wlt_scales = args.mu_reg_1 = args.mu_reg_2 = \
        args.z_dim = args.epochs = 0

    # -- scattering
    args.sdim = 0
    for l in range(args.scat_layers):
        args.sdim += args.wlt_scales ** l

    # -- storage settings
    dir = os.getcwd()
    res_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.res_dir = os.path.join(dir, 'results/', res_name)
    os.makedirs(args.res_dir)

    # -- loading data
    args.sample_dir = os.path.join(dir, 'data/samples')

    # -- dataset specification
    args.atom_dict = {0: 'C', 1: 'O', 2: 'N', 3: 'F', 4: 'H'}
    args.n_node = 9
    args.n_atom_type = 5

    args.data_dir = os.path.join(dir, 'data/' + args.database + '_0.data')
    args.n_bond_type = 4
    args.n_scat_atom_features = args.n_atom_type
    args.y_target = None

    # -- GPU settings
    args.device = torch.device('cuda' if (bool(args.gpu_mode) and torch.cuda.is_available()) else 'cpu')
    if bool(args.gpu_mode) and not torch.cuda.is_available():
        print('No GPUs on this device! Using CPU instead.')

    return args

def main():
    """
        performing UQ analysis
    """

    # -- initialize
    args = parse_args()
    chem = chemf(args)
    my_tools = tools(args)
    all_valid_mols = []

    store_quant = True
    quant_name = 'quants_' + args.sample_file + '.data'

    BB_dir = os.path.join(args.sample_dir, args.sample_file)
    if not os.path.isdir(BB_dir):
        sys.tracebacklimit = 0
        raise OSError('Samples not found. Put samples in {}'.format(BB_dir))

    if store_quant:
        # -- load samples
        for i in range(1, args.BB_samples + 1):
            valid_mol_set = chem.LoadMols(BB_dir + '/samples_' + str(i) + '.data')
            all_valid_mols.append(valid_mol_set)

        # -- plot histograms
        prob_all, bins_all, bounds = my_tools.HistStat(all_valid_mols)
        quants = []
    else:
        with open(args.sample_dir + '/' + quant_name, 'rb') as f:
            quants = pickle.load(f)
            bins_all = pickle.load(f)
            bounds = pickle.load(f)
    for k, bound in enumerate(bounds):

        # -- quantiles
        if store_quant:
            quant = np.quantile(prob_all[str(k)], [0.05, 0.95], axis=1)
            quants.append(quant)
        else:
            quant = quants[k]
        my_tools.ErrorBars(quant, bins_all[k], 'red')

        plt.grid(linestyle='--')
        plt.savefig(args.res_dir + '/error_bars_' + str(k), bbox_inches='tight')
        plt.close()

    if store_quant:
        with open(args.sample_dir + '/' + quant_name, 'wb') as f:
            pickle.dump(quants, f)
            pickle.dump(bins_all, f)
            pickle.dump(bounds, f)

if __name__ == '__main__':
    main()
