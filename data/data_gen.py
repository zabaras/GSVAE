"""
Author: Navid Shervani-Tabar
"""
import pickle
import numpy as np

from rdkit import Chem
from chainer_chemistry import datasets
from rdkit.Chem import Descriptors, Crippen
from chainer_chemistry.dataset.preprocessors.ggnn_preprocessor import GGNNPreprocessor


# -- properties
n_samples = 1           # number of bootstrap samples (set to 1 for no bootstrapping)
data_size = 40000       # train + test
N         = 600         # size of the training dataset
database  = 'QM9'       # dataset name
if database == 'QM9':
    atom_types = ['6', '8', '7', '9', '1']
    n_max_atom = 9

# -- set data indices
X = np.random.choice(133885, size=data_size, replace=False)     # train + test data indices
X_train = X[:N]                                                 # train data indices
X_test = X[N:]                                                  # test data indices

# -- bootstrapped data indices
for b in range(n_samples):
    if n_samples is not 1:
        X_b = np.concatenate(([X_train[np.random.choice(range(N), size=N)], X_test]), axis=0)
    else:
        X_b = X

    # -- generate dataset
    dataset, dataset_smiles = datasets.get_qm9(GGNNPreprocessor(kekulize=True), return_smiles=True, target_index=X_b)

    smils_list = []
    atoms_list = []
    adjs_list = []
    prop_1 = []
    prop_2 = []
    prop_3 = []

    for idx in range(data_size):
        atom, adj, _ = dataset[idx]

        n_atom = len(atom)

        # -- smiles
        smils_list.append(dataset_smiles[idx])
        mol = Chem.MolFromSmiles(dataset_smiles[idx])

        # -- adj
        adjacency = adj[0] + 2 * adj[1] + 3 * adj[2]
        adjacency_append = np.zeros((n_max_atom, n_max_atom)).astype(float)
        adjacency_append[:n_atom, :n_atom] = adjacency
        adjs_list.append(adjacency_append)

        # -- signal
        signal = []
        for i in range(n_atom):
            atom_sig = []
            atom_i = mol.GetAtomWithIdx(i)

            atom_sig += [float(str(atom_i.GetAtomicNum()) == x) for x in atom_types]    # one-hot vector of atom types
            signal.append(atom_sig)

        for i in range(n_max_atom - n_atom):                                            # dummy value for ghost nodes
            atom_sig = []
            atom_sig += list(np.zeros(len(atom_types) - 1)) + [1.]
            signal.append(atom_sig)

        atoms_list.append(np.array(signal).T)

        # -- properties
        prop_1.append(Descriptors.TPSA(mol))
        prop_2.append(Descriptors.MolWt(mol))
        prop_3.append(Crippen.MolLogP(mol))

    signal = np.asarray(atoms_list).transpose([0, 2, 1]).reshape(-1, len(atom_types) * n_max_atom)

    # -- save
    if n_samples is not 1:
        data_name = database + '_N_' + str(N) + '_' + str(b) + '.data'
    else:
        data_name = database + '_0.data'

    with open(data_name, 'wb') as f:
        pickle.dump(smils_list, f)
        pickle.dump(signal, f)
        pickle.dump(adjs_list, f)

        pickle.dump(prop_1, f)
        pickle.dump(prop_2, f)
        pickle.dump(prop_3, f)
