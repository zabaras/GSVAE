Predictive Molecular Graph Latent Space Discovery
======================================

This project is using Graph Scattering Variational AutoEncoder (GSVAE) to discovers a latent space for molecular graphs. A quantitative assessment of the latent space in terms of its predictive ability for organic molecules in the QM9 dataset is presented in the paper. All tests are performed with small sized training data. To account for the limited size training data set, a Bayesian formalism is considered that allows us capturing the uncertainties in the predicted properties.

## Getting Started

### Dependencies

This implementation requires:

* Python (>= 3.5)
* SciPy (>= 1.4.1)
* PyParsing (>= 1.1)
* PyTorch (>= 1.5.0)
* RDKit (>= 2019.09.3)
* NumPy (>= 1.18.1)
* Seaborn (>= 0.9.0)
* scikit-learn (>= 0.22.1)
* Matplotlib (>= 3.1.1)
* chainer-chemistry (>=0.6.0)

### Installation

After downloading the code, you may install it by running

```bash
pip install -r requirements.txt
```

Note that to use functions in `utils.py`, you need to have `RDKit` package installed. You can find more information at https://www.rdkit.org/docs/Install.html .

### Data

Data samples are generated through `data_gen.py`, which also performs classic bootstrapping. The script accepts the following arguments:

```bash
optional arguments:
  --data_size           Total size of the training + test dataset (default: 100000)
  --N                   Size of the training set. This only affects the bootstrapping (default: 600)
  --n_samples           Number of the bootstrap samples (default: 1, no bootstrap)
```

Note that the Bayesian bootstrapping is done in the main code. To generate data, run:

```bash
cd data
python3 data_gen.py
```

## Run

### Training

The model is trained using `main.py`. This code accepts the following arguments:

```bash
optional arguments:
  --epochs              number of epochs to train (default: 800)
  --batch_size          size of each batch (default: 32)
  --gpu_mode            accelerate the script using GPU (default: 1)
  --z_dim               latent space dimensionality (default: 30)
  --seed                random seed (default: 1400)
  --loadtrainedmodel    path to trained model
  --mu_reg_1            regularization parameter for ghost nodes and valence constraint (default: 0)
  --mu_reg_2            regularization parameter for connectivity constraint (default: 0)
  --mu_reg_3            regularization parameter for 3-member cycle constraint (default: 0)
  --mu_reg_4            regularization parameter for cycle with triple bond constraint (default: 0)
  --N_vis               number of test data for visualization (default: 3000)
  --log_interval        number of epochs between visualizations (default: 200)
  --n_samples           number of generated samples from molecular space (default: 10000)
  --wlt_scales          number of wavelet scales (default: 8)
  --scat_layers         number of scattering layers (default: 3)
  --database            name of the training database (default: 'QM9')
  --datafile            name and location of the training file in data folder (default: 'QM9_0.data')
  --BB_samples          index for Bayesian bootstrap sample (default: 0)
  --N                   number of training data (default: 600)
  --res                 path for storing the results (default: 'results/')
  --y_id                index for target property in the conditional design (default: None, unconditional design)
  --y_target            target property value in the conditional design (default: None, unconditional design)
```

After generating the data, run

```bash
python3 main.py
```

to train the base model. To run the constrained model, set the regularization parameters `mu_reg_1`, `mu_reg_2`, `mu_reg_3`, and `mu_reg_4` to a positive value and tune them based on the output statistics.

### Conditional design

This code performs conditional design by setting a target property value for the sampled molecules. Set the property ID with argument `y_id` (0: PSA, 1: MolWt, 2: LogP) and the target value with `y_target`.

### Quantifying uncertainties

To perform UQ analysis, use `utils.py`. The `utils.py` script accepts the following arguments:

```bash
optional arguments:
  --BB_samples          number of samples for uncertainty quantification (default: 0)
  --N                   number of training data (default: 600)
  --database            name of the training database (default: 'QM9')
  --sample_file         predictive samples directory (default: 'BB_600')
  --gpu_mode            accelerate the script using GPU (default: 0)
```

To compute the confidence interval, use the following example script

```bash
ITR=25
DIR=B_200
N=200

for i in `seq 1 ${ITR}`;
do
    python3 main.py --N "$N" --BB_samples "$i" --res results/"${DIR}"
done

mkdir data/samples
mkdir data/samples/${DIR}
mv results/"${DIR}"/*/samples_*.data data/samples/${DIR}

python3 utils.py --BB_samples "$ITR" --N "$N" --sample_file "${DIR}"
```

### Filters

You can run `filter.py` independently in order to perform scattering transform and visualize graph filters. The `filter.py` script accepts the following arguments:

```bash
optional arguments:
  --gpu_mode            accelerate the script using GPU (default: 0)
  --wlt_scales          number of wavelet scales (default: 8)
  --scat_layers         number of scattering layers (default: 3)
  --N                   number of training data (default: 600)
  --database            name of the training database (default: 'QM9')
```

## Citation

You can use this code, as whole or in part, by citing:
```latex
@article{shervani2020physics,
  title={Physics-Constrained Predictive Molecular Latent Space Discovery with Graph Scattering Variational Autoencoder},
  author={Navid, Shervani-Tabar and Zabaras, Nicholas},
  journal={arXiv preprint arXiv:2009.13878},
  year={2020}
}
```

## Questions

For any questions or comments regarding this work, feel free to submit an issue [here](https://github.com/nshervt/GSVAE/issues) or contact Navid Shervani-Tabar (nshervan@nd.edu). In the email title, please use "Regarding GSVAE paper".
