# Gaussian Mixture Neural Network

A model that combines Gaussian mixture models with Neural Networks

## Summary of the Project

The project consists in studying a new algorithm, along the lines of Parzen Neural Networks, brings together the statistical approach and machine learning approaches for training an ANN from an unlabeled data sample of patterns randomly drawn from an underlying probability density function (PDF).

The algorithm leverages both the generalization capabilities of ANNs and
the generality of the maximum-likelihood estimates of the parameters
of Gaussian mixture models. Therefore, the proposed machine is termed
**Gaussian-mixture Neural Network (GNN)**. The best selling points of the
GNN lie in its simplicity and effectiveness.

(This project is the continuation of the work carried out for the A.I. exam., Siena, 2024)

## Paper

This project was submitted to ANNPR 2024.

[Paper LINK ](https://openreview.net/forum?id=foiH9tX3Fc)

## File Structure:

- **`data` folder**:

  Contains the save files of the training and test data.
  This files are automatically generated and saved when run the algorithm.

- **`script` folder**:

  It contains all the code for running the project.

  - **`model` folder** contains scripts for each model used in the paper (for example in `moddel/nn_model.py` there is all the code for building the Neural Network used in the GNN and PNN models)

  - `utils/data_manager.py`: the data manager for the creation of the PDF samples for the training and test set
  - `optuna` folder: contains optuna scripts for searching the best hyperparameters for each model

  - **`main_MLP.py`**: the main function for the training and evaluetion of the GNN and PNN models
  - **`main_STATISTIC.py`**: the main function for the training and evaluetion of Statistic Models (Kn-NN, GMM, Parzen Window)

  - `test_experiment.py`: script for change number of components and number of neurons in GNN models.

- `miscellaneous` folder: some random code created at the beginning of the project
- `optuna_database` folder: contains all the database file created with optuna with all the trials for all the experiments used to find the best hyperparameters
- `project` folder: contains old result created at the beginning of the project for A.I. Exams (with [@AparnaPindali](https://github.com/AparnaPindali))

## Installation

In order to run all the script you need to run the following:

1. Clone the project:

   ```bash
   git clone https://github.com/Duccioo/Gaussian-Mixture-Neural-Network.git
   ```

2. install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. install Pytorch on your machine: https://pytorch.org/get-started/locally/

## Run Locally

 <h3>  A) Run GNN or PNN models </h3>

On the repo folder:

```bash
python script/main_MLP.py
```

You can also specify some additional arguments:

- `--pdf <str s>` : select the PDF type (default: exponential PDF with rate = 0.6). Other possibility is `multivariate` to select a multivariate PDF, otherwise you can specify with your prefer parameters.
- `--model <str s>` : select the model type, possibilities = ['GNN', 'PNN' ] (default: GNN)

- `--jobs <int n>`: specify the number of threads used to create the target samples for the MLP (default: 2)
- `--samples <int n>`: number of samples used for training (default: 100)
- `--show`: toggle used to show the predicted PDF (default: not show the graph)
- `--gpu`: toggle to use, if possible, GPU acceleration

#### Example:

```bash
 python script/main_MLP.py --model=PNN --samples=150 --pdf=multivariate --show
```

This command run the training on the PNN (Parzen Neural Network) on the Multivariate PDF with 150 samples used for the training, and at the end of training show the predicted PDF on the screen

<details>

<summary>  <h3> B) Run Statistical models </h3> </summary>

On the repo folder:

```bash
python script/main_STATISTIC.py
```

You can also specify some additional arguments:

- `--pdf <str s>` : select the PDF type (default: exponential PDF with rate = 0.6). Other possibility is `multivariate` to select a multivariate PDF, otherwise you can specify with your prefer parameters.
- `--model <str s> ` : select the model type, possibilities = ['KNN', 'Parzen', 'GMM' ] (default: GMM)

- `--samples <int n>`: number of samples used for training (default: 100)
- `--show`: toggle used to show the predicted PDF (default: not show the graph)

#### Example:

```bash
python script/main_STATISTICS.py --samples=300 --model=knn --pdf=exponential
```

This command run the training on the KNN (Kn-NN) on the Multivariate PDF with 300 samples used for the training

</details>

## Change Models Parameters (optional)

<details>
<summary>
It is possible to change the parameters of each model (statistical or Neural Network) by entering the specific file and modifying the python dictionaries in the **`__main__ `** part of the file.
 </summary>

### Statistical model

for GMM, Kn-NN and Parzen Window models it is possible to change the parameters that defined this models.
GMM model are created using Scikit-learn library ([GMM model](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html))

#### Some examples:

- **Kn-NN** :
  ```python
  knn_model_params = {"k1": 1.5}
  ```
- **Parzen Window** :
  ```python
  parzen_window_params = {"h": 0.2}
  ```
- **GMM** :
  ```python
  gm_model_params = {
        "random_state": 46,
        "init_params": "random_from_data", # "k-means++" or "random" or "kmeans" or "random_from_data"
        "max_iter": 90,
        "n_components": 5,
        "n_init": 60,
  }
  ```

### MLP model (GNN and PNN)

for MLP model it is possible to change the Neural Network architecture:

- `dropout` parameter (applied at each layer).
- the `hidden_layer` architecture (structured like a list with each element corresponding to a hidden layer with the number of neurons and the Activation function).
- the `last_activation`: set the last activation function to be applied to the output of the model, if `lambda` is specified than it will be applied a sigmoid function with adaptive amplitude parameter _'lambda'_.

#### Example:

```python
mlp_params = {
        "dropout": 0.000,
        "hidden_layer": [
            [9, nn.Tanh()],
            [20, nn.Sigmoid()],
            [34, nn.Sigmoid()],
            [26, nn.Tanh()],
        ],
        "last_activation": "lambda",  # None or lambda
    }
```

It is also possible to change the **training parameters** (_number of epochs, batch size, learning rate, etc..._) and the **target parameters** (using the parameters from GMM or Parzen Window models)

</details>

## Find the Best Parameters

In this project, we leveraged the Optuna library to perform hyperparameter optimization for our neural network models and the statistical modles. Optuna is an automatic hyperparameter optimization software framework, particularly designed for machine learning. It allows for an efficient search of hyperparameters, which can significantly enhance the performance of machine learning models.
The technique used for hyperparameter search is divided into 2 parts, the sampling of hyperparameters using **"TPESampler"** _(Tree-Structured Parzen Estimator)_ and the pruning of bad models using **"SuccessiveHalvingPruner"** _(Asynchronous Successive Halving)_, which are the algorithms default of this Optuna library.

There are 2 files for hyperparameter search:

1. `optuna_mlp.py` takes care of finding the best architectures and parameters for neural network models
2. `optuna_statistic.py` deals with finding the parameters of statistical models.

In each of the files you can modify the parameters for searching within the dictionary `params={}`

You can also specify other parameters from the command line:

- `--dataset` (multivariate or exp)
- `--objective` ([gmm, parzen, knn] for statistical models) ([GMM, PARZEN] for neural models)
- `--samples`
- `--trials` (How many optuna trials to do before the end of the experiment)

There is also a further file, `optuna_plotting.py`, relating to plotting various information and correlations between the R2 score and the various parameters of the experiments with optuna.

## Result

Preliminary experimental results showcased the potential of the GNN, the latter
outperforming the GMM (i.e., the student has become the master) and the most popular non-parametric estimators. Likewise, the GNN compared quite favorably
with the established ANN-based technique.

### Best Model for Multivariate PDF

![multivariate](./result/best/MLP/100/GNN%20MULTIVARIATE%20810a0d8c/pdf_810a0d8c.png)

### Best Model for Exponential PDF

![exponential](./result/best/MLP/200/GNN%20EXP%206b41f25c/pdf_6b41f25c.png)

## Authors

- [Edmondo Trentin](https://www3.diism.unisi.it/~trentin/HomePage.html)
- [@duccioo](https://github.com/Duccioo)

## License

[MIT](https://choosealicense.com/licenses/mit/)
