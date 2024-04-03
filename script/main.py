import numpy as np
import torch
import torch.nn as nn
import os

from sklearn.metrics import (
    r2_score,
)
from sklearn.mixture import GaussianMixture
import argparse

# main.py
import lightning as L
from sklearn.mixture import GaussianMixture
from sklearn.metrics import r2_score


# ---
from model.gm_model import gen_target_with_gm_parallel
from model.parzen_model import ParzenWindow_Model, gen_target_with_parzen_parallel
from model.knn_model import KNN_Model
from utils.utils import set_seed, check_base_dir, generate_unique_id
from utils.summary import Summary
from model.lightning_model import LitModularNN, MetricTracker
from utils.data_manager import PDF


def arg_parsing():
    # command line parsing
    parser = argparse.ArgumentParser(description="Project for AI Exam")
    parser.add_argument("--pdf", type=str, default="default")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--samples", type=int, default=100)

    parser.add_argument("--show", action="store_true", default=False)
    parser.add_argument("--save", action="store_true", default=False)

    parser.add_argument("--components", type=int, default=4)
    parser.add_argument("--mlp_targets", action="store_true", default=False)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--bias", action="store_true", default=False)
    parser.add_argument("--gridsearch", action="store_true", default=False)

    parser.add_argument("--gmm", action="store_true", default=False)
    parser.add_argument("--knn", action="store_true", default=False)
    parser.add_argument("--parzen", action="store_true", default=False)
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = arg_parsing()

    # select model type from "GMM" "MLP" "Parzen Window" "KNN" "Parzen Window + NN" "GMM + NN"
    model_type = "PNN"

    dataset_params = {
        "n_samples": args.samples,
        "seed": 42,
        "target_type": "GMM" if model_type == "GMM + NN" else "PARZEN",
    }

    gm_model_params = {
        "n_components": 4,
        "n_init": 11,
        "max_iter": 684,
        "init_params": "kmeans",
        "random_state": dataset_params["seed"],
    }

    knn_model_params = {"k1": 1.0494451711015031, "kn": 23}

    parzen_window_params = {"h": 0.3795755152130492}  # trovato con optuna

    # --- MLP PARAMS --------
    mlp_params = {
        "dropout": 0.000,
        "hidden_layer": [(64, nn.Tanh()), (56, nn.Tanh()), (38, nn.Tanh())],
        "last_activation": None,
    }

    train_params = {
        "epochs": 300,
        "batch_size": 26,
        "loss_type": "mse_loss",
        "optimizer": "Adam",
        "learning_rate": 0.000874345,
    }

    gmm_target_params = {
        "n_components": 10,
        "n_init": 100,
        "max_iter": 80,
        "init_params": "k-means++",
        "random_state": dataset_params["seed"],
    }

    pw_target_params = {"h": 0.3795755152130492}

    set_seed(dataset_params["seed"])

    if args.pdf in ["exponential", "exp"]:
        pdf = PDF(
            [
                [
                    {"type": "exponential", "rate": 0.6},
                ]
            ],
            name="exponential 0.6",
        )
    elif args.pdf in ["multimodal logistic", "logistic"]:
        pdf = PDF(
            [
                [
                    {"type": "logistic", "mean": 20, "scale": 0.5, "weight": 0.4},
                    {"type": "logistic", "mean": 10, "scale": 4, "weight": 0.4},
                    {"type": "logistic", "mean": 17, "scale": 1, "weight": 0.2},
                ],
            ],
            name="multimodal 3 logistic",
        )
    else:
        pdf = PDF(default="MULTIVARIATE_1254")

    X_train, Y_train = pdf.generate_training(
        n_samples=dataset_params["n_samples"], seed=dataset_params["seed"]
    )

    # generate the data for plotting the pdf
    X_test, Y_test = pdf.generate_test(stepper=0.01)

    X_val, Y_val = pdf.generate_validation(n_samples=50)

    target_y = None
    target_params = None
    train_loss = None
    val_loss = None
    val_score = None
    model_params: dict = {}

    # --------------------------------------------------------------------

    if model_type in ["MLP", "GMM + NN", "Parzen Window + NN", "PNN", "GNN"]:
        print("Training Neural Network")
        model_params = mlp_params

        # check if a saved target file exists:

        base_dir = ["..", "data", "MLP"]
        base_dir = check_base_dir(base_dir)

        if dataset_params["target_type"] == "GMM":
            # generate the id
            target_unique_id = generate_unique_id(
                [pdf.training_X, pdf.test_Y, args.bias, gmm_target_params, dataset_params["seed"]], 5
            )

            save_filename = f"train_mlp{'_Biased' if args.bias == True else '' }_{gmm_target_params['init_params']}_C{gmm_target_params['n_components']}"

            if save_filename is not None:
                save_filename = save_filename.split(".")[0]
                save_filename = save_filename + "_" + target_unique_id + ".npz"
                save_filename = os.path.join(base_dir, save_filename)

            gm_model = GaussianMixture(**gmm_target_params)

            _, target_y = gen_target_with_gm_parallel(
                gm_model=gm_model,
                X=pdf.training_X,
                progress_bar=True,
                n_jobs=-1,
                save_filename=f"train_old-{gmm_target_params['n_components']}.npz",
            )

            target_y = torch.tensor(target_y, dtype=torch.float32)
            target_params = gmm_target_params

        elif dataset_params["target_type"] == "PARZEN":

            parzen_model = ParzenWindow_Model(**pw_target_params)

            _, target_y = gen_target_with_parzen_parallel(
                parzen_model=parzen_model,
                X=pdf.training_X,
                progress_bar=True,
                n_jobs=-1,
            )

            target_y = torch.tensor(target_y, dtype=torch.float32)
            target_params = pw_target_params

        X_train = torch.tensor(pdf.training_X, dtype=torch.float32)
        Y_train = torch.tensor(pdf.training_Y, dtype=torch.float32)

        X_test = torch.tensor(X_test, dtype=torch.float32)
        Y_test = torch.tensor(Y_test, dtype=torch.float32)

        X_val = torch.tensor(X_val, dtype=torch.float32)
        Y_val = torch.tensor(Y_val, dtype=torch.float32)

        xy_train = torch.cat((X_train, target_y), 1)
        xy_test = torch.cat((X_test, Y_test), 1)
        xy_val = torch.cat((X_val, Y_val), 1)

        train_loader = torch.utils.data.DataLoader(
            xy_train,
            batch_size=train_params["batch_size"],
            shuffle=True,
            num_workers=7,
            persistent_workers=True,
        )

        val_loader = torch.utils.data.DataLoader(
            xy_val,
            batch_size=train_params["batch_size"],
            shuffle=False,
            num_workers=0,
        )

        model = LitModularNN(**mlp_params, learning_rate=train_params["learning_rate"])
        cb = MetricTracker()
        trainer = L.Trainer(accelerator="auto", max_epochs=train_params["epochs"], callbacks=[cb])
        trainer.fit(model, train_loader, val_loader)
        train_loss = cb.train_epoch_losses
        val_loss = cb.val_epoch_losses
        val_score = cb.val_epoch_scores

        # evaluate model
        model.eval()
        with torch.no_grad():
            pdf_predicted = model(X_test)
            pdf_predicted = pdf_predicted.detach().numpy()
            r2_value = r2_score(Y_test, pdf_predicted)

        X_train = X_train.detach().numpy()
        Y_train = Y_train.detach().numpy()
        Y_test = Y_test.detach().numpy()
        X_test = X_test.detach().numpy()
        target_y = target_y.detach().numpy()

    # ----------------------------------------------------------------------
    elif model_type in ["Parzen Window", "PARZEN", "parzen"]:
        model_params = parzen_window_params
        model = ParzenWindow_Model(h=parzen_window_params["h"])
        model.fit(training=X_train)
        pdf_predicted = model.predict(test=X_test)

        r2_value = r2_score(Y_test, pdf_predicted)

    # ----------------------------------------------------------------------
    elif model_type == "GMM":
        model_params = gm_model_params
        model = GaussianMixture(**gm_model_params)
        model.fit(X_train, Y_train)

        # predict the pdf with GMM
        pdf_predicted = np.exp(model.score_samples(X_test))

        r2_value = r2_score(Y_test, pdf_predicted)

    # ----------------------------------------------------------------------
    elif model_type == "KNN":
        model_params = knn_model_params
        model = KNN_Model(**knn_model_params)
        model.fit(pdf.training_X)

        pdf_predicted = model.predict(pdf.test_X)

        r2_value = r2_score(pdf.test_Y, pdf_predicted)

    # Creo l'oggetto che mi gestirà il salvataggio dell'esperimento e gli passo tutti i parametri
    experiment_name = f"Experiment "

    if dataset_params["target_type"] == "GMM":
        experiment_name += f" C{gmm_target_params['n_components']} "
    elif dataset_params["target_type"] == "PARZEN":
        experiment_name += f" H{pw_target_params['h']} "

    experiment_name += f"S{dataset_params['n_samples']}"

    summary = Summary(
        experiment=experiment_name,
        model_type=model_type,
        pdf=pdf,
        dataset_params=dataset_params,
        model_params=model_params,
        target_params=target_params,
        train_params=train_params,
        overwrite=True,
    )

    summary.calculate_metrics(pdf.training_Y, pdf.test_Y, pdf_predicted, target_y)
    summary.plot_pdf(pdf.training_X, target_y, pdf.test_X, pdf.test_Y, pdf_predicted)
    summary.plot_loss(train_loss, val_loss, loss_name=train_params["loss_type"])
    summary.log_dataset()
    summary.log_target()
    summary.log_model(model=model)
    summary.log_train_params()
    summary.leaderboard()

    print("ID EXPERIMENT:", summary.id_experiment)
    print("R2 score: ", r2_value)
    print("KL divergence: ", summary.model_metrics.get("kl"))
    print("Done!")
