# main.py
import torch, torch.nn as nn, torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import Callback
from sklearn.mixture import GaussianMixture
from sklearn.metrics import r2_score
import seaborn as sns
from matplotlib import pyplot as plt

# ---
from utils.data_manager import load_multivariate_dataset
from model.gm_model import gen_target_with_gm_parallel
from model.lightning_model import LitModularNN, MetricTracker
from utils.utils import set_seed
from utils.summary import Summary

# --------------------------------
# Step 1: Define a LightningModule
# --------------------------------
# A LightningModule (nn.Module subclass) defines a full *system*
# (ie: an LLM, diffusion model, autoencoder, or simple image classifier).

if __name__ == "__main__":

    mlp_params = {
        "dropout": 0.001,
        "hidden_layer": [
            (38, nn.ReLU()),
            (7, nn.Tanh()),
        ],
        "last_activation": "lambda",
    }

    train_params = {
        "epochs": 700,
        "batch_size": 51,
        "loss_type": "mse_loss",
        "optimizer": "Adam",
        "learning_rate": 0.001,
    }

    dataset_params = {
        "n_samples": 100,
        "seed": 36,
        "target_type": "GMM",
    }

    gmm_target_params = {
        "n_components": 14,
        "n_init": 60,
        "max_iter": 80,
        "init_params": "kmeans",
        "random_state": dataset_params["seed"],
    }

    set_seed(dataset_params["seed"])

    # -------------------
    # Step 2: Define data
    # -------------------
    X_train, Y_train, X_test, Y_test, pdf = load_multivariate_dataset(
        dataset_params["n_samples"], dataset_params["seed"]
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float3)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32)

    gm_model = GaussianMixture(**gmm_target_params)

    _, gmm_target_y = gen_target_with_gm_parallel(
        gm_model=gm_model,
        X=X_train,
        progress_bar=True,
        n_jobs=-1,
        save_filename=f"train_old-{gmm_target_params['n_components']}.npz",
    )

    gmm_target_y = torch.tensor(gmm_target_y, dtype=torch.float32)

    xy_train = torch.cat((X_train, gmm_target_y), 1)

    xy_test = torch.cat((X_test, Y_test), 1)

    train_loader = torch.utils.data.DataLoader(
        xy_train, batch_size=train_params["batch_size"], shuffle=True, num_workers=0
    )

    # -------------------
    # Step 3: Train
    # -------------------
    model = LitModularNN(**mlp_params, learning_rate=train_params["learning_rate"])
    cb = MetricTracker()
    trainer = L.Trainer(
        accelerator="auto", max_epochs=train_params["epochs"], callbacks=[cb]
    )
    trainer.fit(model, train_loader)

    # evaluate model
    model.eval()
    with torch.no_grad():
        y_predicted_mlp = model(torch.tensor(X_test, dtype=torch.float32))
        y_predicted_mlp = y_predicted_mlp.detach().numpy()
        r2 = r2_score(Y_test, y_predicted_mlp)

    print("R2 SCORE", r2)

    summary = Summary(
        experiment=f"Lightning C{gmm_target_params['n_components']} S{dataset_params['n_samples']}",
        model_type="GMM + NN",
        pdf=pdf,
        dataset_params=dataset_params,
        model_params=mlp_params,
        target_params=gmm_target_params,
        train_params=train_params,
        overwrite=True,
    )
    summary.calculate_metrics(
        Y_train.detach().numpy(),
        Y_test.detach().numpy(),
        y_predicted_mlp,
        gmm_target_y.detach().numpy(),
    )
    summary.plot_pdf(
        X_train.detach().numpy(),
        gmm_target_y.detach().numpy(),
        X_test.detach().numpy(),
        Y_test.detach().numpy(),
        y_predicted_mlp,
    )
    summary.plot_loss(cb.train_epoch_losses, loss_name=train_params["loss_type"])
    summary.log_dataset()
    summary.log_target()
    summary.log_model(model=model)
    summary.log_train_params()
    summary.leaderboard()
