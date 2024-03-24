# main.py
import torch, torch.nn as nn, torch.nn.functional as F
import lightning as L
from sklearn.mixture import GaussianMixture
from sklearn.metrics import r2_score
import seaborn as sns
from matplotlib import pyplot as plt

# ---
from utils.data_manager import load_multivariate_dataset
from model.gm_model import gen_target_with_gm_parallel
from model.nn_model import NeuralNetworkModular
from utils.utils import set_seed

# --------------------------------
# Step 1: Define a LightningModule
# --------------------------------
# A LightningModule (nn.Module subclass) defines a full *system*
# (ie: an LLM, diffusion model, autoencoder, or simple image classifier).


class LitAutoEncoder(L.LightningModule):
    def __init__(self):
        super().__init__()
        dropout = 0.001
        hidden_layer = [
            (38, nn.ReLU()),
            (7, nn.Tanh()),
        ]
        last_activation = None

        self.neural_netowrk_modular = NeuralNetworkModular(
            1,
            1,
            dropout=dropout,
            hidden_layer=hidden_layer,
            last_activation=last_activation,
        )
        # self.decoder = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 1))

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.neural_netowrk_modular(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        data = batch[:, 0]
        target = batch[:, 1]
        data = data.view(-1, 1)
        target = target.view(-1, 1)

        x_hat = self.neural_netowrk_modular(data)
        # x_hat = self.decoder(z)
        loss = F.huber_loss(x_hat, target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch[:, 0]
        target = batch[:, 1]
        data = data.view(-1, 1)
        target = target.view(-1, 1)

        x_hat = self.neural_netowrk_modular(data)
        # x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, target)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.neural_netowrk_modular(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        learning_rate = 0.00269
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        return optimizer


if __name__ == "__main__":

    batch_size = 51
    n_samples = 100
    seed = 36
    n_components = 14
    init_params_gmm = "kmeans"
    n_init = 60
    max_iter = 80
    set_seed(seed)
    epochs = 794
    # -------------------
    # Step 2: Define data
    # -------------------
    X_train, Y_train, X_test, Y_test = load_multivariate_dataset(n_samples, seed)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32)

    gm_model = GaussianMixture(
        n_components=n_components,
        init_params=init_params_gmm,
        random_state=seed,
        n_init=n_init,
        max_iter=max_iter,
    )

    # file_name = f"target_gm_C{n_components}_S{n_samples}_P{init_params_gmm}_N{n_init}_M{max_iter}.npz"
    # file_path = os.path.join(tmp_dir, file_name)

    _, gmm_target_y = gen_target_with_gm_parallel(
        gm_model=gm_model,
        X=X_train,
        progress_bar=True,
        n_jobs=-1,
        save_filename=f"quiqui-{n_components}.npz",
    )

    gmm_target_y = torch.tensor(gmm_target_y, dtype=torch.float32)

    xy_train = torch.cat((X_train, gmm_target_y), 1)

    xy_test = torch.cat((X_test, Y_test), 1)

    train_loader = torch.utils.data.DataLoader(
        xy_train, batch_size=batch_size, shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        xy_test,
        batch_size=batch_size,
        num_workers=3,
    )

    # -------------------
    # Step 3: Train
    # -------------------
    model = LitAutoEncoder()
    trainer = L.Trainer(accelerator="cpu", max_epochs=epochs)
    trainer.fit(model, train_loader)

    # evaluate model
    model.eval()
    with torch.no_grad():
        y_predicted_mlp = model(torch.tensor(X_test, dtype=torch.float32))
        y_predicted_mlp = y_predicted_mlp.detach().numpy()
        r2 = r2_score(Y_test, y_predicted_mlp)
        print(r2)

    # print figure

    sns.lineplot(x=X_test.flatten(), y=Y_test.flatten(), color="green", label="True")
    sns.lineplot(
        x=X_test.flatten(), y=y_predicted_mlp.flatten(), label="base", color="red"
    )
    sns.scatterplot(
        x=X_train.flatten(), y=gmm_target_y.flatten(), color="purple", label="GMM"
    )

    plt.legend()
    plt.show()
