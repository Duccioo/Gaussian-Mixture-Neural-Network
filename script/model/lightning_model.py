import torch, torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import Callback
from sklearn.metrics import r2_score

# ---
from model.nn_model import NeuralNetworkModular


class MetricTracker(Callback):

    def __init__(self):
        self.val_batch_losses: list = []
        self.val_epoch_losses: list = []
        self.val_epoch_scores: list = []
        self.train_epoch_losses: list = []
        self.train_batch_losses: list = []

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: torch.Tensor,
        batch: torch.Any,
        batch_idx: int,
    ) -> None:
        tacc = outputs["loss"]  # you can access them here
        self.train_batch_losses.append(tacc)  # track them

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        elogs = trainer.logged_metrics["train_loss"].item()  # access it here
        self.train_epoch_losses.append(elogs)

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: torch.Tensor,
        batch: torch.Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        v_loss = outputs  # you can access them here
        self.val_batch_losses.append(v_loss)  # track them
        return super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        elogs_loss = trainer.logged_metrics["val_loss"].item()  # access it here
        elogs_r2 = trainer.logged_metrics["val_acc"].item()
        self.val_epoch_losses.append(elogs_loss)
        self.val_epoch_scores.append(elogs_r2)
        # do whatever is needed


class LitModularNN(L.LightningModule):
    def __init__(
        self,
        hidden_layer,
        last_activation,
        dropout,
        learning_rate: float = 0.001,
        optimizer_name: str = "Adam",
        loss_name: str = "huber_loss",
    ):
        super().__init__()

        self.neural_netowrk_modular = NeuralNetworkModular(
            1,
            1,
            dropout=dropout,
            hidden_layer=hidden_layer,
            last_activation=last_activation,
        )
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.loss_name = loss_name

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.neural_netowrk_modular(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        data = batch[:, 0]
        target = batch[:, 1]
        data = data.view(-1, 1)
        target = target.view(-1, 1)

        x_hat = self.neural_netowrk_modular(data)
        loss = getattr(F, self.loss_name)(x_hat, target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch[:, 0]
        target = batch[:, 1]
        data = data.view(-1, 1)
        target = target.view(-1, 1)

        x_hat = self.neural_netowrk_modular(data)
        loss = getattr(F, self.loss_name)(x_hat, target)
        self.log("val_loss", loss)
        self.log(
            "val_acc",
            r2_score(target.cpu().detach().numpy(), x_hat.cpu().detach().numpy()),
        )
        return loss

    def test_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        data = batch[:, 0]
        target = batch[:, 1]
        data = data.view(-1, 1)
        target = target.view(-1, 1)

        x_hat = self.neural_netowrk_modular(data)
        loss = getattr(F, self.loss_name)(x_hat, target)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        learning_rate = self.learning_rate
        optimizer = getattr(torch.optim, self.optimizer_name)(
            self.neural_netowrk_modular.parameters(), lr=learning_rate
        )
        return optimizer
