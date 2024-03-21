import os
import optuna
from optuna.trial import TrialState

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

from sklearn.metrics import r2_score
from rich.progress import track
import numpy as np

# ---
from model.parzen_model import ParzenWindow_Model
from utils.data_manager import PDF
from utils.data_manager import save_dataset, load_dataset
from utils.utils import check_base_dir, generate_unique_id
from model.nn_model import AdaptiveSigmoid


def define_MLP(trial: optuna.Trial, params: dict = {}):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int(
        "n_layers", params["n_layers"][0], params["n_layers"][1]
    )
    layers = []

    in_features = 1
    for i in range(n_layers):
        out_features = trial.suggest_int(
            "n_units_l{}".format(i),
            params["range_neurons"][0],
            params["range_neurons"][1],
        )

        out_attivation = trial.suggest_categorical(
            "activation_l{}".format(i), params["suggest_activation"]
        )

        layers.append(nn.Linear(in_features, out_features))
        init.xavier_normal_(layers[-1].weight)

        if out_attivation == "relu":
            layers.append(nn.ReLU())
        elif out_attivation == "tanh":
            layers.append(nn.Tanh())
        elif out_attivation == "sigmoid":
            layers.append(nn.Sigmoid())

        p = trial.suggest_float(
            "dropout_l{}".format(i),
            params["range_dropout"][0],
            params["range_dropout"][1],
            log=True,
        )
        layers.append(nn.Dropout(p))

        in_features = out_features

    layers.append(nn.Linear(in_features, 1))
    init.xavier_normal_(layers[-1].weight)

    last_activation = trial.suggest_categorical(
        "last_activation", params["suggest_last_activation"]
    )

    if last_activation == "lambda":
        layers.append(AdaptiveSigmoid())
    elif last_activation == "sigmoid":
        layers.append(nn.Sigmoid())
    elif last_activation == "relu":
        layers.append(nn.ReLU())

    model = nn.Sequential(*layers)

    return model


def generate_target_MLP_pnn(
    parzen_model: ParzenWindow_Model,
    X,
    bias: bool = False,
    save_filename: str = None,
    progress_bar: bool = False,
):
    # try to load the target data:
    if save_filename is not None and os.path.isfile(save_filename):
        X, Y = load_dataset(file=save_filename)
    else:
        Y = []
        # print(X.shape)
        pb = (
            track(enumerate(X), description="Generating Target: ", total=len(X))
            if progress_bar
            else enumerate(X)
        )

        for indx, sample in pb:
            if bias == False:
                X_1 = np.delete(X, indx, axis=0)

            else:
                X_1 = X

            parzen_model.fit(X_1)
            Y.append(parzen_model.predict(sample.reshape(-1, X_1.shape[1])))
        Y = np.array(Y).reshape(-1, 1)

        if save_filename is not None:
            save_dataset((X, Y), save_filename)

    return X, Y


def objective_MLP(
    trial: optuna.Trial, X_train, Y_train, X_test, Y_test, params, device
):

    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)

    # Generate the model.
    model = define_MLP(trial, params=params).to(device)

    # Generate the optimizers.
    lr = trial.suggest_float(
        "learning_rate",
        params["learning_rate"][0],
        params["learning_rate"][1],
        log=True,
    )
    # loss_name = trial.suggest_categorical("loss", params["loss"])
    s_epoch = trial.suggest_int("epoch", params["epoch"][0], params["epoch"][1])
    optimizer_name = trial.suggest_categorical("optimizer", params["optimizer"])
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    train_loader = get_dataloader(trial, X_train, Y_train, params["batch_size"])

    # Training of the model.
    for epoch in range(s_epoch):
        model.train()
        for batch_idx, train_data in enumerate(train_loader):
            data = train_data[:, 0]
            target = train_data[:, 1]
            # Limiting training data for faster epochs.

            data, target = data.view(data.size(0), 1).to(device), target.view(
                target.size(0), 1
            ).to(device)

            optimizer.zero_grad()
            output = model(data)

            loss_value = F.huber_loss(output, target)
            loss_value.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        with torch.no_grad():
            output = model(X_test)

        r2_value = r2_score(output.numpy(), Y_test)
        # mse = mean_squared_error(output.numpy(), Y_test)

        trial.report(r2_value, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return r2_value


def get_dataloader(trial: optuna.Trial, X_train, Y_train, batch_size: list = [2, 64]):

    batch_size = trial.suggest_int("batch_size", batch_size[0], batch_size[1])

    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)

    xy_train = torch.cat((X_train, Y_train), 1)

    train_loader = torch.utils.data.DataLoader(
        xy_train,
        batch_size=batch_size,
        shuffle=True,
    )

    return train_loader


def load_dataset_optuna(n_samples, seed):
    pdf = PDF(default="MULTIVARIATE_1254")
    stepper_x_test = 0.01
    X_train, Y_train = pdf.generate_training(n_samples=n_samples, seed=seed)

    # generate the data for plotting the pdf
    X_test, Y_test = pdf.generate_test(stepper=stepper_x_test)

    return X_train, Y_train, X_test, Y_test


if __name__ == "__main__":

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    params = {
        # TRAIN PARAMS:
        "optimizer": ("Adam", "RMSprop"),
        "learning_rate": (1e-5, 1e-1),
        "epoch": (100, 1000),
        "batch_size": (2, 64),
        # MODEL PARAMS:
        "n_layers": (1, 4),
        "range_neurons": (4, 64),
        "range_dropout": (0.001, 0.05),
        "suggest_activation": ("relu", "tanh"),
        "suggest_last_activation": ("lambda", None),
    }

    n_samples = 398
    seed = 42
    stepper_x_test = 0.01
    h_parzen = 0.18260573228641733

    pdf = PDF(default="MULTIVARIATE_1254")

    x_training, y_training = pdf.generate_training(n_samples=n_samples, seed=seed)

    # generate the data for plotting the pdf
    x_test, y_test = pdf.generate_test(stepper=stepper_x_test)

    print("NUMBER OF TRAINING DATA", len(x_training))
    print("NUMBER OF TEST DATA", len(x_test))

    pz_model = ParzenWindow_Model(h=h_parzen)

    # generate the id
    unique_id_gmm_target = generate_unique_id(
        [x_training, n_samples, h_parzen, seed], 5
    )

    gmm_target_x, gmm_target_y = generate_target_MLP_pnn(
        parzen_model=pz_model,
        X=x_training,
        bias=False,
        save_filename=f"MLP-h{h_parzen}_S{n_samples}.npz",
    )

    # print(gmm_target_x)

    # optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    optuna.logging.get_logger("optuna")
    study_name = f"Parzen Neural Network S={n_samples} senza dropout"  # Unique identifier of the study.
    storage_name = "sqlite:///test_v2_3.db".format(study_name)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        load_if_exists=True,
    )

    study.set_metric_names(["R2 score"])

    study.optimize(
        lambda trial: objective_MLP(
            trial, gmm_target_x, gmm_target_y, x_test, y_test, params, DEVICE
        ),
        n_trials=300,
        timeout=None,
        n_jobs=2,
        show_progress_bar=True,
    )

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
