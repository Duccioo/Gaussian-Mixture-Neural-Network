import torch.nn as nn

import argparse

from sklearn.mixture import GaussianMixture

# ---
from utils.data_manager import PDF
from model.nn_model import NeuralNetworkModular
from model.gm_model import gen_target_with_gm_parallel
from model.parzen_model import gen_target_with_parzen_parallel, ParzenWindow_Model
from utils.utils import set_seed, check_model_name
from utils.summary import Summary
from training_MLP import training, evaluation


def arg_parsing():
    # command line parsing
    parser = argparse.ArgumentParser(description="Project GNN")
    parser.add_argument("--pdf", type=str, default="exponential")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--samples", type=int, default=100)

    parser.add_argument("--show", action="store_true", default=False)

    # parser.add_argument("--mlp_targets", action="store_true", default=False)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--model", type=str, default="GNN")
    # parser.add_argument("--bias", action="store_true", default=False)

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = arg_parsing()

    model_type = args.model

    model_type, target_type = check_model_name(model_type)

    validation_samples = 50

    dataset_params = {
        "n_samples": int(args.samples) + validation_samples,
        "seed": 31,
        "target_type": target_type,
        "validation_size": validation_samples,
        # "test_range_limit": (0, 5),
    }

    gmm_target_params = {
        "n_components": 11,
        "n_init": 20,
        "max_iter": 100,
        "init_params": "k-means++",  # "k-means++" or "random" or "kmeans" or "random_from_data"
        "random_state": 45,
    }

    pw_target_params = {"h": 0.026604557869874756}

    mlp_params = {
        "dropout": 0.000,
        "hidden_layer": [
            [9, nn.Tanh()],
            (20, nn.Sigmoid()),
            (34, nn.Sigmoid()),
            (26, nn.Tanh()),
        ],
        "last_activation": "lambda",  # None or lambda
    }

    train_params = {
        "epochs": 540,
        "batch_size": 52,
        "loss_type": "huber_loss",  # "huber_loss" or "mse_loss"
        "optimizer": "Adam",  # "RMSprop" or "Adam"
        "learning_rate": 0.003454947958915411,
    }

    device = "cpu"

    set_seed(dataset_params["seed"])

    # choose the pdf for the experiment
    if args.pdf in ["exponential", "exp"]:
        pdf = PDF(default="EXPONENTIAL_06")
    else:
        pdf = PDF(default="MULTIVARIATE_1254")

    # load dataset
    pdf.generate_training(n_samples=dataset_params["n_samples"])
    pdf.generate_validation(n_samples=dataset_params["validation_size"])
    pdf.generate_test()

    if dataset_params["target_type"] == "GMM":
        print("generating target with GMM")
        gm_model = GaussianMixture(**gmm_target_params)
        _, target_y = gen_target_with_gm_parallel(gm_model, X=pdf.training_X, n_jobs=-1, progress_bar=True)
    else:
        print("generating target with Parzen")
        parzen_model = ParzenWindow_Model(**pw_target_params)
        _, target_y = gen_target_with_parzen_parallel(
            parzen_model, X=pdf.training_X, n_jobs=-1, progress_bar=True
        )

    model = NeuralNetworkModular(
        dropout=mlp_params["dropout"],
        hidden_layer=mlp_params["hidden_layer"],
        last_activation=mlp_params["last_activation"],
    )

    train_loss, val_loss, train_metrics, val_metrics = training(
        model,
        pdf.training_X,
        target_y,
        pdf.validation_X,
        pdf.validation_Y,
        train_params["learning_rate"],
        train_params["epochs"],
        train_params["batch_size"],
        train_params["optimizer"],
        train_params["loss_type"],
        device,
    )

    training_r2 = [r2_value["r2"] for r2_value in train_metrics]
    validation_r2 = [r2_value["r2"] for r2_value in val_metrics]

    # evaluate model
    metrics, pdf_predicted = evaluation(model, pdf.test_X, pdf.test_Y, device)

    # ----------------------------- SUMMARY -----------------------------
    # Creo l'oggetto che mi gestirà il salvataggio dell'esperimento e gli passo tutti i parametri
    experiment_name = f"{model_type} {pdf.name}"

    if dataset_params["target_type"] == "GMM":
        target_params = gmm_target_params
        # experiment_name += f" C{gmm_target_params['n_components']} "
    elif dataset_params["target_type"] == "PARZEN":
        target_params = pw_target_params
        # experiment_name += f""

    experiment_name += f"S{dataset_params['n_samples']}"

    summary_obj = Summary(
        experiment=experiment_name,
        model_type=model_type,
        pdf=pdf,
        dataset_params=dataset_params,
        model_params=mlp_params,
        target_params=target_params,
        train_params=train_params,
        overwrite=True,
    )

    summary_obj.calculate_metrics(pdf.training_Y, pdf.test_Y, pdf_predicted, target_y)
    print("*******************************")
    print("SUMMARY:")
    print("Experiment name: ", summary_obj.experiment_name)
    print("Model type: ", summary_obj.model_type)
    print("ID EXPERIMENT:", summary_obj.id_experiment)
    print("R2 score: ", summary_obj.model_metrics.get("r2"))
    print("KL divergence: ", summary_obj.model_metrics.get("kl"))
    print("Done!")
    summary_obj.plot_pdf(pdf.training_X, target_y, pdf.test_X, pdf.test_Y, pdf_predicted, show=args.show)
    summary_obj.plot_loss(train_loss, val_loss, loss_name=train_params["loss_type"], save=True)
    summary_obj.plot_training_score(training_r2, validation_r2, "R2", save=True)
    summary_obj.log_dataset()
    summary_obj.log_target()
    summary_obj.log_model(model=model)
    summary_obj.log_train_params()
    summary_obj.scoreboard()
