import numpy as np

from sklearn.mixture import GaussianMixture
import argparse

# ---
from model.knn_model import KNN_Model
from model.parzen_model import ParzenWindow_Model
from utils.summary import Summary
from utils.data_manager import PDF
from utils.utils import check_model_name


def arg_parsing():
    # command line parsing
    parser = argparse.ArgumentParser(description="Project GNN")
    parser.add_argument("--pdf", type=str, default="default")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--show", action="store_true", default=False)
    parser.add_argument("--model", type=str, default="GMM")

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = arg_parsing()

    # select model type from "GMM" or "Parzen Window" or "KNN"
    model_type = args.model

    model_type, target_type = check_model_name(model_type)

    dataset_params = {
        "n_samples": args.samples,
        "seed": 37,
        "target_type": target_type,
        # "test_range_limit": (0, 5),
    }

    gm_model_params = {
        "random_state": 46,
        "init_params": "random_from_data",
        "max_iter": 90,
        "n_components": 5,
        "n_init": 60,
    }

    knn_model_params = {"k1": 1.5005508828032745, "kn": 23}

    parzen_window_params = {"h": 0.28293348425061676}

    # choose the pdf for the experiment
    if args.pdf in ["exponential", "exp"]:
        pdf = PDF(default="EXPONENTIAL_06")
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

    pdf.generate_training(
        n_samples=dataset_params["n_samples"],
        seed=dataset_params["seed"],
    )

    # generate the data for plotting the pdf
    pdf.generate_test(stepper=0.01)

    target_y = None
    target_params = None
    train_loss = None
    model_params: dict = {}

    # --------------------------------- PARZEN WINDOW -------------------------------------
    if model_type == "Parzen Window":
        model_params = parzen_window_params
        model = ParzenWindow_Model(h=parzen_window_params["h"])
        model.fit(training=pdf.training_X)
        pdf_predicted = model.predict(test=pdf.test_X)

    # --------------------------------- GMM -------------------------------------
    elif model_type == "GMM":
        model_params = gm_model_params
        model = GaussianMixture(**gm_model_params)
        model.fit(pdf.training_X, pdf.training_Y)
        # predict the pdf with GMM
        pdf_predicted = np.exp(model.score_samples(pdf.test_X))

    # --------------------------------- KNN -------------------------------------
    elif model_type == "KNN":
        model_params = knn_model_params
        model = KNN_Model(**knn_model_params)
        model.fit(pdf.training_X)
        pdf_predicted = model.predict(pdf.test_X)

    # ----------------------------- SUMMARY -----------------------------
    # Create an object that will save the experiment and pass it all the parameters
    experiment_name = f"{model_type} {pdf.name}"
    experiment_name += f" S{dataset_params['n_samples']}"

    summary = Summary(
        experiment=experiment_name,
        model_type=model_type,
        pdf=pdf,
        dataset_params=dataset_params,
        model_params=model_params,
        target_params=target_params,
        overwrite=True,
    )

    summary.calculate_metrics(pdf.training_Y, pdf.test_Y, pdf_predicted, target_y)
    print("*******************************")
    print("SUMMARY:")
    print("Experiment name: ", summary.experiment_name)
    print("Model type: ", summary.model_type)
    print("ID EXPERIMENT:", summary.id_experiment)
    print("R2 score: ", summary.model_metrics.get("r2"))
    print("KL divergence: ", summary.model_metrics.get("kl"))
    print("Done!")
    summary.plot_pdf(pdf.training_X, target_y, pdf.test_X, pdf.test_Y, pdf_predicted, show=args.show)
    summary.log_dataset()
    summary.log_target()
    summary.log_model(model=model)
    summary.log_train_params()
    summary.scoreboard()
