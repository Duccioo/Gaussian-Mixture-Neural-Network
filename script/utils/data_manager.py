import numpy as np
import os
from attrs import define, field
from scipy.stats import logistic, expon


# ---
from .utils import check_base_dir, generate_unique_id

BASE_DATA_DIR = ["..", "..", "data"]


def save_dataset(X, file: str or None = None, base_dir: str or None = None):
    if base_dir is not None and os.path.exists(base_dir) and file is not None and not os.path.isfile(file):
        file = os.path.join(base_dir, file)

    if file is not None:
        print(f"Saving on file { file }...")
        np.save(file, X, allow_pickle=True)


def load_dataset(file: str = None, base_dir: str = None):
    if base_dir is not None and os.path.exists(base_dir) and file is not None and not os.path.isfile(file):
        file = os.path.join(base_dir, file)

    if file is not None and os.path.isfile(file):
        print("loading from file")
        dataset = np.load(file, allow_pickle=True)
        X = dataset[:][0]
        y = dataset[:][1]
        return X, y
    else:
        return None, None


def calculate_pdf(
    type: str = "logistic",
    params: dict = {"mean": 0.0, "scale": 1.0},
    weight: float = 1.0,
    random_state=False,
    X_input=None,
):
    sample = 0
    if type in ["logistic", "log"]:
        mean, scale = params["mean"], params["scale"]
        if X_input is not None:
            pdf = weight * logistic.pdf(X_input, loc=mean, scale=scale)
        else:
            sample = random_state.logistic(mean, scale, size=1)
            pdf = weight * logistic.pdf(sample[0], loc=mean, scale=scale)

    elif type in ["exponential", "exp", "expon"]:
        scale = params.get("scale") or params.get("mean") or params.get("rate")
        if X_input is not None:
            pdf = weight * expon.pdf(X_input, scale=scale)
        else:
            sample = random_state.exponential(scale=scale, size=1)
            pdf = weight * expon.pdf(sample[0], scale=scale)

    else:
        raise ValueError("Invalid PDF type. Supported types: 'exponential', 'logistic'")

    return sample, pdf


@define(slots=True)
class PDF:
    params: list = field(factory=list)
    dimension: int = field(default=1)

    training_X: np.ndarray = field(init=True, default=np.array(None))
    training_Y: np.ndarray = field(init=True, default=np.array(None))
    test_X: np.ndarray = field(init=True, default=np.array(None))
    test_Y: np.ndarray = field(init=True, default=np.array(None))

    base_dir: str = field(init=True, default=check_base_dir(BASE_DATA_DIR))
    unique_id_training: str = field(init=True, default="00000")
    unique_id_test: str = field(init=True, default="00000")

    def __init__(self, params: list = []):
        if isinstance(params, list) == False:
            if params.get("weight") == None:
                params["weight"] = 1
            params = [[params]]

        elif isinstance(params[0], list) == False:
            params = [params]

        self.dimension = len(params)

        for d, dim in enumerate(params):
            if len(dim) == 1 and dim[0].get("weight") == None:
                params[d][0]["weight"] = 1
        self.__attrs_init__(params)

    def generate_training(self, n_samples, seed=None, save_filename=None, base_dir=None):
        # --- saving part ---
        # generate the id
        self.unique_id_training = generate_unique_id([self.params, n_samples, seed], 5)

        # check if a saved training file exists:
        if base_dir is None:
            base_dir = self.base_dir

        if save_filename is not None:
            save_filename_t = save_filename.split(".")[0]
            save_filename_t = save_filename_t + "_" + self.unique_id_training + ".npy"
            training_X, training_Y = load_dataset(file=save_filename_t, base_dir=base_dir)

        if save_filename is not None and training_X is not None and training_Y is not None:
            self.training_X, self.training_Y = training_X, training_Y
            return self.training_X, self.training_Y

        #  --- generating part ---
        else:
            if seed is not None:
                random_state = np.random.default_rng(seed)

            samples = np.empty((n_samples, len(self.params)))
            fake_Y = np.zeros((n_samples, len(self.params)))

            for d, params_dim in enumerate(self.params):
                for i in range(n_samples):
                    mode = random_state.choice(len(params_dim), p=[elem["weight"] for elem in params_dim])

                    sample, fake_Y1 = calculate_pdf(
                        params_dim[mode]["type"], params_dim[mode], params_dim[mode]["weight"], random_state
                    )

                    fake_Y[i, d] += fake_Y1
                    samples[i, d] = sample[0]

            self.training_X = np.array(samples)

            self.training_Y = fake_Y[:, 0]
            for d in range(1, len(self.params)):
                self.training_Y *= fake_Y[:, d]

            self.training_Y = self.training_Y.reshape(self.training_Y.shape[0], 1)

            if save_filename is not None:
                save_dataset((self.training_X, self.training_Y), save_filename_t, base_dir=base_dir)
            return self.training_X, self.training_Y

    def generate_test(self, range_limit: tuple = (0, 50), stepper: float = 0.001, save_filename=None, base_dir=None):
        # generate the id
        self.unique_id_test = generate_unique_id([self.params, range_limit, stepper], 5)

        # check if a saved training file exists:
        if base_dir is None:
            base_dir = self.base_dir

        if save_filename is not None:
            save_filename_t = save_filename.split(".")[0]
            save_filename_t = save_filename_t + "_" + self.unique_id_test + ".npy"
            test_X, test_Y = load_dataset(file=save_filename_t, base_dir=base_dir)

        if save_filename is not None and test_X is not None and test_Y is not None:
            self.test_X, self.test_Y = test_X, test_Y
            return self.test_X, self.test_Y

        else:
            self.test_X = generate_points_in_grid(range_limit[0], range_limit[1], stepper, dimensions=len(self.params))
            fake_Y = np.zeros((len(self.test_X), len(self.params)))

            for d, params_dim in enumerate(self.params):
                for pdf_info in params_dim:
                    pdf_type = pdf_info["type"]
                    weight = pdf_info["weight"]

                    _, fake_Y1 = calculate_pdf(pdf_type, pdf_info, weight, X_input=self.test_X[:, d])
                    fake_Y[:, d] += fake_Y1

            self.test_Y = fake_Y[:, 0]
            for d in range(1, len(self.params)):
                self.test_Y *= fake_Y[:, d]

            self.test_Y = self.test_Y.reshape((self.test_Y.shape[0], 1))

            if save_filename is not None:
                save_dataset((self.test_X, self.test_Y), save_filename_t, base_dir=base_dir)
            return self.test_X, self.test_Y


def generate_points_in_grid(start, end, step, dimensions):
    if isinstance(start, (int, float)):
        start = [start] * dimensions
    if isinstance(end, (int, float)):
        end = [end] * dimensions
    ranges = [np.arange(s, e + step, step) for s, e in zip(start, end)]
    grid = np.meshgrid(*ranges)
    points = np.vstack([g.ravel() for g in grid]).T
    return points


if __name__ == "__main__":
    prova = PDF(
        [
            {"type": "log", "mean": 5, "scale": 1, "weight": 0.3},
            {"type": "log", "mean": 15, "scale": 0.5, "weight": 0.4},
            {"type": "log", "mean": 30, "scale": 2, "weight": 0.3},
        ]
    )

    prova.generate_training(5000, seed=42, save_filename="prova_training_2.npy")
    prova.generate_test((0, 10), save_filename="prova_test_2.npy")

    X = prova.training_X
    y = prova.training_Y

    print("X =", X[0:5])
    print("y =", y[0:5], "\n")
