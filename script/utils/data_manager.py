import numpy as np
import os
from attrs import define, field
from scipy.stats import logistic, expon

# ---
from .utils import check_base_dir, generate_unique_id
from .config import BASE_DATA_DIR
from .config import MULTIVARIATE_1254, EXPONENTIAL_06


def save_dataset(X, file: str = None, base_dir: str = None):
    if base_dir is not None and os.path.exists(base_dir) and file is not None and not os.path.isfile(file):
        file = os.path.join(base_dir, file)

    if isinstance(X, tuple) and len(X) > 1:
        if file is not None:
            print(f"Saving on file { file }")
            np.savez(file, input=X[0], output=X[1], allow_pickle=True)
    else:
        if file is not None:
            print(f"Saving on file { file }")
            np.save(file, X, allow_pickle=True)


def load_dataset(file: str = None, base_dir: str = None):
    if base_dir is not None and os.path.exists(base_dir) and file is not None and not os.path.isfile(file):
        file = os.path.join(base_dir, file)

    if file is not None and os.path.isfile(file):
        print("loading from file")
        dataset = np.load(file, allow_pickle=True)

        X = dataset["input"]
        y = dataset["output"]

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
    """
    This function calculates the probability density function (PDF) for different types of distributions, such as logistic and exponential.
    It takes in the type of distribution, parameters for the distribution, weight, random state, and input data.
    It then returns the sample and the PDF value.
    """
    sample = 0
    if type in ["logistic", "log"]:
        mean, scale = params["mean"], params["scale"]
        if X_input is not None:
            # pdf = weight * logistic.pdf(X_input, loc=mean, scale=scale)
            z = np.exp(-(X_input - mean) / scale)
            pdf = weight * (z / scale) * np.exp(-z)
        else:
            sample = random_state.logistic(mean, scale, size=1)
            # pdf = weight * logistic.pdf(sample[0], loc=mean, scale=scale)
            z = np.exp(-(sample - mean) / scale)
            pdf = weight * (z / scale) * np.exp(-z)

    elif type in ["exponential", "exp", "expon"]:
        scale = params.get("scale") or params.get("mean") or params.get("rate")
        negative = 1 if (scale >= 0) else -1
        scale = abs(scale)

        shift = params.get("offset") or params.get("shift") or params.get("translation")
        if shift == None or shift == False:
            shift = 0
        if X_input is not None:
            # print(weight)
            pdf = weight * expon.pdf(negative * X_input, loc=shift, scale=scale)
            # pdf = np.multiply(
            #     weight, expon.pdf(negative * X_input, loc=shift, scale=scale)
            # )
            # pdf = weight * np.exp(negative * X_input + shift) * scale
        else:
            sample = negative * (random_state.exponential(scale=scale, size=1) + shift)
            pdf = weight * expon.pdf(negative * sample[0], scale=scale, loc=-shift)
            # pdf = weight * np.exp(scale * sample[0] + shift) * scale

    else:
        raise ValueError("Invalid PDF type. Supported types: 'exponential', 'logistic'")
    return sample, pdf


# @nb.njit
def generate_points_in_grid(start, end, step, dimensions):
    if isinstance(start, (int, float)):
        start = [start] * dimensions
    if isinstance(end, (int, float)):
        end = [end] * dimensions

    ranges = [np.arange(s, e + step, step) for s, e in zip(start, end)]
    grid = np.meshgrid(*ranges, indexing="ij")

    points = np.column_stack([g.flatten() for g in grid])

    return points


@define(slots=True)
class PDF:
    name: str = field(default="no-name-set")
    params: list = field(factory=list)
    dimension: int = field(default=1)
    default: str = field(default=None)

    training_X: np.ndarray = field(init=True, default=np.array(None))
    training_Y: np.ndarray = field(init=True, default=np.array(None))
    test_X: np.ndarray = field(init=True, default=np.array(None))
    test_Y: np.ndarray = field(init=True, default=np.array(None))
    validation_X: np.ndarray = field(init=True, default=np.array(None))
    validation_Y: np.ndarray = field(init=True, default=np.array(None))

    n_samples_training: int = field(init=True, default=0)
    n_samples_test: int = field(init=True, default=0)
    n_samples_validation: int = field(init=True, default=0)

    base_dir: str = field(init=True, default=check_base_dir(BASE_DATA_DIR))
    unique_id_training: str = field(init=True, default="00000")
    unique_id_test: str = field(init=True, default="00000")

    def __init__(self, params: list = [], default=None, name=None):
        if default == "MULTIVARIATE_1254":
            params = MULTIVARIATE_1254
            if name is None:
                name = default
        elif default == "EXPONENTIAL_06":
            params = EXPONENTIAL_06
            if name is None:
                name = default

        if isinstance(params, list) == False:
            if params.get("weight") == None:
                params["weight"] = 1
            params = [[params]]

        elif isinstance(params[0], list) == False:
            params = [params]

        for d, dim in enumerate(params):
            if len(dim) == 1 and dim[0].get("weight") == None:
                params[d][0]["weight"] = 1

        dimension = len(params)
        self.__attrs_init__(name, params, dimension, default)

    def generate_training(
        self,
        n_samples,
        seed=None,
        save_filename=None,
        base_dir=None,
        scope=(-float("inf"), float("inf")),
    ):
        # --- loading part ---
        # generate the id
        self.unique_id_training = generate_unique_id([self.params, n_samples, seed], 5)

        # check if a saved training file exists:
        if base_dir is None:
            base_dir = self.base_dir

        if save_filename is not None:
            save_filename_t = save_filename.split(".")[0]
            save_filename_t = save_filename_t + "_" + self.unique_id_training + ".npz"
            training_X, training_Y = load_dataset(file=save_filename_t, base_dir=base_dir)

        if save_filename is not None and training_X is not None and training_Y is not None:
            self.training_X, self.training_Y = training_X, training_Y
            self.n_samples_training = len(self.training_X)
            return self.training_X, self.training_Y

        #  --- generating part ---
        elif self.default is not None:
            # print("provo a generare dataset...")
            self.load_default(n_samples=n_samples)
            self.n_samples_training = len(self.training_X)
            return self.training_X, self.training_Y

        else:
            if seed is not None:
                random_state = np.random.default_rng(seed)
            else:
                random_state = np.random.default_rng()
            samples = np.empty((n_samples, len(self.params)))
            fake_Y = np.zeros((n_samples, len(self.params)))
            for d, params_dim in enumerate(self.params):
                for i in range(len(samples)):
                    mode = random_state.choice(len(params_dim), p=[elem["weight"] for elem in params_dim])
                    sample, fake_Y1 = calculate_pdf(
                        params_dim[mode]["type"],
                        params_dim[mode],
                        params_dim[mode]["weight"],
                        random_state,
                    )

                    # check is the sample is outside the range (scope):
                    while sample < scope[0] or sample > scope[1]:
                        sample, fake_Y1 = calculate_pdf(
                            params_dim[mode]["type"],
                            params_dim[mode],
                            params_dim[mode]["weight"],
                            random_state,
                        )

                    fake_Y[i, d] += fake_Y1
                    samples[i, d] = sample[0]

            self.training_X = np.array(samples).reshape(-1, 1)
            self.training_Y = fake_Y[:, 0]
            for d in range(1, len(self.params)):
                self.training_Y *= fake_Y[:, d]

            self.training_Y = self.training_Y.reshape(self.training_Y.shape[0], 1)

            if save_filename is not None:
                save_dataset(
                    (self.training_X, self.training_Y),
                    save_filename_t,
                    base_dir=base_dir,
                )
            self.n_samples_training = len(self.training_X)
            return self.training_X, self.training_Y

    def load_default(self, n_samples):
        if self.default == "MULTIVARIATE_1254":
            filename = os.path.join(self.base_dir, "default", "MULTIVARIATE_1254.txt")
            data_loaded = np.empty((n_samples, 2))

        with open(filename, "r") as file:
            print(f"loading dataset from {filename}")
            lines = file.readlines()
            try:
                data_loaded = [
                    (
                        float(line.strip().split(" ")[0]),
                        float(line.strip().split(" ")[1]),
                    )
                    for line in lines
                ][0:n_samples]
            except:
                print(f"Number of samples requests ({n_samples}) bigger than {self.default} dataset!")

        self.training_X = np.array(data_loaded)[:, 0].reshape(-1, 1)
        self.training_Y = np.array(data_loaded)[:, 1].reshape(-1, 1)
        self.n_samples_training = len(self.training_X)
        return self.training_X, self.training_Y

    def generate_test(
        self,
        range_limit: tuple = None,
        stepper: float = 0.001,
        save_filename=None,
        base_dir=None,
    ):
        """
        Generate test data for the model using specified parameters and return the test data.

        Parameters:
            range_limit (tuple): A tuple containing the lower and upper limits for generating test data.
            stepper (float): The step size for generating the test data.
            save_filename (str): The filename to save the test data.
            base_dir (str): The base directory where the data will be saved.

        Returns:
            tuple: A tuple containing the generated test data X and Y.
        """
        # generate the id
        self.unique_id_test = generate_unique_id([self.params, range_limit, stepper], 5)

        # check if a saved training file exists:
        if base_dir is None:
            base_dir = self.base_dir

        if save_filename is not None:
            save_filename_t = save_filename.split(".")[0]
            save_filename_t = save_filename_t + "_" + self.unique_id_test + ".npz"
            test_X, test_Y = load_dataset(file=save_filename_t, base_dir=base_dir)

        if save_filename is not None and test_X is not None and test_Y is not None:
            self.test_X, self.test_Y = test_X, test_Y
            self.n_samples_test = len(self.test_X)
            return self.test_X, self.test_Y

        else:
            # check if the method generate_dataset is already loaded and range limit is not specified:
            if range_limit is None and self.training_X.size != 0:
                range_limit = (np.min(self.training_X), np.max(self.training_X))

            self.test_X = generate_points_in_grid(
                range_limit[0], range_limit[1], stepper, dimensions=len(self.params)
            )
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
            self.n_samples_test = len(self.test_X)
            return self.test_X, self.test_Y

    def generate_validation(self, percent: float = 0.0, n_samples: int = 0):
        """
        Generates a validation set from the training set.

        Parameters:
            percent (float, optional): The percentage of the training set to use for validation. Defaults to 0.0.
            n_samples (int, optional): The number of samples to use for validation. Defaults to 0.

        Returns:
            tuple: A tuple containing the validation input data (validation_X) and validation target data (validation_Y).
        """
        if self.n_samples_training == 0:
            print(
                "impossible to generate validation set from training set because there is no training data yet"
            )
            print("Try to launch '.generate_training()' first")
        else:
            if n_samples == 0 and percent != 0:
                n_val = int(percent * self.n_samples_training)
            elif n_samples == 0 and percent == 0:
                n_val = 0
            elif n_samples != 0:
                n_val = n_samples

            n_train = self.n_samples_training - n_val

            self.validation_X = self.training_X[n_train:]
            self.validation_Y = self.training_Y[n_train:]

            self.training_X = self.training_X[:n_train]
            self.training_Y = self.training_Y[:n_train]

            self.n_samples_training -= len(self.validation_X)
            self.n_samples_validation = len(self.validation_X)

        return self.validation_X, self.validation_Y


def load_multivariate_dataset(n_samples, seed, stepper_x_test=0.01):
    """
    Generate a multivariate dataset for training and testing.

    Parameters:
    - n_samples: int, number of samples to generate for training
    - seed: int, seed for random number generation
    - stepper_x_test: float, optional, increment value for generating test data along X-axis

    Returns:
    - X_train: array, training dataset inputs
    - Y_train: array, training dataset outputs
    - X_test: array, test dataset inputs
    - Y_test: array, test dataset outputs
    """
    pdf = PDF(default="MULTIVARIATE_1254")
    X_train, Y_train = pdf.generate_training(n_samples=n_samples, seed=seed)

    # generate the data for plotting the pdf
    X_test, Y_test = pdf.generate_test(stepper=stepper_x_test)

    return X_train, Y_train, X_test, Y_test, pdf


if __name__ == "__main__":
    prova = PDF(
        [
            [
                {"type": "logistic", "mean": 20, "scale": 0.5, "weight": 0.4},
                {"type": "logistic", "mean": 10, "scale": 4, "weight": 0.4},
                {"type": "logistic", "mean": 17, "scale": 1, "weight": 0.2},
            ]
        ]
    )

    prova.generate_training(5000, seed=42)
    prova.generate_test((-10, 50))

    X = prova.training_X
    y = prova.training_Y

    print("X =", X[0:5])
    print("y =", y[0:5], "\n")

    # plt.plot(
    #     prova.test_X,
    #     prova.test_Y,
    #     label="Test PDF (MLP)",
    #     color="green",
    # )
    # plt.hist(prova.training_X, bins=32, density=True, alpha=0.5, label="Data", color="dimgray")
    # plt.title("Testing plot")
    # plt.xlabel("X")
    # plt.ylabel("Probability Density")
    # plt.legend()
    # plt.show()
