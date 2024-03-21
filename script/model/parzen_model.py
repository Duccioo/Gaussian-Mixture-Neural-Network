import numpy as np
from attrs import define, field
from rich.progress import track
import os

# ---
from utils.data_manager import load_dataset, save_dataset


@define(slots=True)
class ParzenWindow_Model:
    h: float = field(default=0.0, init=True)
    kn: float = field(default=0.0, init=True)
    training: np.ndarray = field(init=True, default=np.array(None))

    def fit(self, training: np.ndarray):
        self.training = training.ravel()

    def predict(self, test: np.ndarray):
        pdf_predicted = [
            self.parzen_window(self.training, point, self.h) for point in test
        ]
        return pdf_predicted

    @staticmethod
    def parzen_window(data, point, h):
        n = len(data)
        window = 1 / (np.sqrt(2 * np.pi) * h) * np.exp(-0.5 * ((point - data) / h) ** 2)
        return np.sum(window) / (n)


def gen_target_with_parzen(
    parzen_model: ParzenWindow_Model,
    X: np.ndarray,
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


if __name__ == "__main__":
    n_samples = 100
    seed = 42
    data = np.random.exponential(scale=1.0, size=n_samples)
    x_test = np.linspace(0, 5, 100)
    np.random.seed(seed)

    knn = ParzenWindow_Model(h=0.1)
    knn.fit(data)

    pdf_estimated = knn.predict(x_test)
    pdf_true = np.exp(-x_test)
