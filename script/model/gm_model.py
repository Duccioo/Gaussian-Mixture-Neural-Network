import numpy as np
import os
from sklearn.mixture import GaussianMixture
from rich.progress import track

# ---
from utils.data_manager import load_dataset, save_dataset


def gen_target_with_gm(
    gm_model: GaussianMixture,
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

            gm_model.fit(X_1)
            Y.append(np.exp(gm_model.score_samples(sample.reshape(-1, X_1.shape[1]))))
        Y = np.array(Y).reshape(-1, 1)

        if save_filename is not None:
            save_dataset((X, Y), save_filename)

    return X, Y


if __name__ == "__main__":
    pass
