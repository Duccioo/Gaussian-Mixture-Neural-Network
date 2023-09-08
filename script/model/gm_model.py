from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
import numpy as np
import os

# ---
from utils.data_manager import load_dataset, save_dataset


def generate_target_MLP(gm_model, X, bias: bool = False, save_filename: str or None = None):
    # try to load the target data:
    if save_filename is not None and os.path.isfile(save_filename):
        X, Y = load_dataset(file=save_filename)
    else:
        Y = []
        # print(X.shape)
        for indx, sample in enumerate(X):
            if bias == False:
                X_1 = np.delete(X, indx, axis=0)

            else:
                X_1 = X

            gm_model.fit(X_1)
            Y.append(np.exp(gm_model.score_samples(sample.reshape(-1, X_1.shape[1]))))
        Y = np.array(Y).reshape(-1, 1)

        # print(Y.shape)
        # print(X.shape)

        if save_filename is not None:
            save_dataset((X, Y), save_filename)

    return X, Y


if __name__ == "__main__":
    pass
