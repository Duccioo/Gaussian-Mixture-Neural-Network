from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
import numpy as np
import os

# ---
from utils.data_manager import load_dataset, save_dataset


# def GaussianMixtureModel(
#     n_components=4, seed=None, search=None, parameters=None, n_jobs=-1, init_params="random", **kwargs
# ):
#     """Make the Gaussian Mixture model

#     Parameters:
#     ----------
#         n_components (int, optional): _description_. Defaults to 4.
#         seed (_type_, optional): _description_. Defaults to None.
#         covariance_type: Literal['full', 'tied', 'diag', 'spherical'] = "full",
#         tol: Float = 0.001,
#         reg_covar: Float = 0.000001,
#         max_iter: Int = 100,
#         n_init: Int = 1,
#         init_params: Literal['kmeans', 'k-means++', 'random', 'random_from_data'] = "kmeans",
#         weights_init: ArrayLike | None = None,
#         means_init: ArrayLike | None = None,
#         precisions_init: ArrayLike | None = None,

#     Returns:
#     ----------
#         _type_: _description_
#     """

#     if seed is not None:
#         random_state = np.random.RandomState(seed)
#     else:
#         random_state = None

#     if search == "gridsearch" or parameters is not None:
#         model = GaussianMixture(n_components=n_components, init_params=init_params, random_state=random_state)
#         model = GridSearchCV(
#             model,
#             parameters,
#             cv=5,
#             verbose=3,
#             n_jobs=n_jobs,
#         )

#     else:
#         model = GaussianMixture(
#             n_components=n_components, init_params=init_params, random_state=random_state, **kwargs
#         )

#     return model


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

        print(Y.shape)
        print(X.shape)

        if save_filename is not None:
            save_dataset((X, Y), save_filename)

    return X, Y


if __name__ == "__main__":
    pass
