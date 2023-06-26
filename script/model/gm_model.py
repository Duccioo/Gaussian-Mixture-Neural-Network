from sklearn.mixture import GaussianMixture
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import numpy as np


def GaussianMixtureModel(n_components=4, seed=None, search=None, parameters=None, n_jobs=-1, **kwargs):
    """Make the Gaussian Mixture model

    Parameters:
    ----------
        n_components (int, optional): _description_. Defaults to 4.
        seed (_type_, optional): _description_. Defaults to None.
        covariance_type: Literal['full', 'tied', 'diag', 'spherical'] = "full",
        tol: Float = 0.001,
        reg_covar: Float = 0.000001,
        max_iter: Int = 100,
        n_init: Int = 1,
        init_params: Literal['kmeans', 'k-means++', 'random', 'random_from_data'] = "kmeans",
        weights_init: ArrayLike | None = None,
        means_init: ArrayLike | None = None,
        precisions_init: ArrayLike | None = None,

    Returns:
    ----------
        _type_: _description_
    """

    if seed is not None:
        random_state = np.random.seed(seed)
    else:
        random_state = None

    if search == "gridsearch" or parameters is not None:
        model = GaussianMixture(n_components=n_components, init_params="random", random_state=random_state)
        model = GridSearchCV(
            model,
            parameters,
            cv=5,
            verbose=3,
            n_jobs=n_jobs,
        )

    else:
        model = GaussianMixture(n_components=n_components, init_params="random", random_state=random_state, **kwargs)

    return model


def GaussianMixtureModel_unbias():
    pass


if __name__ == "__main__":
    pass
