from sklearn.mixture import GaussianMixture
import numpy as np


def GaussianMixtureModel_bias(n_components=4, seed=None, max_iter=1000):
    if seed is not None:
        random_state = np.random.seed(seed)
    else:
        random_state = None

    model = GaussianMixture(
        n_components=n_components,
        init_params="random",
        covariance_type="full",
        max_iter=max_iter,
        n_init=10,
        random_state=random_state,
    )
    return model

def GaussianMixtureModel_unbias():
    pass


if __name__ == "__main__":
    pass
