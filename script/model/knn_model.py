import numpy as np
from attrs import define, field


@define(slots=True)
class KNN_Model:
    k1: float = field(default=0.0, init=True)
    kn: float = field(default=0.0, init=True)
    training: np.ndarray = field(init=True, default=np.array(None))

    def fit(self, training: np.ndarray):
        _kn = int(self.k1 * np.sqrt(len(training)))

        if _kn > len(training):
            self.kn = self.k1
            print(f"kn ({_kn}) is too large to fit in the training set ({len(training)})\nSetting kn = {self.kn}")
        else:
            self.kn = _kn

        self.training = training.ravel()

    def predict(self, test: np.ndarray):
        pdf_predicted_knn = [self.kn_nn_estimation(self.training, point, self.kn) for point in test]
        return pdf_predicted_knn

    @staticmethod
    def kn_nn_estimation(data: np.ndarray, point: float = 0.0, kn: float = 1.0) -> float:
        n = len(data)
        distances = np.abs(data - point)
        sorted_distances = np.sort(distances)
        kn_density = (kn / n) / (sorted_distances[kn] * 2)
        return kn_density


if __name__ == "__main__":
    n_samples = 100
    seed = 42
    data = np.random.exponential(scale=1.0, size=n_samples)
    x_test = np.linspace(0, 5, 100)
    np.random.seed(seed)

    knn = KNN_Model(k1=4)
    knn.fit(data)

    pdf_estimated = knn.predict(x_test)
    pdf_true = np.exp(-x_test)
