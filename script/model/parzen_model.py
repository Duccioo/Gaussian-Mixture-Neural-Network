import numpy as np
from attrs import define, field


@define(slots=True)
class ParzenWindow_Model:
    h: float = field(default=0.0, init=True)
    kn: float = field(default=0.0, init=True)
    training: np.ndarray = field(init=True, default=np.array(None))

    def fit(self, training: np.ndarray):
        self.training = training.ravel()

    def predict(self, test: np.ndarray):
        pdf_predicted = [self.parzen_window(self.training, point, self.h) for point in test]
        return pdf_predicted

    @staticmethod
    def parzen_window(data, point, h):
        n = len(data)
        window = 1 / (np.sqrt(2 * np.pi) * h) * np.exp(-0.5 * ((point - data) / h) ** 2)
        return np.sum(window) / (n)


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
