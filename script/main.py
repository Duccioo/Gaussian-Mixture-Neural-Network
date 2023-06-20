from data_manager import generate_training, generate_test, save_dataset, load_dataset
from utils import plot_AllInOne, plot_histo
from model.nn_model import NerualNetwork_model
from model.gm_model import GaussianMixtureModel_bias


def main():
    
    x_training, y_training = generate_training()
    save_dataset(x_training, "X_training")
    
    x_test, y_test = generate_test()
    save_dataset(x_test, "X_test")
    save_dataset(y_test, "y_test")
    
    
    
    
    model = GaussianMixtureModel_bias()
    model.fit(x_training)
    
    plot_histo(x_training)


if __name__ == "__main__":
    main()
