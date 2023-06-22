# Gaussian Mixture Neural Network

A model that combines Gaussian mixture models with Neural Networks

## Summary of the Project:

<details>
  <summary>The project consists in studying a new algorithm not covered in class which, along the lines of Parzen Neural Networks, brings together the statistical approach and machine learning approaches.</summary>
  
  The new algorithm consists in using the Gaussian Mixture Model (GMM) instead of the Parzen Window to generate the targets for the Neural Network.
Then the goal is to estimate the PDF of a certain unlabeled dataset.

The project must follow these points:

- [x] find a model for the NN (MLP: hard coding or finding a simulator)
- [x] find a model for GMM (hard coding or finding a simulator)
- [x] generate the dataset: composed of 100 examples/points taken randomly from an exponential distribution

- [ ] carry out the experiments, producing as a result a graph that compares the true PDF of the exponential distribution and the one approximated by the GMM and the GMM+NN.
      Each experiment differs according to the number of components per GNN:

  - 4 components for GMM: estimate the PDF only with the GMM and with the new machine (GMM + NN)
  - 8 components for GMM: estimate the PDF only with the GMM and with the new machine (GMM + NN)
  - 16 components for GMM: estimate the PDF only with the GMM and with the new machine (GMM + NN)
  - 32 components for GMM: estimate the PDF only with the GMM and with the new machine (GMM + NN)

- [ ] do one last experiment: choose the best model of the GMM+NN and check the differences between the unbiased and biased models. Do the same considerations of the Parzen Neural Network apply in this case too?

- [ ] finally, a report on the activity carried out is expected in order to produce a scientific paper type text.
      The report must be structured in the following chapters:
  - title
  - abstracts
  - introduction
  - explanation of the algorithm
  - the experiments and the results through the plots
  - personal conclusions

</details>

### remarks:

1. remark: the MLP must be chosen so as to maximize the result for each experiment, it is therefore expected to do different experiments to choose the best hyperparameters (the comparison can also be done only graphically)

2. pytorch is quite recommended

3. To draw the graph of GNN+NN, do we give it the same 100 examples as input?

   > No. AS we did for the demonstrations during the course, create a set of
   > equally-spaced datapoints at regular intervals, e.g. 0.01, 0.02, 0.03, ...
   > and plot the corresponding outputs from the pdf-estimator at hand as a
   > "continuous" line.

4. Does the output of the MLP need to be normalized to make it a PDF?

   > No. A sigmoid with adaptive amplitude lambda would be best, but that
   > feature will hardly be made available to you by any simulator you decide
   > to use. Effective. I recommend you either go for a standard ReLU (whose
   > output range [0, +INFINITY) matches the range of any pdf), or even a plain
   > linear activation function but in the latter case you need to force to 0.0
   > any possible negative outputs at test-tne

5. Should the input to GNN + MLP be normalized around 0 to get better results?

   > You can do that but that is not needed, experiece witt the exponential pdf
   > shows the neywork can cope withthe expected range of thenon-normalized
   > inputs.

6. Do we compare the various experiments only graphically?
   > Yes. A quantitative comparison would involve computing the Integrated
   > Squared error (ISE) or other simiar measure of distance between he
   > estimated pdf and the true pdf (you can do that you feel like it, of
   > course!)

## To do:

- [ ] do the 4 experiments, changing the number of parameters and the parameters of the gridsarch
- [ ] check out the differences between the GMM biased and unbiased
- [ ] Implement the saving plots for the experiments
- [ ] check the correctness of the ISE score function
- [ ] maybe implement some options to pass with the command line
- [ ] maybe implement a gridsearch also for the GMM algorithm

## File Structure:

- ### `data` folder:

  Contains the save files of the training and test data.
  This files are automatically generated and saved when run the algorithm.

- ### `script` folder:

  It contains all the code for running the project.

  - #### `model/nn_model.py`: the model for the Neural Network part of the project. Implements the
  - #### `model/gm_model.py`:

  - #### `data_manager.py`: the data manager for the

  - #### `utils.py`: the utilities for the

  - #### `main.py`: the main function for the project

- ### `other` folder: some random code created at the beginning of the project

## Installation

In order to run all the script you need to run the following:

1. Clone the project:

   ```bash
   git clone https://github.com/Duccioo/Gaussian-Mixture-Neural-Network.git
   ```

2. install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Run Locally

On the repo folder:

```bash
python script/main.py
```

## Results:

## Authors

- [@AparnaPindali](https://github.com/AparnaPindali)
- [@duccioo](https://github.com/Duccioo)

## License

[MIT](https://choosealicense.com/licenses/mit/)
