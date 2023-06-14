# Gaussian Mixture Neural Network

A model that combine Gaussian mixture models with Neural Networks

Group:

1. Aparna Pindali Mana (matricola: 136597 )
2. Duccio Meconcelli (matricola: 134039 )

## Summary of the Project:

The project consists in studying a new algorithm not covered in class which, along the lines of Parzen Neural Networks, brings together the statistical approach and machine learning approaches.

The new algorithm consists in using the Gaussian Mixture Model (GMM) instead of the Parzen Window to generate the targets for the Neural Network.
Then the goal is to estimate the PDF of a certain unlabeled dataset.

The project must follow these points:

1. find a model for the NN (MLP: hard coding or finding a simulator)
2. find a model for GMM (hard coding or finding a simulator)
3. generate the dataset: composed of 100 examples/points taken randomly from an exponential distribution

4. carry out the experiments, producing as a result a graph that compares the true PDF of the exponential distribution and the one approximated by the GMM and the GMM+NN.
   Each experiment differs according to the number of components per GNN:
   4.a) 4 components for GMM: estimate the PDF only with the GMM and with the new machine (GMM + NN)
   4.b) 8 components for GMM: estimate the PDF only with the GMM and with the new machine (GMM + NN)
   4.c) 16 components for GMM: estimate the PDF only with the GMM and with the new machine (GMM + NN)
   4.d) 32 components for GMM: estimate the PDF only with the GMM and with the new machine (GMM + NN)

5. do one last experiment: choose the best model of the GMM+NN and check the differences between the unbiased and biased models. Do the same considerations of the Parzen Neural Network apply in this case too?

6. finally, a report on the activity carried out is expected in order to produce a scientific paper type text.
   The report must be structured in the following chapters:
   6.a) title
   6.b) abstracts
   6.c) introduction
   6.d) explanation of the algorithm
   6.e) the experiments and the results through the plots
   6.f) personal conclusions






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
