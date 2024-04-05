# Experiment Details Experiment  C10 S150
> from experiment with GMM + NN
> on 2024-03-28 17-24
## Metrics:
                                                                                                     
| type   | r2            | mse          | max_error   | ise          | kl           | evs           |
|--------|---------------|--------------|-------------|--------------|--------------|---------------|
| Target | -0.7765733449 | 0.0051026565 | 0.234289343 | 0.0051026565 | 0.2847815132 | -0.7624816532 |
| Model  | 0.7858        | 0.0007       | 0.1327      | 0.0066       | 0.0381       | 0.8267        |
                                                                                                     
## Plot Prediction

<img src="C:\Users\mecon\Documents\GitHub\Gaussian-Mixture-Neural-Network\script\utils\..\..\result\GMM + NN\4a7598 Experiment  C10
S150\pdf_4a7598.png">

## Loss Plot

<img src="C:\Users\mecon\Documents\GitHub\Gaussian-Mixture-Neural-Network\script\utils\..\..\result\GMM + NN\4a7598 Experiment  C10
S150\loss_4a7598.png">

## Dataset

<details><summary>PDF set as default <b>MULTIVARIATE_1254</b></summary>

#### Dimension 1
                                      
| type        | rate | weight |      |
|-------------|------|--------|------|
| exponential | 1    | 0.2    |      |
| logistic    | 4    | 0.8    | 0.25 |
| logistic    | 5.5  | 0.7    | 0.3  |
| exponential | -1   | 0.25   | -10  |
                                      
</details>
                              
| KEY                | VALUE |
|--------------------|-------|
| dimension          | 1     |
| seed               | 19    |
| n_samples_training | 100   |
| n_samples_test     | 988   |
| n_samples_val      | 50    |
| notes              |       |
                              
## Target
- Using GMM + NN Target
<details><summary>All Params used in the model for generate the target for the MLP </summary>

                            
| KEY          | VALUE     |
|--------------|-----------|
| n_components | 10        |
| n_init       | 100       |
| max_iter     | 80        |
| init_params  | k-means++ |
| random_state | 19        |
                            
</details>

## Model
> using model GMM + NN
#### Model Params:
<details><summary>All Params used in the model </summary>

                                                                
| KEY             | VALUE                                      |
|-----------------|--------------------------------------------|
| dropout         | 0.0                                        |
| hidden_layer    | [(74, ReLU()), (56, Tanh()), (38, Tanh())] |
| last_activation | lambda                                     |
                                                                
</details>

<details><summary>Model Architecture </summary>

LitModularNN(
  (neural_netowrk_modular): NeuralNetworkModular(
    (dropout): Dropout(p=0.0, inplace=False)
    (output_layer): Linear(in_features=38, out_features=1, bias=True)
    (last_activation): AdaptiveSigmoid(
      (sigmoid): Sigmoid()
    )
    (layers): ModuleList(
      (0): Linear(in_features=1, out_features=74, bias=True)
      (1): Linear(in_features=74, out_features=56, bias=True)
      (2): Linear(in_features=56, out_features=38, bias=True)
      (3): AdaptiveSigmoid(
        (sigmoid): Sigmoid()
      )
    )
    (activation): ModuleList(
      (0): ReLU()
      (1-2): 2 x Tanh()
    )
  )
)
</details>

## Training
<details><summary>All Params used for the training </summary>

                               
| KEY           | VALUE       |
|---------------|-------------|
| epochs        | 540         |
| batch_size    | 26          |
| loss_type     | mse_loss    |
| optimizer     | Adam        |
| learning_rate | 0.000874345 |
                               
</details>

