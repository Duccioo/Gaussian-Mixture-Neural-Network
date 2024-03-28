# Experiment Details Experiment  C9 S150
> from experiment with GMM + NN
> on 2024-03-28 15-25
## Metrics:
                                                                                                      
| type   | r2            | mse          | max_error    | ise          | kl           | evs           |
|--------|---------------|--------------|--------------|--------------|--------------|---------------|
| Target | -0.6556899756 | 0.0047554565 | 0.2342893504 | 0.0047554565 | 0.2620650644 | -0.6413040565 |
| Model  | -0.5795       | 0.0049       | 0.1218       | 0.0486       | 0.162        | 0.0           |
                                                                                                      
## Plot Prediction

<img src="C:\Users\mecon\Documents\GitHub\Gaussian-Mixture-Neural-Network\script\utils\..\..\result\GMM + NN\a6b230 Experiment  C9 
S150\pdf_a6b230.png">

## Loss Plot

<img src="C:\Users\mecon\Documents\GitHub\Gaussian-Mixture-Neural-Network\script\utils\..\..\result\GMM + NN\a6b230 Experiment  C9 
S150\loss_a6b230.png">

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
| seed               | 42    |
| n_samples_training | 100   |
| n_samples_test     | 988   |
| n_samples_val      | 50    |
| notes              |       |
                              
## Target
- Using GMM + NN Target
<details><summary>All Params used in the model for generate the target for the MLP </summary>

                         
| KEY          | VALUE  |
|--------------|--------|
| n_components | 9      |
| n_init       | 20     |
| max_iter     | 40     |
| init_params  | kmeans |
| random_state | 42     |
                         
</details>

## Model
> using model GMM + NN
#### Model Params:
<details><summary>All Params used in the model </summary>

                                                                             
| KEY             | VALUE                                                   |
|-----------------|---------------------------------------------------------|
| dropout         | 0.0                                                     |
| hidden_layer    | [(44, ReLU()), (24, ReLU()), (22, Tanh()), (8, ReLU())] |
| last_activation | lambda                                                  |
                                                                             
</details>

<details><summary>Model Architecture </summary>

LitModularNN(
  (neural_netowrk_modular): NeuralNetworkModular(
    (dropout): Dropout(p=0.0, inplace=False)
    (output_layer): Linear(in_features=8, out_features=1, bias=True)
    (last_activation): AdaptiveSigmoid(
      (sigmoid): Sigmoid()
    )
    (layers): ModuleList(
      (0): Linear(in_features=1, out_features=44, bias=True)
      (1): Linear(in_features=44, out_features=24, bias=True)
      (2): Linear(in_features=24, out_features=22, bias=True)
      (3): Linear(in_features=22, out_features=8, bias=True)
      (4): AdaptiveSigmoid(
        (sigmoid): Sigmoid()
      )
    )
    (activation): ModuleList(
      (0-1): 2 x ReLU()
      (2): Tanh()
      (3): ReLU()
    )
  )
)
</details>

## Training
<details><summary>All Params used for the training </summary>

                                
| KEY           | VALUE        |
|---------------|--------------|
| epochs        | 750          |
| batch_size    | 52           |
| loss_type     | huber_loss   |
| optimizer     | Adam         |
| learning_rate | 0.0014401389 |
                                
</details>

