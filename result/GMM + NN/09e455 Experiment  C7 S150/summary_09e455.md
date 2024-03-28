# Experiment Details Experiment  C7 S150
> from experiment with GMM + NN
> on 2024-03-28 12-46
## Metrics:
                                                                   
| type   | r2     | mse    | max_error | ise    | kl     | evs    |
|--------|--------|--------|-----------|--------|--------|--------|
| Target | 0.009  | 0.0028 | 0.1973    | 0.0028 | 0.141  | 0.0111 |
| Model  | 0.8485 | 0.0005 | 0.0886    | 0.0047 | 0.0231 | 0.8588 |
                                                                   
## Plot Prediction

<img src="C:\Users\mecon\Documents\GitHub\Gaussian-Mixture-Neural-Network\script\utils\..\..\result\GMM + NN\09e455 Experiment  C7 
S150\pdf_09e455.png">

## Loss Plot

<img src="C:\Users\mecon\Documents\GitHub\Gaussian-Mixture-Neural-Network\script\utils\..\..\result\GMM + NN\09e455 Experiment  C7 
S150\loss_09e455.png">

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
| n_components | 7      |
| n_init       | 30     |
| max_iter     | 100    |
| init_params  | kmeans |
| random_state | 42     |
                         
</details>

## Model
> using model GMM + NN
#### Model Params:
<details><summary>All Params used in the model </summary>

                                                  
| KEY             | VALUE                        |
|-----------------|------------------------------|
| dropout         | 0.0                          |
| hidden_layer    | [(64, ReLU()), (26, Tanh())] |
| last_activation | lambda                       |
                                                  
</details>

<details><summary>Model Architecture </summary>

LitModularNN(
  (neural_netowrk_modular): NeuralNetworkModular(
    (dropout): Dropout(p=0.0, inplace=False)
    (output_layer): Linear(in_features=26, out_features=1, bias=True)
    (last_activation): AdaptiveSigmoid(
      (sigmoid): Sigmoid()
    )
    (layers): ModuleList(
      (0): Linear(in_features=1, out_features=64, bias=True)
      (1): Linear(in_features=64, out_features=26, bias=True)
      (2): AdaptiveSigmoid(
        (sigmoid): Sigmoid()
      )
    )
    (activation): ModuleList(
      (0): ReLU()
      (1): Tanh()
    )
  )
)
</details>

## Training
<details><summary>All Params used for the training </summary>

                              
| KEY           | VALUE      |
|---------------|------------|
| epochs        | 700        |
| batch_size    | 10         |
| loss_type     | huber_loss |
| optimizer     | Adam       |
| learning_rate | 0.0023067  |
                              
</details>

