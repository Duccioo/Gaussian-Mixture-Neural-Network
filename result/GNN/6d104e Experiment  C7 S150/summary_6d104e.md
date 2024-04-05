# Experiment Details Experiment  C7 S150
> from experiment with GMM + NN
> on 2024-03-28 15-51
## Metrics:
                                                                                                  
| type   | r2          | mse         | max_error    | ise         | kl            | evs          |
|--------|-------------|-------------|--------------|-------------|---------------|--------------|
| Target | 0.008992835 | 0.002846361 | 0.1972617186 | 0.002846361 | 35.8587011729 | 0.0111483534 |
| Model  | 0.7835      | 0.0007      | 0.0845       | 0.0067      | 449.833       | 0.7841       |
                                                                                                  
## Plot Prediction

<img src="C:\Users\mecon\Documents\GitHub\Gaussian-Mixture-Neural-Network\script\utils\..\..\result\GMM + NN\6d104e Experiment  C7 
S150\pdf_6d104e.png">

## Loss Plot

<img src="C:\Users\mecon\Documents\GitHub\Gaussian-Mixture-Neural-Network\script\utils\..\..\result\GMM + NN\6d104e Experiment  C7 
S150\loss_6d104e.png">

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

                            
| KEY          | VALUE     |
|--------------|-----------|
| n_components | 7         |
| n_init       | 100       |
| max_iter     | 100       |
| init_params  | k-means++ |
| random_state | 42        |
                            
</details>

## Model
> using model GMM + NN
#### Model Params:
<details><summary>All Params used in the model </summary>

                                                  
| KEY             | VALUE                        |
|-----------------|------------------------------|
| dropout         | 0.0                          |
| hidden_layer    | [(42, Tanh()), (48, Tanh())] |
| last_activation | lambda                       |
                                                  
</details>

<details><summary>Model Architecture </summary>

LitModularNN(
  (neural_netowrk_modular): NeuralNetworkModular(
    (dropout): Dropout(p=0.0, inplace=False)
    (output_layer): Linear(in_features=48, out_features=1, bias=True)
    (last_activation): AdaptiveSigmoid(
      (sigmoid): Sigmoid()
    )
    (layers): ModuleList(
      (0): Linear(in_features=1, out_features=42, bias=True)
      (1): Linear(in_features=42, out_features=48, bias=True)
      (2): AdaptiveSigmoid(
        (sigmoid): Sigmoid()
      )
    )
    (activation): ModuleList(
      (0-1): 2 x Tanh()
    )
  )
)
</details>

## Training
<details><summary>All Params used for the training </summary>

                              
| KEY           | VALUE      |
|---------------|------------|
| epochs        | 220        |
| batch_size    | 4          |
| loss_type     | huber_loss |
| optimizer     | RMSprop    |
| learning_rate | 0.002207   |
                              
</details>

