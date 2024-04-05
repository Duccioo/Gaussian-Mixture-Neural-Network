# Experiment Details Experiment  C4 S150
> from experiment with GMM + NN
> on 2024-03-28 02-39
## Metrics:
                                                                   
| type   | r2     | mse    | max_error | ise    | kl     | evs    |
|--------|--------|--------|-----------|--------|--------|--------|
| Target | 0.1007 | 0.0026 | 0.1959    | 0.0026 | 0.0884 | 0.1158 |
| Model  | 0.6283 | 0.0012 | 0.1435    | 0.0114 | 0.0626 | 0.6306 |
                                                                   
## Plot Prediction

<img src="/Users/duccio/Documents/GitHub/Gaussian-Mixture-Neural-Network/script/utils/../../result/GMM + NN/e32692 Experiment  C4 S150/pdf_e32692.png">

## Loss Plot

<img src="/Users/duccio/Documents/GitHub/Gaussian-Mixture-Neural-Network/script/utils/../../result/GMM + NN/e32692 Experiment  C4 S150/loss_e32692.png">

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
| seed               | 82    |
| n_samples_training | 100   |
| n_samples_test     | 988   |
| n_samples_val      | 50    |
| notes              |       |
                              
## Target
- Using GMM + NN Target
<details><summary>All Params used in the model for generate the target for the MLP </summary>

                         
| KEY          | VALUE  |
|--------------|--------|
| n_components | 4      |
| n_init       | 11     |
| max_iter     | 684    |
| init_params  | kmeans |
| random_state | 82     |
                         
</details>

## Model
> using model GMM + NN
#### Model Params:
<details><summary>All Params used in the model </summary>

                                                  
| KEY             | VALUE                        |
|-----------------|------------------------------|
| dropout         | 0.0                          |
| hidden_layer    | [(56, Tanh()), (60, ReLU())] |
| last_activation | lambda                       |
                                                  
</details>

<details><summary>Model Architecture </summary>

LitModularNN(
  (neural_netowrk_modular): NeuralNetworkModular(
    (dropout): Dropout(p=0.0, inplace=False)
    (output_layer): Linear(in_features=60, out_features=1, bias=True)
    (last_activation): AdaptiveSigmoid(
      (sigmoid): Sigmoid()
    )
    (layers): ModuleList(
      (0): Linear(in_features=1, out_features=56, bias=True)
      (1): Linear(in_features=56, out_features=60, bias=True)
      (2): AdaptiveSigmoid(
        (sigmoid): Sigmoid()
      )
    )
    (activation): ModuleList(
      (0): Tanh()
      (1): ReLU()
    )
  )
)
</details>

## Training
<details><summary>All Params used for the training </summary>

                                
| KEY           | VALUE        |
|---------------|--------------|
| epochs        | 400          |
| batch_size    | 20           |
| loss_type     | huber_loss   |
| optimizer     | Adam         |
| learning_rate | 0.0052784027 |
                                
</details>

