# Experiment Details Experiment  C4 S50
> from experiment with GNN
> on 2024-04-23 13-27
## Metrics:
                                                                                                   
| type   | r2           | mse         | max_error    | ise          | kl           | evs          |
|--------|--------------|-------------|--------------|--------------|--------------|--------------|
| Target | 0.2026401299 | 0.002311777 | 0.1657670344 | 0.0011558885 | 0.0825076338 | 0.2572101878 |
| Model  | 0.7255       | 0.0008      | 0.0882       | 0.0083       | 0.0512       | 0.7485       |
                                                                                                   
## Plot Prediction

<img src="pdf_212a68d0.png">

## Loss Plot

<img src="loss_212a68d0.png">

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
| seed               | 8     |
| n_samples_training | 50    |
| n_samples_test     | 985   |
| n_samples_val      | 50    |
| notes              |       |
                              
## Target
- Using GNN Target
<details><summary>All Params used in the model for generate the target for the MLP </summary>

                            
| KEY          | VALUE     |
|--------------|-----------|
| n_components | 4         |
| n_init       | 70        |
| max_iter     | 30        |
| init_params  | k-means++ |
| random_state | 27        |
                            
</details>

## Model
> using model GNN
#### Model Params:
<details><summary>All Params used in the model </summary>

                                                                                 
| KEY             | VALUE                                                       |
|-----------------|-------------------------------------------------------------|
| dropout         | 0.0                                                         |
| hidden_layer    | [(50, ReLU()), (50, Tanh()), (22, Tanh()), (44, Sigmoid())] |
| last_activation | lambda                                                      |
                                                                                 
</details>

<details><summary>Model Architecture </summary>

LitModularNN(
  (neural_netowrk_modular): NeuralNetworkModular(
    (dropout): Dropout(p=0.0, inplace=False)
    (output_layer): Linear(in_features=44, out_features=1, bias=True)
    (last_activation): AdaptiveSigmoid(
      (sigmoid): Sigmoid()
    )
    (layers): ModuleList(
      (0): Linear(in_features=1, out_features=50, bias=True)
      (1): Linear(in_features=50, out_features=50, bias=True)
      (2): Linear(in_features=50, out_features=22, bias=True)
      (3): Linear(in_features=22, out_features=44, bias=True)
      (4): AdaptiveSigmoid(
        (sigmoid): Sigmoid()
      )
    )
    (activation): ModuleList(
      (0): ReLU()
      (1-2): 2 x Tanh()
      (3): Sigmoid()
    )
  )
)
</details>

## Training
<details><summary>All Params used for the training </summary>

                             
| KEY           | VALUE     |
|---------------|-----------|
| epochs        | 770       |
| batch_size    | 34        |
| loss_type     | mse_loss  |
| optimizer     | RMSprop   |
| learning_rate | 0.0007619 |
                             
</details>

