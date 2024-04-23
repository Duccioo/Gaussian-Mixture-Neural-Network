# Experiment Details Experiment  C9 S150
> from experiment with GNN
> on 2024-04-23 11-20
## Metrics:
                                                                                                      
| type   | r2            | mse          | max_error    | ise          | kl           | evs           |
|--------|---------------|--------------|--------------|--------------|--------------|---------------|
| Target | -0.3638147702 | 0.0039171354 | 0.1973909116 | 0.0039171354 | 0.2092746026 | -0.3527095809 |
| Model  | 0.7657        | 0.0007       | 0.0585       | 0.0072       | 0.0338       | 0.8714        |
                                                                                                      
## Plot Prediction

<img src="pdf_520c7e01.png">

## Loss Plot

<img src="loss_520c7e01.png">

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
| seed               | 56    |
| n_samples_training | 100   |
| n_samples_test     | 988   |
| n_samples_val      | 50    |
| notes              |       |
                              
## Target
- Using GNN Target
<details><summary>All Params used in the model for generate the target for the MLP </summary>

                            
| KEY          | VALUE     |
|--------------|-----------|
| n_components | 9         |
| n_init       | 40        |
| max_iter     | 10        |
| init_params  | k-means++ |
| random_state | 51        |
                            
</details>

## Model
> using model GNN
#### Model Params:
<details><summary>All Params used in the model </summary>

                                                  
| KEY             | VALUE                        |
|-----------------|------------------------------|
| dropout         | 0.0                          |
| hidden_layer    | [(58, Tanh()), (38, ReLU())] |
| last_activation | lambda                       |
                                                  
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
      (0): Linear(in_features=1, out_features=58, bias=True)
      (1): Linear(in_features=58, out_features=38, bias=True)
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

                                         
| KEY           | VALUE                 |
|---------------|-----------------------|
| epochs        | 950                   |
| batch_size    | 4                     |
| loss_type     | huber_loss            |
| optimizer     | RMSprop               |
| learning_rate | 0.0006734373910499807 |
                                         
</details>

