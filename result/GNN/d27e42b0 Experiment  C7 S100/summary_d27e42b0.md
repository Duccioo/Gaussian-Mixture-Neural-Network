# Experiment Details Experiment  C7 S100
> from experiment with GNN
> on 2024-05-22 15-21
## Metrics:
                                                                                                    
| type   | r2           | mse          | max_error    | ise          | kl           | evs          |
|--------|--------------|--------------|--------------|--------------|--------------|--------------|
| Target | 0.0077618145 | 0.0028498968 | 0.1973881326 | 0.0028498968 | 0.1215398778 | 0.0094143431 |
| Model  | 0.696        | 0.0009       | 0.0648       | 0.0912       | 0.0481       | 0.7768       |
                                                                                                    
## Plot Prediction

<img src="pdf_d27e42b0.png">

## Loss Plot

<img src="loss_d27e42b0.png">

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
| seed               | 98    |
| n_samples_training | 100   |
| n_samples_test     | 9840  |
| n_samples_val      | 0     |
| notes              |       |
                              
## Target
- Using GNN Target
<details><summary>All Params used in the model for generate the target for the MLP </summary>

                            
| KEY          | VALUE     |
|--------------|-----------|
| n_components | 7         |
| n_init       | 10        |
| max_iter     | 30        |
| init_params  | k-means++ |
| random_state | 62        |
                            
</details>

## Model
> using model GNN
#### Model Params:
<details><summary>All Params used in the model </summary>

                                                                                 
| KEY             | VALUE                                                       |
|-----------------|-------------------------------------------------------------|
| dropout         | 0.0                                                         |
| hidden_layer    | [(84, Sigmoid()), (26, Tanh()), (24, Tanh()), (54, ReLU())] |
| last_activation | None                                                        |
                                                                                 
</details>

<details><summary>Model Architecture </summary>

NeuralNetworkModular(
  (dropout): Dropout(p=0.0, inplace=False)
  (output_layer): Linear(in_features=54, out_features=1, bias=True)
  (layers): ModuleList(
    (0): Linear(in_features=1, out_features=84, bias=True)
    (1): Linear(in_features=84, out_features=26, bias=True)
    (2): Linear(in_features=26, out_features=24, bias=True)
    (3): Linear(in_features=24, out_features=54, bias=True)
  )
  (activation): ModuleList(
    (0): Sigmoid()
    (1-2): 2 x Tanh()
    (3): ReLU()
  )
)
</details>

## Training
<details><summary>All Params used for the training </summary>

                            
| KEY           | VALUE    |
|---------------|----------|
| epochs        | 360      |
| batch_size    | 40       |
| loss_type     | mse_loss |
| optimizer     | Adam     |
| learning_rate | 0.00266  |
                            
</details>

