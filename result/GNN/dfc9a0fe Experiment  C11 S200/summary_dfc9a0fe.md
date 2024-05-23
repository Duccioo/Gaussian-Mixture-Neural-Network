# Experiment Details Experiment  C11 S200
> from experiment with GNN
> on 2024-05-22 14-31
## Metrics:
                                                                                                     
| type   | r2            | mse          | max_error   | ise          | kl           | evs           |
|--------|---------------|--------------|-------------|--------------|--------------|---------------|
| Target | -0.1455245186 | 0.0029957184 | 0.178781144 | 0.0059914368 | 0.1102181256 | -0.1199206766 |
| Model  | 0.1486        | 0.0027       | 0.1235      | 0.2686       | 0.1274       | 0.2209        |
                                                                                                     
## Plot Prediction

<img src="pdf_dfc9a0fe.png">

## Loss Plot

<img src="loss_dfc9a0fe.png">

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
| seed               | 31    |
| n_samples_training | 200   |
| n_samples_test     | 9973  |
| n_samples_val      | 0     |
| notes              |       |
                              
## Target
- Using GNN Target
<details><summary>All Params used in the model for generate the target for the MLP </summary>

                            
| KEY          | VALUE     |
|--------------|-----------|
| n_components | 11        |
| n_init       | 20        |
| max_iter     | 100       |
| init_params  | k-means++ |
| random_state | 37        |
                            
</details>

## Model
> using model GNN
#### Model Params:
<details><summary>All Params used in the model </summary>

                                                                                    
| KEY             | VALUE                                                          |
|-----------------|----------------------------------------------------------------|
| dropout         | 0.0                                                            |
| hidden_layer    | [(28, Tanh()), (20, Sigmoid()), (34, Sigmoid()), (26, Tanh())] |
| last_activation | lambda                                                         |
                                                                                    
</details>

<details><summary>Model Architecture </summary>

NeuralNetworkModular(
  (dropout): Dropout(p=0.0, inplace=False)
  (output_layer): Linear(in_features=26, out_features=1, bias=True)
  (last_activation): AdaptiveSigmoid(
    (sigmoid): Sigmoid()
  )
  (layers): ModuleList(
    (0): Linear(in_features=1, out_features=28, bias=True)
    (1): Linear(in_features=28, out_features=20, bias=True)
    (2): Linear(in_features=20, out_features=34, bias=True)
    (3): Linear(in_features=34, out_features=26, bias=True)
    (4): AdaptiveSigmoid(
      (sigmoid): Sigmoid()
    )
  )
  (activation): ModuleList(
    (0): Tanh()
    (1-2): 2 x Sigmoid()
    (3): Tanh()
  )
)
</details>

## Training
<details><summary>All Params used for the training </summary>

                                         
| KEY           | VALUE                 |
|---------------|-----------------------|
| epochs        | 540                   |
| batch_size    | 52                    |
| loss_type     | huber_loss            |
| optimizer     | Adam                  |
| learning_rate | 0.0011300000000000001 |
                                         
</details>

