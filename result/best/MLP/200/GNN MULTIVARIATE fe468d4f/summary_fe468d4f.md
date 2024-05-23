# Experiment Details Experiment  C11 S250
> from experiment with GNN
> on 2024-05-23 19-22
## Metrics:
                                                                                                   
| type   | r2          | mse          | max_error    | ise         | kl           | evs           |
|--------|-------------|--------------|--------------|-------------|--------------|---------------|
| Target | -0.27649104 | 0.0033382155 | 0.2110728598 | 0.006676431 | 0.1144209164 | -0.2729617878 |
| Model  | 0.9235      | 0.0002       | 0.0704       | 0.0241      | 0.0118       | 0.924         |
                                                                                                   
## Plot Prediction

<img src="pdf_fe468d4f.png">

## Loss Plot

<img src="loss_fe468d4f.png">

## Training Metric Plot

<img src="train_metric_fe468d4f.png">

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
| n_samples_val      | 50    |
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
| random_state | 45        |
                            
</details>

## Model
> using model GNN
#### Model Params:
<details><summary>All Params used in the model </summary>

                                                                                    
| KEY             | VALUE                                                          |
|-----------------|----------------------------------------------------------------|
| dropout         | 0.0                                                            |
| hidden_layer    | [[39, Tanh()], (20, Sigmoid()), (34, Sigmoid()), (26, Tanh())] |
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
    (0): Linear(in_features=1, out_features=39, bias=True)
    (1): Linear(in_features=39, out_features=20, bias=True)
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

                                        
| KEY           | VALUE                |
|---------------|----------------------|
| epochs        | 540                  |
| batch_size    | 52                   |
| loss_type     | huber_loss           |
| optimizer     | Adam                 |
| learning_rate | 0.003454947958915411 |
                                        
</details>

