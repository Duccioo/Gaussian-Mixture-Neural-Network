# Experiment Details GNN MULTIVARIATE_1254S200
> from experiment with GNN
> on 2024-06-04 17-00
## Metrics:
                                                                                                    
| type   | r2           | mse          | max_error    | ise          | kl           | evs          |
|--------|--------------|--------------|--------------|--------------|--------------|--------------|
| Target | 0.0067070199 | 0.0027798158 | 0.1853515872 | 0.0041697237 | 0.1083834561 | 0.0637528129 |
| Model  | 0.4022       | 0.0018       | 0.1459       | 0.1823       | 0.1023       | 0.4315       |
                                                                                                    
## Plot Prediction

<img src="pdf_5e1b5db2.png">

## Loss Plot

<img src="loss_5e1b5db2.png">

## Training Metric Plot

<img src="train_metric_5e1b5db2.png">

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
| n_samples_training | 150   |
| n_samples_test     | 9865  |
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

                                                                                   
| KEY             | VALUE                                                         |
|-----------------|---------------------------------------------------------------|
| dropout         | 0.0                                                           |
| hidden_layer    | [[9, Tanh()], (20, Sigmoid()), (34, Sigmoid()), (26, Tanh())] |
| last_activation | lambda                                                        |
                                                                                   
</details>

<details><summary>Model Architecture </summary>

NeuralNetworkModular(
  (dropout): Dropout(p=0.0, inplace=False)
  (output_layer): Linear(in_features=26, out_features=1, bias=True)
  (last_activation): AdaptiveSigmoid(
    (sigmoid): Sigmoid()
  )
  (layers): ModuleList(
    (0): Linear(in_features=1, out_features=9, bias=True)
    (1): Linear(in_features=9, out_features=20, bias=True)
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

