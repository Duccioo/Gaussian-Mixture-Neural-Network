# Experiment Details Experiment  C25 S250
> from experiment with GNN
> on 2024-05-23 16-28
## Metrics:
                                                                                                     
| type   | r2           | mse          | max_error    | ise          | kl            | evs          |
|--------|--------------|--------------|--------------|--------------|---------------|--------------|
| Target | 0.6689535849 | 0.0762224021 | 1.2035861518 | 0.1524448042 | 0.0462480869  | 0.6689539511 |
| Model  | 0.9978       | 0.0004       | 0.0766       | 0.0116       | 10000000000.0 | 0.9978       |
                                                                                                     
## Plot Prediction

<img src="pdf_6b41f25c.png">

## Loss Plot

<img src="loss_6b41f25c.png">

## Training Metric Plot

<img src="train_metric_6b41f25c.png">

## Dataset

<details><summary>PDF set as default <b>EXPONENTIAL_06</b></summary>

#### Dimension 1
                               
| type        | rate | weight |
|-------------|------|--------|
| exponential | 0.6  | 1      |
                               
</details>
                              
| KEY                | VALUE |
|--------------------|-------|
| dimension          | 1     |
| seed               | 18    |
| n_samples_training | 200   |
| n_samples_test     | 3210  |
| n_samples_val      | 50    |
| notes              |       |
                              
## Target
- Using GNN Target
<details><summary>All Params used in the model for generate the target for the MLP </summary>

                         
| KEY          | VALUE  |
|--------------|--------|
| n_init       | 90     |
| max_iter     | 40     |
| n_components | 25     |
| random_state | 64     |
| init_params  | random |
                         
</details>

## Model
> using model GNN
#### Model Params:
<details><summary>All Params used in the model </summary>

                                    
| KEY             | VALUE          |
|-----------------|----------------|
| dropout         | 0.0            |
| hidden_layer    | [(32, ReLU())] |
| last_activation | None           |
                                    
</details>

<details><summary>Model Architecture </summary>

NeuralNetworkModular(
  (dropout): Dropout(p=0.0, inplace=False)
  (output_layer): Linear(in_features=32, out_features=1, bias=True)
  (layers): ModuleList(
    (0): Linear(in_features=1, out_features=32, bias=True)
  )
  (activation): ModuleList(
    (0): ReLU()
  )
)
</details>

## Training
<details><summary>All Params used for the training </summary>

                              
| KEY           | VALUE      |
|---------------|------------|
| learning_rate | 0.00519    |
| epochs        | 170        |
| loss_type     | huber_loss |
| optimizer     | Adam       |
| batch_size    | 78         |
                              
</details>

