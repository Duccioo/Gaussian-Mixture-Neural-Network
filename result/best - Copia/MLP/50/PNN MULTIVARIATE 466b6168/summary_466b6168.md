# Experiment Details Experiment  H0.24853179838613948 S100
> from experiment with PNN
> on 2024-05-23 16-39
## Metrics:
                                                                                                   
| type   | r2           | mse          | max_error    | ise         | kl           | evs          |
|--------|--------------|--------------|--------------|-------------|--------------|--------------|
| Target | 0.4132451522 | 0.0017011721 | 0.1017419942 | 0.000850586 | 0.0902992766 | 0.4144137769 |
| Model  | 0.8297       | 0.0005       | 0.0607       | 0.0454      | 0.0313       | 0.8341       |
                                                                                                   
## Plot Prediction

<img src="pdf_466b6168.png">

## Loss Plot

<img src="loss_466b6168.png">

## Training Metric Plot

<img src="train_metric_466b6168.png">

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
| seed               | 64    |
| n_samples_training | 50    |
| n_samples_test     | 9520  |
| n_samples_val      | 50    |
| notes              |       |
                              
## Target
- Using PNN Target
<details><summary>All Params used in the model for generate the target for the MLP </summary>

                             
| KEY | VALUE               |
|-----|---------------------|
| h   | 0.24853179838613948 |
                             
</details>

## Model
> using model PNN
#### Model Params:
<details><summary>All Params used in the model </summary>

                                                                             
| KEY             | VALUE                                                   |
|-----------------|---------------------------------------------------------|
| dropout         | 0.0                                                     |
| hidden_layer    | [(52, Tanh()), (6, Tanh()), (58, Tanh()), (38, ReLU())] |
| last_activation | lambda                                                  |
                                                                             
</details>

<details><summary>Model Architecture </summary>

NeuralNetworkModular(
  (dropout): Dropout(p=0.0, inplace=False)
  (output_layer): Linear(in_features=38, out_features=1, bias=True)
  (last_activation): AdaptiveSigmoid(
    (sigmoid): Sigmoid()
  )
  (layers): ModuleList(
    (0): Linear(in_features=1, out_features=52, bias=True)
    (1): Linear(in_features=52, out_features=6, bias=True)
    (2): Linear(in_features=6, out_features=58, bias=True)
    (3): Linear(in_features=58, out_features=38, bias=True)
    (4): AdaptiveSigmoid(
      (sigmoid): Sigmoid()
    )
  )
  (activation): ModuleList(
    (0-2): 3 x Tanh()
    (3): ReLU()
  )
)
</details>

## Training
<details><summary>All Params used for the training </summary>

                                          
| KEY           | VALUE                  |
|---------------|------------------------|
| learning_rate | 0.00043285400651533746 |
| epochs        | 370                    |
| loss_type     | huber_loss             |
| optimizer     | RMSprop                |
| batch_size    | 6                      |
                                          
</details>

