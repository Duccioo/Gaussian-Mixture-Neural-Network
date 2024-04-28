# Experiment Details Experiment  H0.30886195236981046 S50
> from experiment with PNN
> on 2024-04-23 13-49
## Metrics:
                                                                                                   
| type   | r2           | mse         | max_error    | ise          | kl           | evs          |
|--------|--------------|-------------|--------------|--------------|--------------|--------------|
| Target | 0.5308812538 | 0.001360111 | 0.0965411794 | 0.0006800555 | 0.0680912263 | 0.5310004091 |
| Model  | 0.1017       | 0.0027      | 0.1212       | 0.0271       | 0.1314       | 0.2024       |
                                                                                                   
## Plot Prediction

<img src="pdf_3b513638.png">

## Loss Plot

<img src="loss_3b513638.png">

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
| seed               | 6     |
| n_samples_training | 50    |
| n_samples_test     | 985   |
| n_samples_val      | 50    |
| notes              |       |
                              
## Target
- Using PNN Target
<details><summary>All Params used in the model for generate the target for the MLP </summary>

                             
| KEY | VALUE               |
|-----|---------------------|
| h   | 0.30886195236981046 |
                             
</details>

## Model
> using model PNN
#### Model Params:
<details><summary>All Params used in the model </summary>

                                                                                 
| KEY             | VALUE                                                       |
|-----------------|-------------------------------------------------------------|
| dropout         | 0.0                                                         |
| hidden_layer    | [(18, ReLU()), (24, Tanh()), (64, ReLU()), (64, Sigmoid())] |
| last_activation | lambda                                                      |
                                                                                 
</details>

<details><summary>Model Architecture </summary>

LitModularNN(
  (neural_netowrk_modular): NeuralNetworkModular(
    (dropout): Dropout(p=0.0, inplace=False)
    (output_layer): Linear(in_features=64, out_features=1, bias=True)
    (last_activation): AdaptiveSigmoid(
      (sigmoid): Sigmoid()
    )
    (layers): ModuleList(
      (0): Linear(in_features=1, out_features=18, bias=True)
      (1): Linear(in_features=18, out_features=24, bias=True)
      (2): Linear(in_features=24, out_features=64, bias=True)
      (3): Linear(in_features=64, out_features=64, bias=True)
      (4): AdaptiveSigmoid(
        (sigmoid): Sigmoid()
      )
    )
    (activation): ModuleList(
      (0): ReLU()
      (1): Tanh()
      (2): ReLU()
      (3): Sigmoid()
    )
  )
)
</details>

## Training
<details><summary>All Params used for the training </summary>

                                         
| KEY           | VALUE                 |
|---------------|-----------------------|
| epochs        | 100                   |
| batch_size    | 12                    |
| loss_type     | huber_loss            |
| optimizer     | Adam                  |
| learning_rate | 0.0007396434514115792 |
                                         
</details>

