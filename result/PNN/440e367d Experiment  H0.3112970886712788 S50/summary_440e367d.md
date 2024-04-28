# Experiment Details Experiment  H0.3112970886712788 S50
> from experiment with PNN
> on 2024-04-23 13-51
## Metrics:
                                                                                                 
| type   | r2           | mse        | max_error    | ise         | kl           | evs          |
|--------|--------------|------------|--------------|-------------|--------------|--------------|
| Target | 0.5337236814 | 0.00135187 | 0.0963450018 | 0.000675935 | 0.0675486109 | 0.5338930259 |
| Model  | 0.0388       | 0.0027     | 0.0925       | 0.0257      | 0.12         | 0.2707       |
                                                                                                 
## Plot Prediction

<img src="pdf_440e367d.png">

## Loss Plot

<img src="loss_440e367d.png">

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
| n_samples_test     | 953   |
| n_samples_val      | 0     |
| notes              |       |
                              
## Target
- Using PNN Target
<details><summary>All Params used in the model for generate the target for the MLP </summary>

                            
| KEY | VALUE              |
|-----|--------------------|
| h   | 0.3112970886712788 |
                            
</details>

## Model
> using model PNN
#### Model Params:
<details><summary>All Params used in the model </summary>

                                                                                 
| KEY             | VALUE                                                       |
|-----------------|-------------------------------------------------------------|
| dropout         | 0.0                                                         |
| hidden_layer    | [(24, ReLU()), (22, Tanh()), (64, ReLU()), (64, Sigmoid())] |
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
      (0): Linear(in_features=1, out_features=24, bias=True)
      (1): Linear(in_features=24, out_features=22, bias=True)
      (2): Linear(in_features=22, out_features=64, bias=True)
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
| epochs        | 110                   |
| batch_size    | 14                    |
| loss_type     | huber_loss            |
| optimizer     | Adam                  |
| learning_rate | 0.0006600995607244531 |
                                         
</details>

