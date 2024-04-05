# Experiment Details Experiment  H0.36197695 S150
> from experiment with Parzen Window + NN
> on 2024-03-28 02-26
## Metrics:
                                                                   
| type   | r2     | mse    | max_error | ise    | kl     | evs    |
|--------|--------|--------|-----------|--------|--------|--------|
| Target | 0.6679 | 0.001  | 0.1364    | 0.001  | 0.0284 | 0.7179 |
| Model  | 0.1389 | 0.0027 | 0.1348    | 0.0265 | 0.0981 | 0.3672 |
                                                                   
## Plot Prediction

<img src="/Users/duccio/Documents/GitHub/Gaussian-Mixture-Neural-Network/script/utils/../../result/Parzen Window + NN/adbb47 Experiment  H0.36197695 
S150/pdf_adbb47.png">

## Loss Plot

<img src="/Users/duccio/Documents/GitHub/Gaussian-Mixture-Neural-Network/script/utils/../../result/Parzen Window + NN/adbb47 Experiment  H0.36197695 
S150/loss_adbb47.png">

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
| seed               | 82    |
| n_samples_training | 100   |
| n_samples_test     | 988   |
| n_samples_val      | 50    |
| notes              |       |
                              
## Target
- Using Parzen Window + NN Target
<details><summary>All Params used in the model for generate the target for the MLP </summary>

                    
| KEY | VALUE      |
|-----|------------|
| h   | 0.36197695 |
                    
</details>

## Model
> using model Parzen Window + NN
#### Model Params:
<details><summary>All Params used in the model </summary>

                                                  
| KEY             | VALUE                        |
|-----------------|------------------------------|
| dropout         | 0.0                          |
| hidden_layer    | [(52, ReLU()), (40, ReLU())] |
| last_activation | lambda                       |
                                                  
</details>

<details><summary>Model Architecture </summary>

LitModularNN(
  (neural_netowrk_modular): NeuralNetworkModular(
    (dropout): Dropout(p=0.0, inplace=False)
    (output_layer): Linear(in_features=40, out_features=1, bias=True)
    (last_activation): AdaptiveSigmoid(
      (sigmoid): Sigmoid()
    )
    (layers): ModuleList(
      (0): Linear(in_features=1, out_features=52, bias=True)
      (1): Linear(in_features=52, out_features=40, bias=True)
      (2): AdaptiveSigmoid(
        (sigmoid): Sigmoid()
      )
    )
    (activation): ModuleList(
      (0-1): 2 x ReLU()
    )
  )
)
</details>

## Training
<details><summary>All Params used for the training </summary>

                                
| KEY           | VALUE        |
|---------------|--------------|
| epochs        | 20           |
| batch_size    | 2            |
| loss_type     | huber_loss   |
| optimizer     | RMSprop      |
| learning_rate | 0.0052784027 |
                                
</details>

