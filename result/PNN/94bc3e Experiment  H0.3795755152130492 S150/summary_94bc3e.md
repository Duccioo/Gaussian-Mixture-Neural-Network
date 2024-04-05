# Experiment Details Experiment  H0.3795755152130492 S150
> from experiment with GMM + NN
> on 2024-03-28 02-37
## Metrics:
                                                                   
| type   | r2     | mse    | max_error | ise    | kl     | evs    |
|--------|--------|--------|-----------|--------|--------|--------|
| Target | 0.6515 | 0.001  | 0.1375    | 0.001  | 0.0283 | 0.7103 |
| Model  | 0.7526 | 0.0008 | 0.1117    | 0.0076 | 0.037  | 0.771  |
                                                                   
## Plot Prediction

<img src="/Users/duccio/Documents/GitHub/Gaussian-Mixture-Neural-Network/script/utils/../../result/GMM + NN/94bc3e Experiment  H0.3795755152130492 
S150/pdf_94bc3e.png">

## Loss Plot

<img src="/Users/duccio/Documents/GitHub/Gaussian-Mixture-Neural-Network/script/utils/../../result/GMM + NN/94bc3e Experiment  H0.3795755152130492 
S150/loss_94bc3e.png">

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
- Using GMM + NN Target
<details><summary>All Params used in the model for generate the target for the MLP </summary>

                            
| KEY | VALUE              |
|-----|--------------------|
| h   | 0.3795755152130492 |
                            
</details>

## Model
> using model GMM + NN
#### Model Params:
<details><summary>All Params used in the model </summary>

                                                  
| KEY             | VALUE                        |
|-----------------|------------------------------|
| dropout         | 0.0                          |
| hidden_layer    | [(56, Tanh()), (60, ReLU())] |
| last_activation | lambda                       |
                                                  
</details>

<details><summary>Model Architecture </summary>

LitModularNN(
  (neural_netowrk_modular): NeuralNetworkModular(
    (dropout): Dropout(p=0.0, inplace=False)
    (output_layer): Linear(in_features=60, out_features=1, bias=True)
    (last_activation): AdaptiveSigmoid(
      (sigmoid): Sigmoid()
    )
    (layers): ModuleList(
      (0): Linear(in_features=1, out_features=56, bias=True)
      (1): Linear(in_features=56, out_features=60, bias=True)
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

                                
| KEY           | VALUE        |
|---------------|--------------|
| epochs        | 400          |
| batch_size    | 20           |
| loss_type     | huber_loss   |
| optimizer     | Adam         |
| learning_rate | 0.0052784027 |
                                
</details>

