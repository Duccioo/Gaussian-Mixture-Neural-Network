# Experiment Details Experiment  H0.3795755152130492 S150
> from experiment with Parzen Window + NN
> on 2024-03-28 01-33
## Metrics:
                                                                   
| type   | r2     | mse    | max_error | ise    | kl     | evs    |
|--------|--------|--------|-----------|--------|--------|--------|
| Target | 0.5568 | 0.0013 | 0.1435    | 0.0013 | 0.0313 | 0.6581 |
| Model  | 0.7408 | 0.0008 | 0.1202    | 0.008  | 0.0377 | 0.7485 |
                                                                   
## Plot Prediction

<img src="/Users/duccio/Documents/GitHub/Gaussian-Mixture-Neural-Network/script/utils/../../result/Parzen Window + NN/7c3f12 Experiment  
H0.3795755152130492 S150/pdf_7c3f12.png">

## Loss Plot

<img src="/Users/duccio/Documents/GitHub/Gaussian-Mixture-Neural-Network/script/utils/../../result/Parzen Window + NN/7c3f12 Experiment  
H0.3795755152130492 S150/loss_7c3f12.png">

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
| seed               | 72    |
| n_samples_training | 100   |
| n_samples_test     | 988   |
| n_samples_val      | 50    |
| notes              |       |
                              
## Model
> using model Parzen Window + NN
#### Model Params:
<details><summary>All Params used in the model </summary>

                                                                              
| KEY             | VALUE                                                    |
|-----------------|----------------------------------------------------------|
| dropout         | 0.0                                                      |
| hidden_layer    | [(56, Tanh()), (60, ReLU()), (28, ReLU()), (58, Tanh())] |
| last_activation | None                                                     |
                                                                              
</details>

<details><summary>Model Architecture </summary>

LitModularNN(
  (neural_netowrk_modular): NeuralNetworkModular(
    (dropout): Dropout(p=0.0, inplace=False)
    (output_layer): Linear(in_features=58, out_features=1, bias=True)
    (layers): ModuleList(
      (0): Linear(in_features=1, out_features=56, bias=True)
      (1): Linear(in_features=56, out_features=60, bias=True)
      (2): Linear(in_features=60, out_features=28, bias=True)
      (3): Linear(in_features=28, out_features=58, bias=True)
    )
    (activation): ModuleList(
      (0): Tanh()
      (1-2): 2 x ReLU()
      (3): Tanh()
    )
  )
)
</details>

## Training
<details><summary>All Params used for the training </summary>

                               
| KEY           | VALUE       |
|---------------|-------------|
| epochs        | 100         |
| batch_size    | 2           |
| loss_type     | huber_loss  |
| optimizer     | Adam        |
| learning_rate | 0.000438031 |
                               
</details>

