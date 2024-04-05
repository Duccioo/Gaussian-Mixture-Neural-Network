# Experiment Details Experiment  H0.11685252311419939 S100
> from experiment with PNN
> on 2024-04-05 17-32
## Metrics:
                                                                                                    
| type   | r2           | mse          | max_error    | ise          | kl           | evs          |
|--------|--------------|--------------|--------------|--------------|--------------|--------------|
| Target | 0.8279043291 | 0.0412311063 | 0.6473360251 | 0.0412311063 | 0.0210905471 | 0.8733180689 |
| Model  | 0.9533       | 0.006        | 0.4439       | 0.0301       | 0.1123       | 0.955        |
                                                                                                    
## Plot Prediction

<img src="pdf_c39a4876.png">

## Loss Plot

<img src="loss_c39a4876.png">

## Dataset

<details><summary>PDF attribute</summary>

#### Dimension 1
                               
| type        | rate | weight |
|-------------|------|--------|
| exponential | 0.6  | 1      |
                               
</details>
                              
| KEY                | VALUE |
|--------------------|-------|
| dimension          | 1     |
| seed               | 60    |
| n_samples_training | 100   |
| n_samples_test     | 501   |
| n_samples_val      | 0     |
| notes              |       |
                              
## Target
- Using PNN Target
<details><summary>All Params used in the model for generate the target for the MLP </summary>

                             
| KEY | VALUE               |
|-----|---------------------|
| h   | 0.11685252311419939 |
                             
</details>

## Model
> using model PNN
#### Model Params:
<details><summary>All Params used in the model </summary>

                                                                
| KEY             | VALUE                                      |
|-----------------|--------------------------------------------|
| dropout         | 0.0                                        |
| hidden_layer    | [(22, Tanh()), (24, ReLU()), (60, ReLU())] |
| last_activation | lambda                                     |
                                                                
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
      (0): Linear(in_features=1, out_features=22, bias=True)
      (1): Linear(in_features=22, out_features=24, bias=True)
      (2): Linear(in_features=24, out_features=60, bias=True)
      (3): AdaptiveSigmoid(
        (sigmoid): Sigmoid()
      )
    )
    (activation): ModuleList(
      (0): Tanh()
      (1-2): 2 x ReLU()
    )
  )
)
</details>

## Training
<details><summary>All Params used for the training </summary>

                                 
| KEY           | VALUE         |
|---------------|---------------|
| epochs        | 380           |
| batch_size    | 2             |
| loss_type     | huber_loss    |
| optimizer     | RMSprop       |
| learning_rate | 0.00660307851 |
                                 
</details>

