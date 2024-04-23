# Experiment Details Experiment S100
> from experiment with GMM
> on 2024-04-12 11-11
## Metrics:
                                                                   
| type  | r2     | mse    | max_error | ise      | kl    | evs    |
|-------|--------|--------|-----------|----------|-------|--------|
| Model | 0.7827 | 0.0347 | 1.1339    | 362.1963 | 0.702 | 0.7827 |
                                                                   
## Plot Prediction

<img src="pdf_60f8ea65.png">

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
| seed               | 42    |
| n_samples_training | 100   |
| n_samples_test     | 319   |
| n_samples_val      | 0     |
| notes              |       |
                              
## Model
> using model GMM
#### Model Params:
<details><summary>All Params used in the model </summary>

                         
| KEY          | VALUE  |
|--------------|--------|
| n_components | 4      |
| n_init       | 100    |
| max_iter     | 100    |
| init_params  | random |
| random_state | 42     |
                         
</details>

<details><summary>Model Architecture </summary>

GaussianMixture(init_params='random', n_components=4, n_init=100,
                random_state=42)
</details>

