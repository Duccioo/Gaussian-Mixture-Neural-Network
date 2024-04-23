# Experiment Details Experiment S100
> from experiment with GMM
> on 2024-04-23 11-23
## Metrics:
                                                                   
| type  | r2     | mse    | max_error | ise      | kl    | evs    |
|-------|--------|--------|-----------|----------|-------|--------|
| Model | 0.7847 | 0.0344 | 1.1301    | 364.2961 | 0.702 | 0.7847 |
                                                                   
## Plot Prediction

<img src="pdf_a6aebb2f.png">

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
| seed               | 56    |
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
| n_components | 30     |
| n_init       | 100    |
| max_iter     | 30     |
| init_params  | random |
| random_state | 93     |
                         
</details>

<details><summary>Model Architecture </summary>

GaussianMixture(init_params='random', max_iter=30, n_components=30, n_init=100,
                random_state=93)
</details>

