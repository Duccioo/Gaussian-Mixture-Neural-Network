# Experiment Details Experiment S50
> from experiment with GMM
> on 2024-05-23 17-19
## Metrics:
                                                                   
| type  | r2     | mse    | max_error | ise      | kl    | evs    |
|-------|--------|--------|-----------|----------|-------|--------|
| Model | 0.7645 | 0.0376 | 1.0493    | 395.0874 | 0.702 | 0.7647 |
                                                                   
## Plot Prediction

<img src="pdf_f2c36c8b.png">

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
| seed               | 37    |
| n_samples_training | 50    |
| n_samples_test     | 319   |
| n_samples_val      | 0     |
| notes              |       |
                              
## Model
> using model GMM
#### Model Params:
<details><summary>All Params used in the model </summary>

                         
| KEY          | VALUE  |
|--------------|--------|
| random_state | 49     |
| init_params  | random |
| n_components | 31     |
| n_init       | 30     |
| max_iter     | 10     |
                         
</details>

<details><summary>Model Architecture </summary>

GaussianMixture(init_params='random', max_iter=10, n_components=31, n_init=30,
                random_state=49)
</details>
