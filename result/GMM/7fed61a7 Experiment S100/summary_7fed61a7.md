# Experiment Details Experiment S100
> from experiment with GMM
> on 2024-05-23 18-01
## Metrics:
                                                                  
| type  | r2     | mse    | max_error | ise    | kl     | evs    |
|-------|--------|--------|-----------|--------|--------|--------|
| Model | 0.2405 | 0.0023 | 0.1698    | 83.263 | 0.1609 | 0.2409 |
                                                                  
## Plot Prediction

<img src="pdf_7fed61a7.png">

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
| seed               | 37    |
| n_samples_training | 100   |
| n_samples_test     | 985   |
| n_samples_val      | 0     |
| notes              |       |
                              
## Model
> using model GMM
#### Model Params:
<details><summary>All Params used in the model </summary>

                                   
| KEY          | VALUE            |
|--------------|------------------|
| random_state | 52               |
| init_params  | random_from_data |
| max_iter     | 10               |
| n_components | 5                |
| n_init       | 60               |
                                   
</details>

<details><summary>Model Architecture </summary>

GaussianMixture(init_params='random_from_data', max_iter=10, n_components=5,
                n_init=60, random_state=52)
</details>

