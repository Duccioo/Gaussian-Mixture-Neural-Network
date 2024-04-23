# Experiment Details Experiment S200
> from experiment with GMM
> on 2024-04-23 11-27
## Metrics:
                                                                  
| type  | r2     | mse    | max_error | ise     | kl     | evs   |
|-------|--------|--------|-----------|---------|--------|-------|
| Model | 0.7536 | 0.0008 | 0.1737    | 68.8703 | 0.1627 | 0.754 |
                                                                  
## Plot Prediction

<img src="pdf_5628c462.png">

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
| seed               | 0     |
| n_samples_training | 200   |
| n_samples_test     | 999   |
| n_samples_val      | 0     |
| notes              |       |
                              
## Model
> using model GMM
#### Model Params:
<details><summary>All Params used in the model </summary>

                                   
| KEY          | VALUE            |
|--------------|------------------|
| n_components | 5                |
| n_init       | 60               |
| max_iter     | 90               |
| init_params  | random_from_data |
| random_state | 46               |
                                   
</details>

<details><summary>Model Architecture </summary>

GaussianMixture(init_params='random_from_data', max_iter=90, n_components=5,
                n_init=60, random_state=46)
</details>

