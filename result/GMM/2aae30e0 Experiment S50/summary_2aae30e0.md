# Experiment Details Experiment S50
> from experiment with GMM
> on 2024-05-09 13-31
## Metrics:
                                                                    
| type  | r2      | mse    | max_error | ise    | kl     | evs     |
|-------|---------|--------|-----------|--------|--------|---------|
| Model | -0.1441 | 0.0035 | 0.2256    | 89.446 | 0.1609 | -0.1424 |
                                                                    
## Plot Prediction

<img src="pdf_2aae30e0.png">

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
| seed               | 8     |
| n_samples_training | 50    |
| n_samples_test     | 985   |
| n_samples_val      | 50    |
| notes              |       |
                              
## Model
> using model GMM
#### Model Params:
<details><summary>All Params used in the model </summary>

                                   
| KEY          | VALUE            |
|--------------|------------------|
| n_components | 3                |
| n_init       | 10               |
| max_iter     | 100              |
| init_params  | random_from_data |
| random_state | 78               |
                                   
</details>

<details><summary>Model Architecture </summary>

GaussianMixture(init_params='random_from_data', n_components=3, n_init=10,
                random_state=78)
</details>

