# Experiment Details Experiment S50
> from experiment with GMM
> on 2024-05-09 13-35
## Metrics:
                                                                    
| type  | r2      | mse    | max_error | ise     | kl     | evs    |
|-------|---------|--------|-----------|---------|--------|--------|
| Model | -0.0006 | 0.0028 | 0.2248    | 82.7223 | 0.1576 | 0.0186 |
                                                                    
## Plot Prediction

<img src="pdf_a39bdf12.png">

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
| n_samples_test     | 953   |
| n_samples_val      | 0     |
| notes              |       |
                              
## Model
> using model GMM
#### Model Params:
<details><summary>All Params used in the model </summary>

                                   
| KEY          | VALUE            |
|--------------|------------------|
| n_components | 3                |
| n_init       | 10               |
| max_iter     | 10               |
| init_params  | random_from_data |
| random_state | 78               |
                                   
</details>

<details><summary>Model Architecture </summary>

GaussianMixture(init_params='random_from_data', max_iter=10, n_components=3,
                n_init=10, random_state=78)
</details>

