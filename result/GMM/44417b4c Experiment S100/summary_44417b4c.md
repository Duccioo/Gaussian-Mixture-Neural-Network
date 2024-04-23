# Experiment Details Experiment S100
> from experiment with GMM
> on 2024-04-23 11-10
## Metrics:
                                                                   
| type  | r2     | mse    | max_error | ise     | kl     | evs    |
|-------|--------|--------|-----------|---------|--------|--------|
| Model | 0.4636 | 0.0016 | 0.1521    | 69.2541 | 0.1609 | 0.4638 |
                                                                   
## Plot Prediction

<img src="pdf_44417b4c.png">

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
| seed               | 55    |
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
| n_components | 3                |
| n_init       | 60               |
| max_iter     | 10               |
| init_params  | random_from_data |
| random_state | 52               |
                                   
</details>

<details><summary>Model Architecture </summary>

GaussianMixture(init_params='random_from_data', max_iter=10, n_components=3,
                n_init=60, random_state=52)
</details>

