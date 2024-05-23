# Experiment Details Experiment S50
> from experiment with GMM
> on 2024-05-23 17-46
## Metrics:
                                                                  
| type  | r2    | mse    | max_error | ise     | kl     | evs    |
|-------|-------|--------|-----------|---------|--------|--------|
| Model | 0.072 | 0.0026 | 0.2248    | 82.3759 | 0.1576 | 0.0913 |
                                                                  
## Plot Prediction

<img src="pdf_9afce42c.png">

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
| n_samples_training | 50    |
| n_samples_test     | 953   |
| n_samples_val      | 0     |
| notes              |       |
                              
## Model
> using model GMM
#### Model Params:
<details><summary>All Params used in the model </summary>

                         
| KEY          | VALUE  |
|--------------|--------|
| random_state | 55     |
| init_params  | kmeans |
| max_iter     | 70     |
| n_components | 4      |
| n_init       | 60     |
                         
</details>

<details><summary>Model Architecture </summary>

GaussianMixture(max_iter=70, n_components=4, n_init=60, random_state=55)
</details>

