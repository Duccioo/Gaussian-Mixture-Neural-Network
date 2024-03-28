# Experiment Details Experiment C14 S100
> from experiment with GMM
> on 2024-03-27 23-16
## Metrics:
                                                                   
| type  | r2     | mse    | max_error | ise     | kl     | evs    |
|-------|--------|--------|-----------|---------|--------|--------|
| Model | 0.3641 | 0.0019 | 0.1698    | 74.8098 | 0.1609 | 0.3645 |
                                                                   
## Plot Prediction

<img src="/Users/duccio/Documents/GitHub/Gaussian-Mixture-Neural-Network/script/utils/../../result/GMM/ab757d Experiment C14 
S100/pdf_ab757d.png">

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
| seed               | 36    |
| n_samples_training | 100   |
| n_samples_test     | 985   |
| n_samples_val      | 0     |
| notes              |       |
                              
## Model
> using model GMM
#### Model Params:
<details><summary>All Params used in the model </summary>

                         
| KEY          | VALUE  |
|--------------|--------|
| n_components | 4      |
| n_init       | 11     |
| max_iter     | 684    |
| init_params  | kmeans |
| random_state | 69     |
                         
</details>

<details><summary>Model Architecture </summary>

GaussianMixture(max_iter=684, n_components=4, n_init=11, random_state=69)
</details>

