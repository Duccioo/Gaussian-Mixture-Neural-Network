# Experiment Details Experiment C14 S100
> from experiment with GMM
> on 2024-03-27 22-53
## Metrics:
                                                                      
| type  | r2      | mse    | max_error | ise      | kl     | evs     |
|-------|---------|--------|-----------|----------|--------|---------|
| Model | -0.7974 | 0.0055 | 0.3769    | 119.1889 | 0.1609 | -0.7965 |
                                                                      
## Plot Prediction

<img src="/Users/duccio/Documents/GitHub/Gaussian-Mixture-Neural-Network/script/utils/../../result/GMM/6a6112 Experiment C14 
S100/pdf_6a6112.png">

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

                            
| KEY          | VALUE     |
|--------------|-----------|
| n_components | 10        |
| n_init       | 100       |
| max_iter     | 100       |
| init_params  | k-means++ |
| random_state | 85        |
                            
</details>

<details><summary>Model Architecture </summary>

GaussianMixture(init_params='k-means++', n_components=10, n_init=100,
                random_state=85)
</details>

