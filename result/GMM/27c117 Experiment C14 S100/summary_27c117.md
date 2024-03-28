# Experiment Details Experiment C14 S100
> from experiment with GMM
> on 2024-03-27 22-34
## Metrics:
                                                                   
| type  | r2     | mse    | max_error | ise     | kl     | evs    |
|-------|--------|--------|-----------|---------|--------|--------|
| Model | 0.2388 | 0.0023 | 0.1707    | 85.9788 | 0.1609 | 0.2401 |
                                                                   
## Plot Prediction

<img src="/Users/duccio/Documents/GitHub/Gaussian-Mixture-Neural-Network/script/utils/../../result/GMM/27c117 Experiment C14 
S100/pdf_27c117.png">

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
| n_components | 6         |
| n_init       | 41        |
| max_iter     | 203       |
| init_params  | k-means++ |
| random_state | 85        |
                            
</details>

<details><summary>Model Architecture </summary>

GaussianMixture(init_params='k-means++', max_iter=203, n_components=6,
                n_init=41, random_state=85)
</details>

