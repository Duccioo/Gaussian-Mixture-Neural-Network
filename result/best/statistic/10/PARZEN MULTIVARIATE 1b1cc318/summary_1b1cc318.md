# Experiment Details Experiment S10
> from experiment with Parzen Window
> on 2024-05-23 17-16
## Metrics:
                                                                  
| type  | r2    | mse    | max_error | ise     | kl     | evs    |
|-------|-------|--------|-----------|---------|--------|--------|
| Model | 0.457 | 0.0016 | 0.0938    | 46.1844 | 0.1706 | 0.6277 |
                                                                  
## Plot Prediction

<img src="pdf_1b1cc318.png">

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
| n_samples_training | 10    |
| n_samples_test     | 775   |
| n_samples_val      | 0     |
| notes              |       |
                              
## Model
> using model Parzen Window
#### Model Params:
<details><summary>All Params used in the model </summary>

                            
| KEY | VALUE              |
|-----|--------------------|
| h   | 0.9999926786987219 |
                            
</details>

<details><summary>Model Architecture </summary>

ParzenWindow_Model(h=0.9999926786987219, training=array([9.375019, 4.89003 , 7.043401, 4.748877, 5.748054, 5.605426,
       6.239142, 5.282705, 1.641122, 3.493585]))
</details>

