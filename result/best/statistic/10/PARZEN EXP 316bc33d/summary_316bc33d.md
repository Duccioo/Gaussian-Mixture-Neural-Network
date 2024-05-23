# Experiment Details Experiment S10
> from experiment with Parzen Window
> on 2024-05-23 17-13
## Metrics:
                                                                   
| type  | r2     | mse    | max_error | ise     | kl     | evs    |
|-------|--------|--------|-----------|---------|--------|--------|
| Model | 0.9142 | 0.0117 | 0.457     | 74.6299 | 0.3069 | 0.9146 |
                                                                   
## Plot Prediction

<img src="pdf_316bc33d.png">

## Dataset

<details><summary>PDF set as default <b>EXPONENTIAL_06</b></summary>

#### Dimension 1
                               
| type        | rate | weight |
|-------------|------|--------|
| exponential | 0.6  | 1      |
                               
</details>
                              
| KEY                | VALUE |
|--------------------|-------|
| dimension          | 1     |
| seed               | 37    |
| n_samples_training | 10    |
| n_samples_test     | 180   |
| n_samples_val      | 0     |
| notes              |       |
                              
## Model
> using model Parzen Window
#### Model Params:
<details><summary>All Params used in the model </summary>

                            
| KEY | VALUE              |
|-----|--------------------|
| h   | 0.1697189191100632 |
                            
</details>

<details><summary>Model Architecture </summary>

ParzenWindow_Model(h=0.1697189191100632, training=array([1.40171379, 0.16787657, 0.87159631, 1.87457757, 0.62793651,
       0.65341418, 0.23213689, 0.09226395, 0.18910752, 0.2477912 ]))
</details>

