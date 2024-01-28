# autoEncoders
from Vanilla to Graph AutoEncoder


## Overview
Models are in **encoders** folder. Train loop, evaluation and sampling (VAE, gVAE) in **main.py**. \
To show the differences between models, class "Sneaker" was chosen as an example. Models (except VAE) trained 1 epoch with default settings of `torch.optim.Adam()` optimizer and `nn.MSELoss()` loss. 

In RESULTS sesction provided SETUPs for models, results of training and evaluating, examples of reconstructing images, key observations and more. The results of the experiments are written in double manner:
**without | with** transforms of the dataset images, where transforms are: 
```
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=0, std=1),
                               ])
```
### Results overview 
|**Auto Encoders**||||||
|:---:    | :---:   |      :---:     |    :---:      |  :---: |      :---:    |
|model    | Vanilla |   Multilayer   | Convolutional | Sparse | **Denoising** |
|mean loss| 0.0243  |     0.0192     |     0.0172    | 0.0331 |   **0.0039**  |
|**Variational Auto Encoders**|
|model    | Vanilla | Convolutional |     Graph     |        |           |
|mean loss| 0.0000  |    0.0000     |     0.0000    | 0.0000 |   0.0000  |



## Dataset
Used FashionMNIST, default train_test_split, val_size from test split 0.8. 

## Results
### Vanilla AutoEncoder 
Evaluation MSE **1527.023 | 0.0243**. hidden_size=64, train_mode='any'.
<p float="left">
  <img
    src="https://github.com/dorochka8/autoEncoders/assets/97133490/06ce8c3d-42de-43f3-a083-01d978c5f5bf"
    title="TrainLossVanilla"
    style="display: inline-block; margin: 0 auto; width: 45%"
    align="center" 
    height="300"
  >
  <img
    src="https://github.com/dorochka8/autoEncoders/assets/97133490/3653c514-93ff-4f0c-b87c-3ecb034379f9"
    title="ExampleFromEvaluationVanilla"
    style="display: inline-block; margin: 0 auto; width: 45%"
    align="center" 
    height="300"
  >
</p>

### Multilayer AutoEncoder 
Evaluation MSE **1268.625 | 0.0192**. hidden_size=128, coder_size=64, train_mode='any'.
<p float="left">
  <img
    src="https://github.com/dorochka8/autoEncoders/assets/97133490/e2f3298f-1c64-483f-ae7b-42cc8f33134d"
    title="TrainLossMultilayer"
    style="display: inline-block; margin: 0 auto; width: 45%"
    align="center" 
    height="300"
  >
  <img
    src="https://github.com/dorochka8/autoEncoders/assets/97133490/fbbe7117-1df2-4019-8d2d-5a1d8dba3e51"
    title="ExampleFromEvaluationMultilayer"
    style="display: inline-block; margin: 0 auto; width: 45%"
    align="center" 
    height="300"
  >
</p>

### Convolutional AutoEncoder 
It was made quite simple. Used only `nn.Conv2d`, and `nn.MaxPool2d` and `nn.Upsample` for encoder and decoder respectively. 
Evaluation MSE **1369.393 | 0.0172**. input_size=1, train_mode='any'.
<p float="left">
  <img
    src="https://github.com/dorochka8/autoEncoders/assets/97133490/449b5f0a-e2ec-468c-aca0-570139adc7d9"
    title="TrainLossConvolutional"
    style="display: inline-block; margin: 0 auto; width: 45%"
    align="center" 
    height="300"
  >
  <img
    src="https://github.com/dorochka8/autoEncoders/assets/97133490/193d96e9-4d25-4c71-8683-97546114ec8d"
    title="ExampleFromEvaluationConvolutional"
    style="display: inline-block; margin: 0 auto; width: 45%"
    align="center" 
    height="300"
  >
</p>

### Sparse AutoEncoder 
Evaluation MSE **1457.481 | 0.0331**. l1_coef=10e-5, hidden_size=64, train_mode='sparse'.
<p float="left">
  <img
    src="https://github.com/dorochka8/autoEncoders/assets/97133490/00d58c3d-af4d-4c96-ba71-4a1ae3fe2daf"
    title="TrainLossSparse"
    style="display: inline-block; margin: 0 auto; width: 45%"
    align="center" 
    height="300"
  >
  <img
    src="https://github.com/dorochka8/autoEncoders/assets/97133490/159b02c1-afa1-4ed8-9579-8010937fcc88"
    title="ExampleFromEvaluationSparse"
    style="display: inline-block; margin: 0 auto; width: 45%"
    align="center" 
    height="300"
  >
</p>

### Denoising AutoEncoder 
Evaluation **MSE 270.227 | 0.0039**. input_size=1, train_mode='any'.
<p float="left">
  <img
    src="https://github.com/dorochka8/autoEncoders/assets/97133490/61df8dbc-66ab-4da1-8155-6fa79c44d406"
    title="TrainLossDenoising"
    style="display: inline-block; margin: 0 auto; width: 45%"
    align="center" 
    height="300"
  >
  <img
    src="https://github.com/dorochka8/autoEncoders/assets/97133490/5eb53285-76fe-42e8-95ee-fe4d24fa879a"
    title="ExampleFromEvaluationDenoising"
    style="display: inline-block; margin: 0 auto; width: 45%"
    align="center"
    height="300"
  >
</p>

### VariationalAutoEncoder 
Added KL divergence to the total loss (acc. to https://arxiv.org/pdf/1312.6114.pdf, p.10 *Gaussian Case*). 
```
  KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
```
Also the validity of MSE was increased 5 times.  All the experiments were done on normalized data, and another train function for batched data. 


### Convolutional VariationalAutoEncoder 
Key observation: in encoder, when one convolves input, immediately increase the number of *out_channels* in the very first convolutional layer, to have better resuls. Doing *in_channels=1, out_channels=16* gives significantly better results, than *in_channels=1, out_channels=3*. Scheduler is not helping in here.\
Evaluation **MSE 0.0000**. input_size=1, hidden_size=256, epochs=20. 





 



