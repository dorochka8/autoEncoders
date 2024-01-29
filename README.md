# autoEncoders
from Vanilla to Graph AutoEncoder


## Overview

Used FashionMNIST, default train_test_split, val_size from test split 0.8. Models are in **encoders** folder. Train loop, evaluation and sampling (AE, VAE) in **main.py**. 
### AE
To show the differences between models, class "Sneaker" was chosen as an example. Models are trained 1 epoch with default settings of `torch.optim.Adam()` optimizer and `nn.MSELoss()` loss. The results of the experiments are written in double manner:
**without | with** transforms of the dataset, where transforms are: 
```
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=0, std=1),
                               ])
```
### VAE
To show the differences between models, class "Sandal" was chosen as an example. Models are trained 20 epoch with default settings of `torch.optim.Adam()` optimizer and `nn.MSELoss() + KLD` loss. The results of the experiments are provided for the transformed dataset.

In RESULTS section provided SETUPs for models, results of training (train loss plot) and evaluating, examples of reconstructing images while evaluating, key observations and more. 

### Results overview 
|**Auto Encoders**||||||
|:---:    | :---:   |      :---:     |    :---:      |  :---: |      :---:    |
|model    | Vanilla |   Multilayer   | Convolutional | Sparse | **Denoising** |
|mean loss| 0.0243  |     0.0192     |     0.0172    | 0.0331 |   **0.0039**  |
|**Variational Auto Encoders**|
|model    | Vanilla | Convolutional |     Graph     |        |           |
|mean loss| 0.0000  |    415.2351   |     0.0000    | 0.0000 |   0.0000  |

## Results
### Vanilla AutoEncoder 
Evaluation MSE **1527.023 | 0.0243**. hidden_size=64, train_mode='any'.
<p float="left">
  <img
    src="https://github.com/dorochka8/autoEncoders/assets/97133490/06ce8c3d-42de-43f3-a083-01d978c5f5bf"
    title="TrainLossVanilla"
    style="display: inline-block; margin: 0 auto; width: 45%"
    align="center" 
    height=45%
  >
  <img
    src="https://github.com/dorochka8/autoEncoders/assets/97133490/3653c514-93ff-4f0c-b87c-3ecb034379f9"
    title="ExampleFromEvaluationVanilla"
    style="display: inline-block; margin: 0 auto; width: 45%"
    align="center" 
    height=45%
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
    height=45%
  >
  <img
    src="https://github.com/dorochka8/autoEncoders/assets/97133490/fbbe7117-1df2-4019-8d2d-5a1d8dba3e51"
    title="ExampleFromEvaluationMultilayer"
    style="display: inline-block; margin: 0 auto; width: 45%"
    align="center" 
    height=45%
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
    height=45%
  >
  <img
    src="https://github.com/dorochka8/autoEncoders/assets/97133490/193d96e9-4d25-4c71-8683-97546114ec8d"
    title="ExampleFromEvaluationConvolutional"
    style="display: inline-block; margin: 0 auto; width: 45%"
    align="center" 
    height=45%
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
    height=45%
  >
  <img
    src="https://github.com/dorochka8/autoEncoders/assets/97133490/159b02c1-afa1-4ed8-9579-8010937fcc88"
    title="ExampleFromEvaluationSparse"
    style="display: inline-block; margin: 0 auto; width: 45%"
    align="center" 
    height=45%
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
    height=45%
  >
  <img
    src="https://github.com/dorochka8/autoEncoders/assets/97133490/5eb53285-76fe-42e8-95ee-fe4d24fa879a"
    title="ExampleFromEvaluationDenoising"
    style="display: inline-block; margin: 0 auto; width: 45%"
    align="center"
    height=45%
  >
</p>

### VariationalAutoEncoder 
Added KL divergence to the total loss (acc. to https://arxiv.org/pdf/1312.6114.pdf, p.10 *Gaussian Case*). 
```
  KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
```
All the experiments were done on normalized data, and another train function for batched data. 
Evaluation **MSE 0.0000**. input_size=0, batch_size=000, hidden_size=00, epochs=20. 

#### Sampling from latent space: 
```
num_samplings = 1 
samples = model.sampling(num_samplings).detach().cpu()
for sample in samples:
  plt.imshow(sample.reshape(28, 28))
  plt.show()
  clear_output(wait=True)
```
RESULTS //////////

### Convolutional VariationalAutoEncoder 
Key observation: in encoder, when one convolves input, immediately increase the number of *out_channels* in the very first convolutional layer, to have better resuls. Doing *in_channels=1, out_channels=32* gives significantly better results, than gradually increasing number of channels *in_channels=1, out_channels=3*. Scheduler is not helping in here.\
Evaluation **MSE 415.2351**. input_size=1, batch_size=125, hidden_size=32, epochs=20. 
<p float="left">
  <img
    src="https://github.com/dorochka8/autoEncoders/assets/97133490/2a0b1512-cfb8-4477-b642-75a079748af3"
    title="TrainLossConvolutionalVAE"
    style="display: inline-block; margin: 0 auto; width: 45%"
    align="center" 
    height=45%
  >
  <img
    src="https://github.com/dorochka8/autoEncoders/assets/97133490/3fa6dc37-236f-4b25-ae65-7dfafb2bbf94"
    title="ExampleFromEvaluationConvolutionalVAE"
    style="display: inline-block; margin: 0 auto; width: 45%"
    align="center" 
    height=45%
  >
</p>

#### Sampling from latent space: 
<p float="left">
  <img
    src="https://github.com/dorochka8/autoEncoders/assets/97133490/af156466-3d35-4bc7-8c96-c4949ee289ee"
    title="sample1ConvolutionalVAE"
    style="display: inline-block; margin: 0 auto; width: 15%"
    align="center" 
    height=15%
  >
  <img
    src="https://github.com/dorochka8/autoEncoders/assets/97133490/c4f4bffd-e135-4a9c-927c-7a5d8ca73610"
    title="sample2ConvolutionalVAE"
    style="display: inline-block; margin: 0 auto; width: 15%"
    align="center" 
    height=15%
  >
  <img
    src="https://github.com/dorochka8/autoEncoders/assets/97133490/4b794231-3a01-4de2-bede-4825d5ce664f"
    title="sample3ConvolutionalVAE"
    style="display: inline-block; margin: 0 auto; width: 15%"
    align="center" 
    height=15%
  >
  <img
    src="https://github.com/dorochka8/autoEncoders/assets/97133490/e1135d33-d2ce-4b4c-b551-758293f197a3"
    title="sample4ConvolutionalVAE"
    style="display: inline-block; margin: 0 auto; width: 15%"
    align="center" 
    height=15%
  >
  <img
    src="https://github.com/dorochka8/autoEncoders/assets/97133490/804d3103-6295-4710-a91a-8b6b19bd324e"
    title="sample5ConvolutionalVAE"
    style="display: inline-block; margin: 0 auto; width: 15%"
    align="center" 
    height=15%
  >
</p>

