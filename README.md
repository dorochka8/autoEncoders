# autoEncoders
from Vanilla to Graph AutoEncoder

Models are in `encoders` folder. Train loop, evaluation and sampling (VAE, gVAE) in `main.py`. 

## Dataset
Used FashionMNIST, default split on train and test, val_size from test split 0.8. 

## Results
### Vanilla AutoEncoder 
Evaluation MSE 1527.023.
<p float="left">
  <img
    src="https://github.com/dorochka8/autoEncoders/assets/97133490/06ce8c3d-42de-43f3-a083-01d978c5f5bf"
    title="TrainLossVanilla"
    style="display: inline-block; margin: 0 auto; width: 500px"
    align="center" 
    height="350"
  >
  <img
    src="https://github.com/dorochka8/autoEncoders/assets/97133490/3653c514-93ff-4f0c-b87c-3ecb034379f9"
    title="ExampleFromEvaluationVanilla"
    style="display: inline-block; margin: 0 auto; width: 500px"
    align="center" 
    height="350"
  >
</p>

### Multilayer AutoEncoder 
Evaluation MSE 1268.625.
<p float="left">
  <img
    src="https://github.com/dorochka8/autoEncoders/assets/97133490/e2f3298f-1c64-483f-ae7b-42cc8f33134d"
    title="TrainLossMultilayer"
    style="display: inline-block; margin: 0 auto; width: 500px"
    align="center" 
    height="350"
  >
  <img
    src="https://github.com/dorochka8/autoEncoders/assets/97133490/fbbe7117-1df2-4019-8d2d-5a1d8dba3e51"
    title="ExampleFromEvaluationMultilayer"
    style="display: inline-block; margin: 0 auto; width: 500px"
    align="center" 
    height="350"
  >
</p>

### Convolutional AutoEncoder 
Evaluation MSE 1369.393.
<p float="left">
  <img
    src="https://github.com/dorochka8/autoEncoders/assets/97133490/449b5f0a-e2ec-468c-aca0-570139adc7d9"
    title="TrainLossConvolutional"
    style="display: inline-block; margin: 0 auto; width: 500px"
    align="center" 
    height="350"
  >
  <img
    src="https://github.com/dorochka8/autoEncoders/assets/97133490/193d96e9-4d25-4c71-8683-97546114ec8d"
    title="ExampleFromEvaluationConvolutional"
    style="display: inline-block; margin: 0 auto; width: 500px"
    align="center" 
    height="350"
  >
</p>






