# Head and Neck cancer analysis using CNN U-net

## Introduction
In this repository, the U-net models are trained to auto-delineate head & neck cancer tumors.

Five-fold cross-validation was used on the training set with single or multimodality PET/CT input.

The `experiment` module from the [deoxys](https://pypi.org/project/deoxys/) package was used to set-up the U-net model structures, preprocess the inputs, and train the models.

## Run the script on Orion

```
sbatch slurm.sh config/2d_unet.json 2d_unet
```

This will load the configuration from the `config/2d_unet.json` file, train for 200 epochs and store the results in the folder `$HOME/unet/2d_unet/`.

To customize the frequency of saving model and prediction

```
sbatch slurm.sh config/3d_vnet_32_normalize.json 3d_vnet_32_normalize 100 --model_checkpoint_period 5 --prediction_checkpoint_period 5
```
