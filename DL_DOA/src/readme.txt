This folder contains the core source code.

Files:
- tvt_data_generation_v3.py : Functions for generating validation data.
- tvt_models.py             : Definitions of ResNet and UNet architectures.
- TVT_Blob_Inference.py     : Inference pipeline to compute RMSE and Pd metrics for each condition.

These modules are imported by the main scripts under `z_resnet`, `z_unet`, and `z_resnet_vs_unet`.
