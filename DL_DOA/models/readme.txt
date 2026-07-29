This folder contains the pre-trained model checkpoints.

Models are loaded by their respective main.py files to perform inference.

This folder contains the pre-trained model checkpoints.

Files:

    inf_model_007_256_resnet.h5 : Trained ResNet model with 256x256 output.

    inf_model_007_256_unet : Due to GitHub size limitations, this model has been split into several compressed parts using 7-Zip:

        inf_model_007_256_unet.7z.001

        inf_model_007_256_unet.7z.002

        inf_model_007_256_unet.7z.003

        inf_model_007_256_unet.7z.004

        inf_model_007_256_unet.7z.005

Models are loaded by their respective main.py files to perform inference.

To reconstruct inf_model_007_256_unet.h5, you first need to install 7-Zip and extract the files:

sudo apt-get update
sudo apt-get install p7zip-full
7z x inf_model_007_256_unet.7z.001
And Make sure all parts (.7z.001 to .7z.005) are in the same folder before extraction.
