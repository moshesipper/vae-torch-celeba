# vae-torch-celeba

Accompanying code for my Medium article:
[A Basic Variational Autoencoder in PyTorch Trained on the CelebA Dataset
](https://medium.com/the-generator/a-basic-variational-autoencoder-in-pytorch-trained-on-the-celeba-dataset-f29c75316b26).

Files:
* `vae.py`: Class `VAE` + some definitions. You can change `IMAGE_SIZE`, `LATENT_DIM`, and `CELEB_PATH`.
* `trainvae.py`: Main code, training and testing. You can change `EPOCHS` and `BATCH_SIZE`. The models and images are placed in a directory `vaemodels-??????`, where `??????` are 6 random characters.
* `utils.py`: a couple of small utility functions.
* `genpics.py`: creates a panel of original image + 7 reconstructed ones.
* `vae_model_20.pth`: a trained VAE.

Running a trained model on a CPU is fine. 

Training on a CPU is possible, but slow: ⚡👉 GPU.
