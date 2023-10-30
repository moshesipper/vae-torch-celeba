#  based on https://github.com/pytorch/examples/blob/main/vae/main.py
import os
import torch
import torch.utils.data
from os import mkdir
from torch import optim
from torch.nn import functional as F
from torchvision.utils import save_image
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader

# project modules
from utils import print, rndstr
from vae import VAE, IMAGE_SIZE, LATENT_DIM, CELEB_PATH, image_dim, celeb_transform

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

EPOCHS = 20  # number of training epochs
BATCH_SIZE = 16  # for data loaders
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('EPOCHS', EPOCHS, 'BATCH_SIZE', BATCH_SIZE, 'device', device)

# for model and results
directory = f'vaemodels-{rndstr()}'
mkdir(directory)
print(directory)

# download dataset
train_dataset = CelebA(CELEB_PATH, transform=celeb_transform, download=False, split='train')
test_dataset = CelebA(CELEB_PATH, transform=celeb_transform, download=False, split='valid') # or 'test'

# create train and test dataloaders
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, log_var):
    MSE =F.mse_loss(recon_x, x.view(-1, image_dim))
    KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    kld_weight = 0.00025
    loss = MSE + kld_weight * KLD  
    return loss


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        torch.cuda.empty_cache()
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data)
        log_var = torch.clamp_(log_var, -10, 10)
        loss = loss_function(recon_batch, data, mu, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, log_var = model(data)
            test_loss += loss_function(recon_batch, data, mu, log_var).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)[:n]])
                save_image(comparison.cpu(),
                           f'{directory}/reconstruction_{str(epoch)}.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    print(f'epochs: {EPOCHS}')

    for epoch in range(1, EPOCHS + 1):
        train(epoch)
        torch.save(model, f'{directory}/vae_model_{epoch}.pth')
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, LATENT_DIM).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 3, IMAGE_SIZE, IMAGE_SIZE),
                       f'{directory}/sample_{str(epoch)}.png')
