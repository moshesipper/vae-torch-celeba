import torch
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from vae import IMAGE_SIZE, celeb_transform, CELEB_PATH

MODEL_FILE = 'vae_model_20.pth'

dataset = CelebA(CELEB_PATH, transform=celeb_transform, download=False, split='all')
loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
model = torch.load(MODEL_FILE, map_location='cpu')

for pic, _ in loader:  # batch size is 1, loader is shuffled, so this gets one random pic
    pics = pic.to('cpu')
    break
orig = torch.clone(pics)

for _ in range(7):
    recon, _, _ = model(pics)
    pic = recon[0].view(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    pics = torch.cat((pics, pic), dim=0)

save_image(pics, 'rndpics.jpg', nrow=8)


# use code below if you want to manually tweak the latent vector
# mu, log_var = model.encode(orig)

# for _ in range(7):
#     w = 1e-11
#     std = torch.exp(w * log_var)
#     eps = torch.randn_like(std)
#     z = eps * std + mu
#     recon = model.decode(z)
#     pic = recon[0].view(1, 3, IMAGE_SIZE, IMAGE_SIZE)
#     pics = torch.cat((pics, pic), dim=0)

# save_image(pics, 'rndpics.jpg', nrow=8)
