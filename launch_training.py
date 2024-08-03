import os
import torch
from dataset import get_dataloader
from gan import Generator, Discriminator
from train_gan import train_gan

# Hyperparamètres
latent_dim = 100
img_shape = (3, 64, 64)
batch_size = 32
n_epochs = 20000
lr = 0.0002
b1 = 0.5
b2 = 0.999
sample_interval = 1000
save_interval = 1000

# Dossier des images et fichiers de descriptions
image_folder = 'data/images/'
description_file = 'data/descriptions.txt'
model_folder = 'models/'

# Préparer les données
dataloader = get_dataloader(image_folder, description_file, batch_size)

# Instancier le générateur et le discriminateur
generator = Generator(latent_dim, img_shape)
discriminator = Discriminator(img_shape)

# Entraîner le GAN
train_gan(generator, discriminator, dataloader, latent_dim, n_epochs,
          lr, b1, b2, sample_interval, save_interval, model_folder)

# Sauvegarder les modèles finaux
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
torch.save(generator.state_dict(), os.path.join(
    model_folder, 'generator_final.pth'))
torch.save(discriminator.state_dict(), os.path.join(
    model_folder, 'discriminator_final.pth'))
