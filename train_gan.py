import torch
from torch.utils.data import DataLoader
import os

def train_gan(generator, discriminator, dataloader, latent_dim, n_epochs, lr, b1, b2, sample_interval, save_interval, model_folder):
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    for epoch in range(n_epochs):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0

        for i, (imgs, labels, indices) in enumerate(dataloader):
            # Afficher les indices des images
            print(f"Batch {i} | Epoch {epoch}")

            # Adversarial ground truths
            valid = torch.ones(imgs.size(0), 1, requires_grad=False)
            fake = torch.zeros(imgs.size(0), 1, requires_grad=False)

            # Configure input
            real_imgs = imgs

            # Train Generator
            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = torch.randn(imgs.size(0), latent_dim)

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()

            # Loss for real images
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            # Loss for fake images
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            # Total discriminator loss
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # Accumuler les pertes
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()

        # Afficher les pertes à la fin de chaque époque
        avg_d_loss = epoch_d_loss / len(dataloader)
        avg_g_loss = epoch_g_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{n_epochs} | D loss: {avg_d_loss} | G loss: {avg_g_loss}")

        # Sauvegarder le modèle à intervalles spécifiés
        if (epoch + 1) % save_interval == 0:
            save_model(generator, discriminator, epoch, path=model_folder)

def save_model(generator, discriminator, epoch, path='models'):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(generator.state_dict(), f"{path}/generator_epoch_{epoch}.pth")
    torch.save(discriminator.state_dict(), f"{path}/discriminator_epoch_{epoch}.pth")

def load_model(generator, discriminator, epoch, path='models'):
    generator.load_state_dict(torch.load(f"{path}/generator_epoch_{epoch}.pth"))
    discriminator.load_state_dict(torch.load(f"{path}/discriminator_epoch_{epoch}.pth"))
