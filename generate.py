import torch
import matplotlib.pyplot as plt
from gan import Generator
from prompt_to_latent import prompt_to_latent
import os

# Hyperparamètres
latent_dim = 100
img_shape = (3, 64, 64)
model_folder = 'models/'

# Charger le générateur entraîné
generator = Generator(latent_dim, img_shape)
generator.load_state_dict(torch.load(
    os.path.join(model_folder, 'generator_final.pth')))
generator.eval()


def generate_emoji(prompt):
    z = prompt_to_latent(prompt, latent_dim)
    with torch.no_grad():
        gen_img = generator(z).cpu().numpy().reshape(64, 64, 3)
        gen_img = (gen_img + 1) / 2.0

    plt.imshow(gen_img)
    plt.show()


# Exemple d'utilisation
generate_emoji("grinning face")
