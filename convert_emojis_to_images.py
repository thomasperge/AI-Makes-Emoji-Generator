import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from PIL import Image, ImageFilter, ImageEnhance
import io
import base64
import os

print(mpl.__version__)

emoji = pd.read_csv('data/full_emoji.csv')
print(emoji.head())
print(len(emoji))

# Création répertoire pour save les fichiers PNG
output_dir = 'data/images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Boucle sur les 5 premières lignes du DataFrame
for i in range(5):
    base64_decoded = base64.b64decode(emoji['Apple'][i].split(',')[-1])
    image = Image.open(io.BytesIO(base64_decoded)).convert('RGBA')
    
    # Redimensionner l'image avec une résolution plus élevée
    upscale_factor = 1
    image = image.resize((image.width * upscale_factor, image.height * upscale_factor), Image.LANCZOS)
    
    # Augmenter légèrement la netteté de l'image
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.8)
    
    # Sauvegarder l'image en fichier PNG
    output_path = os.path.join(output_dir, f'emoji_{i+1}.png')
    image.save(output_path, format='PNG')
    print(f'Sauvegardé: {output_path}')
