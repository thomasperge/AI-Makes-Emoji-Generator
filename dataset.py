import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# Charger le fichier CSV
csv_file = 'data/emoji_data.csv'  # Remplace par le chemin de ton fichier CSV
data = pd.read_csv(csv_file)
print(data)

# Créer le dossier pour les images si ce n'est pas déjà fait
image_folder = 'data/images/'
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

# Définir la taille et la police pour les images d'emojis
image_size = (64, 64)
font_size = 48
font = ImageFont.truetype("NotoColorEmoji-Regular.ttf", font_size)  # Assure-toi d'avoir la police arial.ttf ou spécifie une autre police

for index, row in data.iterrows():
    emoji_char = row['Emoji']
    print(emoji_char)
    description = row['Description']
    image = Image.new('RGB', image_size, (255, 255, 255))
    draw = ImageDraw.Draw(image)
    
    # Utiliser textbbox pour obtenir la taille du texte
    bbox = draw.textbbox((0, 0), emoji_char, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    
    draw.text(((image_size[0] - w) / 2, (image_size[1] - h) / 2), emoji_char, font=font, fill=(0, 0, 0))
    image_file = os.path.join(image_folder, f"{index}.png")
    image.save(image_file)

print("Conversion des emojis en images terminée.")
