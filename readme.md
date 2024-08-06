IA to make custom emojis from text prompts. Input a description, and the AI generates a unique emoji based on your specifications

# Crée et active un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows, utilise `venv\Scripts\activate`

# Installe les bibliothèques nécessaires
pip install torch torchvision transformers numpy pillow matplotlib

# Commandes pour lancer le projet
pip install -r requirements.txt

# Lancer l'entraînement du GAN :
python launch_training.py

# Générer des emojis :
python generate.py
