import torch
import torch.nn as nn

# Définir un exemple de modèle
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Chemin vers le fichier .pth (à remplacer par votre chemin)
file_path = 'chemin/vers/votre_fichier.pth'

# Charger le fichier .pth
checkpoint = torch.load(file_path)

# Afficher le contenu du fichier
print("Contenu du fichier .pth:")
print(checkpoint)

# Charger les poids dans le modèle si le fichier .pth contient un state_dict
model = Net()
model.load_state_dict(checkpoint)
model.eval()  # mettre le modèle en mode évaluation si nécessaire

# Afficher les poids du modèle
print("\nPoids du modèle après chargement:")
for name, param in model.named_parameters():
    print(f'Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n')
