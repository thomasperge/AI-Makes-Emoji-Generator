import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class EmojiDataset(Dataset):
    def __init__(self, image_folder, description_file, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_files = []
        self.descriptions = []

        # Lire le fichier de descriptions
        with open(description_file, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                if ',' in line:
                    parts = line.split(',', 1)
                    if len(parts) == 2:
                        img_file, description = parts
                        self.image_files.append(img_file.strip())
                        self.descriptions.append(description.strip())
                    else:
                        print(f"Skipping invalid line: {line}")
                else:
                    print(f"Skipping invalid line: {line}")

        # Vérifier que des images et descriptions ont été chargées
        if not self.image_files or not self.descriptions:
            raise ValueError("Aucune image ou description trouvée. Vérifie le format du fichier.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        description = self.descriptions[idx]
        return image, description

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def get_dataloader(image_folder, description_file, batch_size):
    dataset = EmojiDataset(image_folder=image_folder, description_file=description_file, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
