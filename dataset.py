import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class EmojiDataset(Dataset):
    def __init__(self, image_folder, csv_file, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.data = pd.read_csv(csv_file)
        self.data = self.data.head(150)  # Prendre seulement les 5 premières lignes pour tester
        self.image_files = [os.path.join(image_folder, f"emoji_{i+1}.png") for i in range(len(self.data))]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"File not found: {img_path}")
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.data.iloc[idx]['name'], idx
    
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def get_dataloader(image_folder, csv_file, batch_size):
    dataset = EmojiDataset(image_folder=image_folder, csv_file=csv_file, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
