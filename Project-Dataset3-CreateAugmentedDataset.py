## Imports
import os
import shutil
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
from tqdm import tqdm

class SpeciesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.species_map = {"Ak": 0, "Ala_Idris": 1, "Buzgulu": 2, "Dimnit": 3, "Nazli": 4}
        
        for species in self.species_map:
            species_dir = os.path.join(root_dir, species)
            for img_name in os.listdir(species_dir):
                img_path = os.path.join(species_dir, img_name)
                if img_path.lower().endswith(('png', 'jpg', 'jpeg')):
                    try:
                        Image.open(img_path).verify()
                        self.image_paths.append(img_path)
                        self.labels.append(self.species_map[species])
                    except:
                        continue
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

class DiseaseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.disease_labels = []
        self.disease_map = {
            "Black_Rot": 0, "Brown_Spot": 1, "Downy_Mildew": 2,
            "Mites_Disease": 3, "Healthy_Leaves": 4
        }
        
        for disease in self.disease_map:
            disease_dir = os.path.join(root_dir, disease)
            for img_name in os.listdir(disease_dir):
                img_path = os.path.join(disease_dir, img_name)
                if img_path.lower().endswith(('png', 'jpg', 'jpeg')):
                    try:
                        Image.open(img_path).verify()
                        self.image_paths.append(img_path)
                        self.disease_labels.append(disease)
                    except:
                        continue
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        disease_label = self.disease_labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_path, disease_label

def train_model(model, train_loader, val_loader, num_epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1) 
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        scheduler.step()
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

def predict_and_compose(model, disease_loader, device, output_dir="dataset-3/Augmented_Dataset"):
    model.eval()
    species_map = {0: "Ak", 1: "Ala_Idris", 2: "Buzgulu", 3: "Dimnit", 4: "Nazli", 5: "unknown"}
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for images, img_paths, disease_labels in tqdm(disease_loader, desc="Predicting"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            max_probs, predicted = torch.max(probs, 1)
            
            for i in range(len(predicted)):
                species_pred = predicted[i].item()
                species_label = species_map[species_pred]
                disease_label = disease_labels[i]
                confidence = max_probs[i].item()
                
                if confidence < 0.5:
                    species_label = "unknown"
                
                augmented_folder = os.path.join(output_dir, f"{species_label}_{disease_label}")
                os.makedirs(augmented_folder, exist_ok=True)
                shutil.copy(img_paths[i], os.path.join(augmented_folder, os.path.basename(img_paths[i])))

def main():
    # Transformation
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Training 
    species_dataset = SpeciesDataset("dataset-3/Species", transform=transform)
    dataset_size = len(species_dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    train_split = int(0.8 * dataset_size)
    train_indices, val_indices = indices[:train_split], indices[train_split:]
    
    train_loader = DataLoader(species_dataset, batch_size=16, sampler=SubsetRandomSampler(train_indices))  # Smaller batch size
    val_loader = DataLoader(species_dataset, batch_size=16, sampler=SubsetRandomSampler(val_indices))
    
    # Model Setup with Transfer Learning
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 6) 
    model.to(device)
    
    # Train Model
    train_model(model, train_loader, val_loader, num_epochs=20, device=device)
    
    # Prediction Dataset
    disease_dataset = DiseaseDataset("dataset-3/Diseases", transform=transform)
    disease_loader = DataLoader(disease_dataset, batch_size=16, shuffle=False)
    
    # Predict and Create Augmented Dataset
    predict_and_compose(model, disease_loader, device)

if __name__ == "__main__":
    main()