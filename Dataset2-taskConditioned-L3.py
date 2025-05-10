## Imports
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from PIL import Image
import os
import re
import numpy as np

### Dataset
datasetName = "Dataset-2"
### Architecture
architectureName = "TaskConditionedCNN"
### Number of layers
layersNum = 3
### Running info
print(datasetName, architectureName, layersNum)

## Load Dataset
class MultiTaskDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.species_labels = []
        self.health_labels = []
        
        species_map = {"Camphor": 0, "HariTaki": 1, "Neem": 2, "Sojina": 3}
        health_map = {
            "Bacterial Spot": 0, "Healthy Leaf": 1, "Shot Hole": 2, 
            "Yellow Leaf": 3, "Powdery Mildew": 4
        }
        
        for species_folder in os.listdir(root_dir):
            species_path = os.path.join(root_dir, species_folder)
            if os.path.isdir(species_path) and species_folder in species_map:
                species_label = species_map[species_folder]
                
                for disease_folder in os.listdir(species_path):
                    disease_path = os.path.join(species_path, disease_folder)
                    if os.path.isdir(disease_path) and disease_folder in health_map:
                        health_label = health_map[disease_folder]
                        
                        for img_name in os.listdir(disease_path):
                            img_path = os.path.join(disease_path, img_name)
                            if img_path.lower().endswith(('png', 'jpg', 'jpeg')):
                                try:
                                    Image.open(img_path).verify()
                                    self.image_paths.append(img_path)
                                    self.species_labels.append(species_label)
                                    self.health_labels.append(health_label)
                                except (IOError, SyntaxError) as e:
                                    print(f"Skipping corrupted image: {img_path}")
                                    continue
                    else:
                        print(f"Skipping folder {disease_folder}: Unknown disease")
            else:
                print(f"Skipping folder {species_folder}: Unknown species")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        species_label = torch.tensor(self.species_labels[idx], dtype=torch.long)
        health_label = torch.tensor(self.health_labels[idx], dtype=torch.long)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, (species_label, health_label)

## Task Conditioned CNN
class TaskConditionedCNN(nn.Module):
    def __init__(self):
        super(TaskConditionedCNN, self).__init__()
        # Shared Backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),                           
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),                           
            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((1, 1))                  
        )
        
        # Species head
        self.species_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 1 * 1, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 4) 
        )
        
        # Health head conditioned on species
        self.health_condition = nn.Sequential(
            nn.Linear(4, 32),  
            nn.ReLU()
        )
        self.health_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 1 * 1 + 32, 64), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 5)  
        )
    
    def forward(self, x):
        features = self.backbone(x)  
        
        # Species prediction
        species_pred = self.species_head(features)  
        
        # Health conditioned on species
        species_condition = self.health_condition(species_pred)  
        combined_features = torch.cat((features.view(features.size(0), -1), species_condition), dim=1)  
        health_pred = self.health_head(combined_features) 
        
        return species_pred, health_pred

## Metric Calculations
def compute_metrics(preds, labels, num_classes, task_name=""):
    preds = preds.cpu()
    labels = labels.cpu()
    
    tp = torch.zeros(num_classes)
    fp = torch.zeros(num_classes)
    fn = torch.zeros(num_classes)
    tn = torch.zeros(num_classes)
    
    for c in range(num_classes):
        tp[c] = ((preds == c) & (labels == c)).sum().float()
        fp[c] = ((preds == c) & (labels != c)).sum().float()
        fn[c] = ((preds != c) & (labels == c)).sum().float()
        tn[c] = ((preds != c) & (labels != c)).sum().float()
    
    tp_sum = tp.sum().item()
    fp_sum = fp.sum().item()
    fn_sum = fn.sum().item()
    tn_sum = tn.sum().item()
    total_samples = len(labels)
    
    print("FP/FN checking")
    print(f"FP per class: {fp}")
    print(f"FN per class: {fn}")
    
    print(f"\n{task_name} Raw Metrics:")
    print(f"True Positives (TP): {tp_sum:.0f}")
    print(f"False Positives (FP): {fp_sum:.0f}")
    print(f"False Negatives (FN): {fn_sum:.0f}")
    print(f"True Negatives (TN): {tn_sum:.0f}")
    print(f"Total Samples: {total_samples}")
    
    # Avoid div 0 (+ 1e-10)
    precision = tp.sum() / (tp.sum() + fp.sum() + 1e-10)
    recall = tp.sum() / (tp.sum() + fn.sum() + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    accuracy = tp.sum() / (total_samples + 1e-10)
    specificity = tn.sum() / (tn.sum() + fp.sum() + 1e-10)
    
    return {
        'F1-score': f1.item(),
        'Precision': precision.item(),
        'Recall': recall.item(),
        'Accuracy': accuracy.item(),
        'Specificity': specificity.item()
    }

## Model Training
def train_model(model, train_loader, val_loader, num_epochs, device):
    criterion_species = nn.CrossEntropyLoss()
    criterion_health = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_species_loss = 0.0
        running_health_loss = 0.0
        species_correct = 0
        health_correct = 0
        total_samples = 0
        
        for images, (species_labels, health_labels) in tqdm.tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            images = images.to(device)
            species_labels = species_labels.to(device)
            health_labels = health_labels.to(device)
            
            optimizer.zero_grad()
            species_pred, health_pred = model(images)
            
            loss_species = criterion_species(species_pred, species_labels)
            loss_health = criterion_health(health_pred, health_labels)
            loss = loss_species + loss_health
            
            loss.backward()
            optimizer.step()
            
            _, species_predicted = torch.max(species_pred.data, 1)
            _, health_predicted = torch.max(health_pred.data, 1)
            
            total_samples += species_labels.size(0)
            species_correct += (species_predicted == species_labels).sum().item()
            health_correct += (health_predicted == health_labels).sum().item()
            
            running_loss += loss.item()
            running_species_loss += loss_species.item()
            running_health_loss += loss_health.item()
        
        train_loss = running_loss / len(train_loader)
        train_species_loss = running_species_loss / len(train_loader)
        train_health_loss = running_health_loss / len(train_loader)
        train_species_acc = 100 * species_correct / total_samples
        train_health_acc = 100 * health_correct / total_samples
        
        model.eval()
        val_loss = 0.0
        val_species_loss = 0.0
        val_health_loss = 0.0
        val_species_correct = 0
        val_health_correct = 0
        val_total_samples = 0
        
        with torch.no_grad():
            for images, (species_labels, health_labels) in tqdm.tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                images = images.to(device)
                species_labels = species_labels.to(device)
                health_labels = health_labels.to(device)
                
                species_pred, health_pred = model(images)
                
                loss_species = criterion_species(species_pred, species_labels)
                loss_health = criterion_health(health_pred, health_labels)
                val_loss += (loss_species + loss_health).item()
                val_species_loss += loss_species.item()
                val_health_loss += loss_health.item()
                
                _, species_predicted = torch.max(species_pred.data, 1)
                _, health_predicted = torch.max(health_pred.data, 1)
                
                val_total_samples += species_labels.size(0)
                val_species_correct += (species_predicted == species_labels).sum().item()
                val_health_correct += (health_predicted == health_labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_species_loss = val_species_loss / len(val_loader)
        val_health_loss = val_health_loss / len(val_loader)
        val_species_acc = 100 * val_species_correct / val_total_samples
        val_health_acc = 100 * val_health_correct / val_total_samples
        
        # Results per epoch
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training Total Loss: {train_loss:.4f}, Species Loss: {train_species_loss:.4f}, Health Loss: {train_health_loss:.4f}')
        print(f'Training Species Acc: {train_species_acc:.2f}%, Health Acc: {train_health_acc:.2f}%')
        print(f'Validation Total Loss: {val_loss:.4f}, Species Loss: {val_species_loss:.4f}, Health Loss: {val_health_loss:.4f}')
        print(f'Validation Species Acc: {val_species_acc:.2f}%, Health Acc: {val_health_acc:.2f}%')
        print('-------------------')

def main():
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    root_dir = "dataset-2"
    dataset = MultiTaskDataset(root_dir, transform=transform)
    
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    
    train_split = int(np.floor(0.8 * dataset_size))
    val_split = int(np.floor(0.9 * dataset_size))
    
    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]
    
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    batch_size = 32
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TaskConditionedCNN()
    
    train_model(model, train_loader, val_loader, num_epochs=20, device=device)
    
    model.eval()
    test_loss = 0.0
    test_species_loss = 0.0
    test_health_loss = 0.0
    test_species_correct = 0
    test_health_correct = 0
    test_total_samples = 0
    criterion_species = nn.CrossEntropyLoss()
    criterion_health = nn.CrossEntropyLoss()
    
    all_species_preds = []
    all_species_labels = []
    all_health_preds = []
    all_health_labels = []
    
    with torch.no_grad():
        for images, (species_labels, health_labels) in tqdm.tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            species_labels = species_labels.to(device)
            health_labels = health_labels.to(device)
            
            species_pred, health_pred = model(images)
            
            loss_species = criterion_species(species_pred, species_labels)
            loss_health = criterion_health(health_pred, health_labels)
            test_loss += (loss_species + loss_health).item()
            test_species_loss += loss_species.item()
            test_health_loss += loss_health.item()
            
            _, species_predicted = torch.max(species_pred.data, 1)
            _, health_predicted = torch.max(health_pred.data, 1)
            
            test_total_samples += species_labels.size(0)
            test_species_correct += (species_predicted == species_labels).sum().item()
            test_health_correct += (health_predicted == health_labels).sum().item()
            
            all_species_preds.append(species_predicted)
            all_species_labels.append(species_labels)
            all_health_preds.append(health_predicted)
            all_health_labels.append(health_labels)
    
    all_species_preds = torch.cat(all_species_preds)
    all_species_labels = torch.cat(all_species_labels)
    all_health_preds = torch.cat(all_health_preds)
    all_health_labels = torch.cat(all_health_labels)
    
    species_metrics = compute_metrics(all_species_preds, all_species_labels, num_classes=4, task_name="Species")
    health_metrics = compute_metrics(all_health_preds, all_health_labels, num_classes=5, task_name="Health")
    combined_true = all_species_labels * 5 + all_health_labels  
    combined_pred = all_species_preds * 5 + all_health_preds
    combined_metrics = compute_metrics(combined_pred, combined_true, num_classes=20, task_name="Combined Species_Health")
    
    test_loss = test_loss / len(test_loader)
    test_species_loss = test_species_loss / len(test_loader)
    test_health_loss = test_health_loss / len(test_loader)
    test_species_acc = 100 * test_species_correct / test_total_samples
    test_health_acc = 100 * test_health_correct / test_total_samples
    
    # Results
    print('Test Results:')
    print(f'Test Total Loss: {test_loss:.4f}, Species Loss: {test_species_loss:.4f}, Health Loss: {test_health_loss:.4f}')
    print(f'Test Species Acc: {test_species_acc:.2f}%, Health Acc: {test_health_acc:.2f}%')
    print('\nSpecies Classification Metrics:')
    for metric, value in species_metrics.items():
        print(f'{metric}: {value:.4f}')
    print('\nHealth Classification Metrics:')
    for metric, value in health_metrics.items():
        print(f'{metric}: {value:.4f}')
    print('\nCombined Species_Health Classification Metrics:')
    for metric, value in combined_metrics.items():
        print(f'{metric}: {value:.4f}')
    
    # Save Model
    torch.save(model.state_dict(), 'Dataset2-taskConditioned-L3.pth')

if __name__ == "__main__":
    main()