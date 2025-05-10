## Imports
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from PIL import Image
import os
import numpy as np

### Dataset
datasetName = "Dataset-2"
### Architecture
architectureName = "SingleTaskCNN_Species"
### Number of layers
layersNum = 5
### Running info
print(datasetName, architectureName, layersNum)

## Load Dataset
class SpeciesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.species_labels = []
        
        species_map = {"Camphor": 0, "HariTaki": 1, "Neem": 2, "Sojina": 3}
        
        for species in os.listdir(root_dir):
            species_path = os.path.join(root_dir, species)
            if os.path.isdir(species_path):
                if species not in species_map:
                    print(f"Skipping folder {species}: Unknown species")
                    continue
                species_label = species_map[species]
                
                for disease in os.listdir(species_path):
                    disease_path = os.path.join(species_path, disease)
                    if os.path.isdir(disease_path):
                        for img_name in os.listdir(disease_path):
                            img_path = os.path.join(disease_path, img_name)
                            if img_path.lower().endswith(('png', 'jpg', 'jpeg')):
                                try:
                                    Image.open(img_path).verify()
                                    self.image_paths.append(img_path)
                                    self.species_labels.append(species_label)
                                except (IOError, SyntaxError) as e:
                                    print(f"Skipping corrupted image: {img_path}")
                                    continue
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        species_label = torch.tensor(self.species_labels[idx], dtype=torch.long)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, species_label

## Single Task Species CNN
class SingleTaskCNN_Species(nn.Module):
    def __init__(self):
        super(SingleTaskCNN_Species, self).__init__()
        # Backbone
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
            nn.MaxPool2d(2, 2),                            
            nn.Conv2d(128, 256, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),                            
            nn.Conv2d(256, 512, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((1, 1))                   
        )
        
        # Head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 1 * 1, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 4)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        species_pred = self.head(features)
        return species_pred

## Metric Calculations
def compute_metrics(preds, labels, num_classes, task_name="Species"):
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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total_samples = 0
        
        for images, species_labels in tqdm.tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            images = images.to(device)
            species_labels = species_labels.to(device)
            
            optimizer.zero_grad()
            species_pred = model(images)
            
            loss = criterion(species_pred, species_labels)
            
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(species_pred.data, 1)
            total_samples += species_labels.size(0)
            correct += (predicted == species_labels).sum().item()
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total_samples
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total_samples = 0
        
        with torch.no_grad():
            for images, species_labels in tqdm.tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                images = images.to(device)
                species_labels = species_labels.to(device)
                
                species_pred = model(images)
                loss = criterion(species_pred, species_labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(species_pred.data, 1)
                val_total_samples += species_labels.size(0)
                val_correct += (predicted == species_labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total_samples
        
        # Results per epoch
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training Loss: {train_loss:.4f}, Species Acc: {train_acc:.2f}%')
        print(f'Validation Loss: {val_loss:.4f}, Species Acc: {val_acc:.2f}%')
        print('-------------------')

def main():
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    root_dir = "dataset-2"
    dataset = SpeciesDataset(root_dir, transform=transform)
    
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
    model = SingleTaskCNN_Species()
    
    train_model(model, train_loader, val_loader, num_epochs=20, device=device)
    
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total_samples = 0
    criterion = nn.CrossEntropyLoss()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, species_labels in tqdm.tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            species_labels = species_labels.to(device)
            
            species_pred = model(images)
            loss = criterion(species_pred, species_labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(species_pred.data, 1)
            test_total_samples += species_labels.size(0)
            test_correct += (predicted == species_labels).sum().item()
            
            all_preds.append(predicted)
            all_labels.append(species_labels)
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    species_metrics = compute_metrics(all_preds, all_labels, num_classes=4, task_name="Species")
    
    test_loss = test_loss / len(test_loader)
    test_acc = 100 * test_correct / test_total_samples
    
    # Results 
    print('Test Results:')
    print(f'Test Loss: {test_loss:.4f}, Species Acc: {test_acc:.2f}%')
    print('\nSpecies Classification Metrics:')
    for metric, value in species_metrics.items():
        print(f'{metric}: {value:.4f}')
    
    # Save Model
    torch.save(model.state_dict(), 'Dataset2-species-L5.pth')

if __name__ == "__main__":
    main()