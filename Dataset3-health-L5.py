## Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from PIL import Image
import os
import numpy as np
from tqdm import tqdm

### Dataset
datasetName = "Dataset-3"
### Architecture
architectureName = "SingleTaskCNN_Health"
### Number of layers
layersNum = 5
### Running info
print(datasetName, architectureName, layersNum)

## Load Dataset
class HealthDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.health_labels = []
        
        health_map = {
            "Black_Rot": 0, "Brown_Spot": 1, "Downy_Mildew": 2, 
            "Healthy_Leaves": 3, "Mites_Disease": 4
        }
        species_list = ["Ak", "Ala_Idris", "Buzgulu", "Nazli", "unknown"]
        species_list = sorted(species_list, key=len, reverse=True)
        
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                species = None
                for s in species_list:
                    if folder_name.startswith(s):
                        if len(folder_name) == len(s) or folder_name[len(s)] == '_':
                            species = s
                            break
                
                if species is None:
                    print(f"Skipping folder {folder_name}: No matching species")
                    continue
                
                health_part = folder_name[len(species):].strip('_')
                if not health_part:
                    print(f"Skipping folder {folder_name}: No health condition specified")
                    continue
                
                if health_part in health_map:
                    health_label = health_map[health_part]
                else:
                    print(f"Skipping folder {folder_name}: Unknown health '{health_part}'")
                    continue
                
                for img_name in os.listdir(folder_path):
                    img_path = os.path.join(folder_path, img_name)
                    if img_path.lower().endswith(('png', 'jpg', 'jpeg')):
                        try:
                            Image.open(img_path).verify()
                            self.image_paths.append(img_path)
                            self.health_labels.append(health_label)
                        except (IOError, SyntaxError) as e:
                            print(f"Skipping corrupted image: {img_path}")
                            continue
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        health_label = torch.tensor(self.health_labels[idx], dtype=torch.long)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, health_label

## Single Task Health CNN
class SingleTaskCNN_Health(nn.Module):
    def __init__(self):
        super(SingleTaskCNN_Health, self).__init__()
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
            nn.Linear(256, 5)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        health_pred = self.head(features)
        return health_pred

## Metric Calculations
def compute_metrics(preds, labels, num_classes, task_name="Health"):
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
        
        for images, health_labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            images = images.to(device)
            health_labels = health_labels.to(device)
            
            optimizer.zero_grad()
            health_pred = model(images)
            
            loss = criterion(health_pred, health_labels)
            
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(health_pred.data, 1)
            total_samples += health_labels.size(0)
            correct += (predicted == health_labels).sum().item()
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total_samples
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total_samples = 0
        
        with torch.no_grad():
            for images, health_labels in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                images = images.to(device)
                health_labels = health_labels.to(device)
                
                health_pred = model(images)
                loss = criterion(health_pred, health_labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(health_pred.data, 1)
                val_total_samples += health_labels.size(0)
                val_correct += (predicted == health_labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total_samples
        
        # Results per epoch
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training Loss: {train_loss:.4f}, Health Acc: {train_acc:.2f}%')
        print(f'Validation Loss: {val_loss:.4f}, Health Acc: {val_acc:.2f}%')
        print('-------------------')

def main():
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    root_dir = "dataset-3/Augmented_Dataset"
    dataset = HealthDataset(root_dir, transform=transform)
    
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.seed(42)
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
    model = SingleTaskCNN_Health().to(device)
    
    train_model(model, train_loader, val_loader, num_epochs=20, device=device)
    
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total_samples = 0
    criterion = nn.CrossEntropyLoss()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, health_labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            health_labels = health_labels.to(device)
            
            health_pred = model(images)
            loss = criterion(health_pred, health_labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(health_pred.data, 1)
            test_total_samples += health_labels.size(0)
            test_correct += (predicted == health_labels).sum().item()
            
            all_preds.append(predicted)
            all_labels.append(health_labels)
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    health_metrics = compute_metrics(all_preds, all_labels, num_classes=5, task_name="Health")
    
    test_loss = test_loss / len(test_loader)
    test_acc = 100 * test_correct / test_total_samples
    
    # Results
    print('Test Results:')
    print(f'Test Loss: {test_loss:.4f}, Health Acc: {test_acc:.2f}%')
    print('\nHealth Classification Metrics:')
    for metric, value in health_metrics.items():
        print(f'{metric}: {value:.4f}')
    
    # Save Model
    torch.save(model.state_dict(), 'Dataset3-health-L5.pth')

if __name__ == "__main__":
    main()