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
datasetName = "Dataset-1"
### Architecture
architectureName = "SingleTaskCNN_Health"
### Number of layers
layersNum = 3
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
            "Healthy": 0, "Powdery_mildew": 1, "Black_rot": 2, "Esca": 3, "Blight": 4,
            "Bacterial_spot": 5, "Leaf_scorch": 6, "Leaf_Mold": 7, "Septoria_leaf_spot": 8,
            "Target_Spot": 9, "Spider_mites": 10, "Yellow_Leaf_Curl_Virus": 11,
            "Leaf_blight": 12
        }
        
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                if folder_name.startswith("Pepper,_bell___"):
                    health_status = folder_name.replace("Pepper,_bell___", "")
                else:
                    match = re.match(r"(Cherry|Grape|Strawberry|Tomato)_(.*)", folder_name)
                    if match:
                        health_status = match.group(2)
                    else:
                        print(f"Skipping folder {folder_name}: Unknown format")
                        continue
                
                health_status = re.sub(r'\(.*\)', '', health_status)
                health_status = re.sub(r'_+', '_', health_status).strip('_')
                
                if health_status.lower() == "healthy":
                    health_label = health_map["Healthy"]
                elif health_status in health_map:
                    health_label = health_map[health_status]
                elif "Early_blight" in health_status or "Late_blight" in health_status:
                    health_label = health_map["Blight"]
                else:
                    found = False
                    for key in health_map:
                        if key.lower() in health_status.lower():
                            health_label = health_map[key]
                            found = True
                            break
                    if not found:
                        print(f"Skipping folder {folder_name}: Unknown health status '{health_status}'")
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
            nn.AdaptiveAvgPool2d((1, 1))                  
        )
        
        # Head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 1 * 1, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 13)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        health_pred = self.head(features)
        return health_pred

# Metric Calculations
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
        
        for images, health_labels in tqdm.tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
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
            for images, health_labels in tqdm.tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
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
    
    root_dir = "dataset-1"
    dataset = HealthDataset(root_dir, transform=transform)
    
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
    model = SingleTaskCNN_Health()
    
    train_model(model, train_loader, val_loader, num_epochs=20, device=device)
    
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total_samples = 0
    criterion = nn.CrossEntropyLoss()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, health_labels in tqdm.tqdm(test_loader, desc="Testing"):
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
    
    health_metrics = compute_metrics(all_preds, all_labels, num_classes=13, task_name="Health")
    
    test_loss = test_loss / len(test_loader)
    test_acc = 100 * test_correct / test_total_samples
    
    # Results
    print('Test Results:')
    print(f'Test Loss: {test_loss:.4f}, Health Acc: {test_acc:.2f}%')
    print('\nHealth Classification Metrics:')
    for metric, value in health_metrics.items():
        print(f'{metric}: {value:.4f}')
    
    # Save Model
    torch.save(model.state_dict(), 'Dataset1-health-L3.pth')

if __name__ == "__main__":
    main()