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
datasetName = "Dataset-3"
### Architecture
architectureName = "MultiModalCNN-brightness"
### Number of layers
layersNum = 7
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
        
        species_map = {"Ak": 0, "Ala_Idris": 1, "Buzgulu": 2, "Nazli": 3, "unknown": 4}
        health_map = {
            "Black_Rot": 0, "Brown_Spot": 1, "Downy_Mildew": 2, 
            "Healthy_Leaves": 3, "Mites_Disease": 4
        }
        self.species_list = sorted(species_map.keys(), key=len, reverse=True)
        self.health_list = list(health_map.keys())
        
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                species = None
                for s in self.species_list:
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
                    health = health_part
                else:
                    print(f"Skipping folder {folder_name}: Unknown health '{health_part}'")
                    continue
                
                species_label = species_map[species]
                health_label = health_map[health]
                
                for img_name in os.listdir(folder_path):
                    img_path = os.path.join(folder_path, img_name)
                    if img_path.lower().endswith(('png', 'jpg', 'jpeg')):
                        try:
                            Image.open(img_path).verify()
                            self.image_paths.append(img_path)
                            self.species_labels.append(species_label)
                            self.health_labels.append(health_label)
                        except (IOError, SyntaxError) as e:
                            print(f"Skipping corrupted image: {img_path}")
                            continue
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        species_label = torch.tensor(self.species_labels[idx], dtype=torch.long)
        health_label = torch.tensor(self.health_labels[idx], dtype=torch.long)
        image = Image.open(img_path).convert('RGB')
        
        # Brightness metadata
        img_array = np.array(image)
        brightness_mean = img_array.mean() / 255.0
        brightness_std = img_array.std() / 255.0
        metadata = torch.tensor([brightness_mean, brightness_std], dtype=torch.float)
        
        if self.transform:
            image = self.transform(image)
        
        return image, metadata, (species_label, health_label)

## Multi Modal brightness CNN
class MultiModalCNN(nn.Module):
    def __init__(self):
        super(MultiModalCNN, self).__init__()
        # Image Backbone
        self.image_backbone = nn.Sequential(
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
            nn.MaxPool2d(2, 2),                            
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),                            
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(3, 3)                             
        )
        
        # Metadata Branch
        self.metadata_branch = nn.Sequential(
            nn.Linear(2, 64),  
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # Species head
        self.species_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 1 * 1 + 128, 256), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 5)
        )
        
        # Health head
        self.health_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 1 * 1 + 128, 256),  
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 5)
        )
        
        # Metadata head
        self.metadata_head = nn.Linear(512 * 1 * 1 + 128, 2)  
    
    def forward(self, image, metadata):
        image_features = self.image_backbone(image) 
        metadata_features = self.metadata_branch(metadata)  
        
        image_flat = image_features.view(image_features.size(0), -1) 
        combined_features = torch.cat((image_flat, metadata_features), dim=1)  
        
        species_pred = self.species_head(combined_features)
        health_pred = self.health_head(combined_features)
        metadata_pred = self.metadata_head(combined_features)
        
        return species_pred, health_pred, metadata_pred

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
def train_model(model, train_loader, val_loader, num_epochs, device, gamma=0.1):
    criterion_species = nn.CrossEntropyLoss()
    criterion_health = nn.CrossEntropyLoss()
    criterion_metadata = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_species_loss = 0.0
        running_health_loss = 0.0
        running_metadata_loss = 0.0
        species_correct = 0
        health_correct = 0
        total_samples = 0
        
        for images, metadata, (species_labels, health_labels) in tqdm.tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            images = images.to(device)
            metadata = metadata.to(device)
            species_labels = species_labels.to(device)
            health_labels = health_labels.to(device)
            
            optimizer.zero_grad()
            species_pred, health_pred, metadata_pred = model(images, metadata)
            
            loss_species = criterion_species(species_pred, species_labels)
            loss_health = criterion_health(health_pred, health_labels)
            loss_metadata = criterion_metadata(metadata_pred, metadata)
            loss = loss_species + loss_health + gamma * loss_metadata
            
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
            running_metadata_loss += loss_metadata.item()
        
        train_loss = running_loss / len(train_loader)
        train_species_loss = running_species_loss / len(train_loader)
        train_health_loss = running_health_loss / len(train_loader)
        train_metadata_loss = running_metadata_loss / len(train_loader)
        train_species_acc = 100 * species_correct / total_samples
        train_health_acc = 100 * health_correct / total_samples
        
        model.eval()
        val_loss = 0.0
        val_species_loss = 0.0
        val_health_loss = 0.0
        val_metadata_loss = 0.0
        val_species_correct = 0
        val_health_correct = 0
        val_total_samples = 0
        
        with torch.no_grad():
            for images, metadata, (species_labels, health_labels) in tqdm.tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                images = images.to(device)
                metadata = metadata.to(device)
                species_labels = species_labels.to(device)
                health_labels = health_labels.to(device)
                
                species_pred, health_pred, metadata_pred = model(images, metadata)
                
                loss_species = criterion_species(species_pred, species_labels)
                loss_health = criterion_health(health_pred, health_labels)
                loss_metadata = criterion_metadata(metadata_pred, metadata)
                val_loss += (loss_species + loss_health + gamma * loss_metadata).item()
                val_species_loss += loss_species.item()
                val_health_loss += loss_health.item()
                val_metadata_loss += loss_metadata.item()
                
                _, species_predicted = torch.max(species_pred.data, 1)
                _, health_predicted = torch.max(health_pred.data, 1)
                
                val_total_samples += species_labels.size(0)
                val_species_correct += (species_predicted == species_labels).sum().item()
                val_health_correct += (health_predicted == health_labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_species_loss = val_species_loss / len(val_loader)
        val_health_loss = val_health_loss / len(val_loader)
        val_metadata_loss = val_metadata_loss / len(val_loader)
        val_species_acc = 100 * val_species_correct / val_total_samples
        val_health_acc = 100 * val_health_correct / val_total_samples
        
        # Results per epoch
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training Total Loss: {train_loss:.4f}, Species Loss: {train_species_loss:.4f}, Health Loss: {train_health_loss:.4f}, Metadata Loss: {train_metadata_loss:.4f}')
        print(f'Training Species Acc: {train_species_acc:.2f}%, Health Acc: {train_health_acc:.2f}%')
        print(f'Validation Total Loss: {val_loss:.4f}, Species Loss: {val_species_loss:.4f}, Health Loss: {val_health_loss:.4f}, Metadata Loss: {val_metadata_loss:.4f}')
        print(f'Validation Species Acc: {val_species_acc:.2f}%, Health Acc: {val_health_acc:.2f}%')
        print('-------------------')

def main():
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    root_dir = "dataset-3/Augmented_Dataset"
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
    model = MultiModalCNN()
    
    train_model(model, train_loader, val_loader, num_epochs=20, device=device, gamma=0.1)
    
    model.eval()
    test_loss = 0.0
    test_species_loss = 0.0
    test_health_loss = 0.0
    test_metadata_loss = 0.0
    test_species_correct = 0
    test_health_correct = 0
    test_total_samples = 0
    gamma = 0.1
    criterion_species = nn.CrossEntropyLoss()
    criterion_health = nn.CrossEntropyLoss()
    criterion_metadata = nn.MSELoss()
    
    all_species_preds = []
    all_species_labels = []
    all_health_preds = []
    all_health_labels = []
    
    with torch.no_grad():
        for images, metadata, (species_labels, health_labels) in tqdm.tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            metadata = metadata.to(device)
            species_labels = species_labels.to(device)
            health_labels = health_labels.to(device)
            
            species_pred, health_pred, metadata_pred = model(images, metadata)
            
            loss_species = criterion_species(species_pred, species_labels)
            loss_health = criterion_health(health_pred, health_labels)
            loss_metadata = criterion_metadata(metadata_pred, metadata)
            test_loss += (loss_species + loss_health + gamma * loss_metadata).item()
            test_species_loss += loss_species.item()
            test_health_loss += loss_health.item()
            test_metadata_loss += loss_metadata.item()
            
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
    
    species_metrics = compute_metrics(all_species_preds, all_species_labels, num_classes=5, task_name="Species")
    health_metrics = compute_metrics(all_health_preds, all_health_labels, num_classes=5, task_name="Health")
    combined_true = all_species_labels * 5 + all_health_labels
    combined_pred = all_species_preds * 5 + all_health_preds
    combined_metrics = compute_metrics(combined_pred, combined_true, num_classes=25, task_name="Combined Species_Health")
    
    test_loss = test_loss / len(test_loader)
    test_species_loss = test_species_loss / len(test_loader)
    test_health_loss = test_health_loss / len(test_loader)
    test_metadata_loss = test_metadata_loss / len(test_loader)
    test_species_acc = 100 * test_species_correct / test_total_samples
    test_health_acc = 100 * test_health_correct / test_total_samples
    
    # Results
    print('Test Results:')
    print(f'Test Total Loss: {test_loss:.4f}, Species Loss: {test_species_loss:.4f}, Health Loss: {test_health_loss:.4f}, Metadata Loss: {test_metadata_loss:.4f}')
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
    torch.save(model.state_dict(), 'Dataset3-multiModal-brightness-L7.pth')

if __name__ == "__main__":
    main()