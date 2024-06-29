import csv
import os
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms, models
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import AlexNet_Weights, GoogLeNet_Weights
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models
import torch.nn as nn
import os



def calculate_metrics(labels, outputs):
    predictions = (outputs >= 0.5).astype(int)  # Convert logits to binary predictions
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='micro')
    recall = recall_score(labels, predictions, average='micro')
    f1 = f1_score(labels, predictions, average='micro')
    return accuracy, precision, recall, f1



class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform
        self.label_to_index = {}  # New dictionary to map labels to indices
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row if present
            for row in reader:
                self.data.append(row[0])  # Assuming first column is image path
                label = str(row[1])  # Assuming second column is label
                self.labels.append(label)
                if label not in self.label_to_index:  # If label is not in dictionary, add it
                    self.label_to_index[label] = len(self.label_to_index)  # The value is the current size of the dictionary

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        label_index = self.label_to_index[label]  # Convert label from string to integer
        label_one_hot = F.one_hot(torch.tensor(label_index),
                                  num_classes=len(self.label_to_index)).float()  # Convert label index to one-hot tensor
        return image, label_one_hot.squeeze()



def load_model(model_path, device): #CAMBIA QUI IL MODELLO DA USARE IL NUMERO DI FEATURES
    # ResNet50  ---
    #model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    #model.fc = nn.Linear(model.fc.in_features, 2)
    #GoogleNet
    model = models.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1) #modello googlenet
    model.fc = nn.Linear(model.fc.in_features, 2)
    # AlexNet ---
    #model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
    #num_ftrs = model.classifier[6].in_features
    #model.classifier[6] = nn.Linear(num_ftrs, 38)
    #----
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_dataset_path = 'final_dataset/test/df_paths_test.csv'  # Update with actual path
test_dataset = CustomDataset(test_dataset_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, pin_memory=True)

model_path = 'models/test4_google_net/checkpoint_epoch_1.pt'  # CAMBIA QUI IL MODELLO DA USARE
model = load_model(model_path, device)
model.eval()

criterion = nn.BCEWithLogitsLoss()
running_test_loss = 0.0
all_labels = []
all_outputs = []

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Testing"):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(inputs)
        test_loss = criterion(outputs, labels)
        running_test_loss += test_loss.item()
        all_labels.extend(labels.cpu().numpy().tolist())
        all_outputs.extend(outputs.cpu().numpy().tolist())

# Convert lists to numpy arrays
all_labels = np.array(all_labels)
all_outputs = np.array(all_outputs)

# Reshape all_labels and all_outputs to match the batch size
all_labels = all_labels.reshape(-1, 38)
all_outputs = all_outputs.reshape(-1, 38)

test_loss = running_test_loss / len(test_loader)

# Assuming calculate_metrics is defined elsewhere
test_accuracy, test_precision, test_recall, test_f1 = calculate_metrics(all_labels, all_outputs)

# Creazione del file CSV nella directory dei checkpoint
checkpoint_dir = 'test_results/'  # Update with actual path
csv_output_path = os.path.join(checkpoint_dir, 'test4_results.csv') #   CAMBIA QUI IL NOME DEL FILE DEI RISULTATI
with open(csv_output_path, 'w', newline='') as csvfile:
    fieldnames = ['test_loss', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1
    })

print(f"Risultati del testing salvati in: {csv_output_path}")
