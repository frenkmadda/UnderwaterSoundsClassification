import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
from tqdm import tqdm

# Dataset personalizzato
class SpectrogramDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = self.data_frame.iloc[idx, 0]
        image = Image.open(img_name).convert('RGB')
        label = int(self.data_frame.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label

# Definizione della CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)  # Assuming input image size is 224x224
        self.fc2 = nn.Linear(128, 2)  # 2 classes: target and non-target

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Funzione di training
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, patience=5):
    best_model_wts = model.state_dict()
    best_loss = float('inf')
    early_stopping_counter = 0

    metrics = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Imposta il modello in modalità training
            else:
                model.eval()   # Imposta il modello in modalità evaluation

            running_loss = 0.0
            running_corrects = 0
            all_labels = []
            all_preds = []

            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch}"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            precision = precision_score(all_labels, all_preds, average='binary')
            recall = recall_score(all_labels, all_preds, average='binary')
            f1 = f1_score(all_labels, all_preds, average='binary')

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Precision: {precision:.4f} Recall: {recall:.4f} F1: {f1:.4f}')

            if phase == 'val':
                metrics.append([epoch, epoch_loss, epoch_acc.item(), precision, recall, f1])

                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = model.state_dict()
                    early_stopping_counter = 0
                    torch.save(model.state_dict(), f'checkpoint_epoch_{epoch}.pt')
                else:
                    early_stopping_counter += 1

                if early_stopping_counter >= patience:
                    print('Early stopping')
                    model.load_state_dict(best_model_wts)
                    return model, pd.DataFrame(metrics, columns=['Epoch', 'Loss', 'Accuracy', 'Precision', 'Recall', 'F1'])

        print()

    model.load_state_dict(best_model_wts)
    return model, pd.DataFrame(metrics, columns=['Epoch', 'Loss', 'Accuracy', 'Precision', 'Recall', 'F1'])

# Parametri
train_csv_file = 'path_to_your_train_csv_file.csv'
val_csv_file = 'path_to_your_val_csv_file.csv'
batch_size = 32
num_epochs = 50
patience = 5
learning_rate = 0.001

# Creazione dei dataset e dataloader
image_datasets = {
    'train': SpectrogramDataset(train_csv_file),
    'val': SpectrogramDataset(val_csv_file)
}
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4),
    'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=4)
}

# Inizializzazione del modello, della loss function e dell'optimizer
model = SimpleCNN()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Addestramento del modello
model, metrics_df = train_model(model, dataloaders, criterion, optimizer, num_epochs=num_epochs, patience=patience)

# Salvataggio del modello migliore
torch.save(model.state_dict(), 'best_model.pt')

# Salvataggio delle metriche
metrics_df.to_csv('training_metrics.csv', index=False)
