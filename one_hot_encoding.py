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
from torchvision import models, transforms
from torchvision.models import GoogLeNet_Weights, AlexNet_Weights
import platform

from training import write_parameters_to_txt

class SpectrogramDataset(Dataset):
    def __init__(self, csv_file, classes, transform=None):
        self.classes = classes

        if classes == 2:
            self.data_frame = pd.read_csv(csv_file, usecols=['FilePath', 'Label'])
            self.data_frame['Label'] = self.data_frame['Label'].apply(lambda x: 1 if x == 'Target' else 0)
        else:
            self.data_frame = pd.read_csv(csv_file, usecols=['FilePath', 'Classe'])
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.data_frame['Classe'].unique())}
            self.data_frame['Classe'] = self.data_frame['Classe'].apply(lambda x: self.class_to_idx[x])

        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = self.data_frame.iloc[idx, 0]
        image = Image.open(img_name).convert('RGB')  # Convertire l'immagine in RGB

        label = int(self.data_frame.iloc[idx, 1])
        if self.classes != 2:
            label = torch.nn.functional.one_hot(torch.tensor(label), num_classes=self.classes).float()

        if self.transform:
            image = self.transform(image)

        return image, label

def train_model(model, dataloaders, criterion, optimizer, average, num_classes, num_epochs=25, patience=5):
    best_model_wts = model.state_dict()
    best_loss = float('inf')
    early_stopping_counter = 0

    metrics = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        epoch_metrics = {'epoch': epoch}

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Imposta il modello in modalità training
            else:
                model.eval()  # Imposta il modello in modalità evaluation

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
                    if num_classes == 2:
                        _, preds = torch.max(outputs, 1)
                    else:
                        preds = torch.argmax(outputs, dim=1)

                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                if num_classes == 2:
                    running_corrects += torch.sum(preds == labels.data)
                else:
                    running_corrects += torch.sum(preds == torch.argmax(labels, dim=1))
                all_labels.extend(torch.argmax(labels, dim=1).cpu().numpy() if num_classes != 2 else labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)
            precision = precision_score(all_labels, all_preds, average=average)
            recall = recall_score(all_labels, all_preds, average=average)
            f1 = f1_score(all_labels, all_preds, average=average)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Precision: {precision:.4f} Recall: {recall:.4f} F1: {f1:.4f}')

            epoch_metrics[f'{phase}_loss'] = epoch_loss
            epoch_metrics[f'{phase}_accuracy'] = epoch_acc.item()
            epoch_metrics[f'{phase}_precision'] = precision
            epoch_metrics[f'{phase}_recall'] = recall
            epoch_metrics[f'{phase}_f1'] = f1

            if phase == 'val':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = model.state_dict()
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1

                if early_stopping_counter >= patience:
                    print('Early stopping')
                    model.load_state_dict(best_model_wts)
                    metrics.append(epoch_metrics)
                    return model, pd.DataFrame(metrics)

        torch.save(model.state_dict(), f'{path_model}/checkpoint_epoch_{epoch}.pt')
        metrics.append(epoch_metrics)
        print()

    model.load_state_dict(best_model_wts)
    return model, pd.DataFrame(metrics)

if __name__ == "__main__":
    # Parametri
    train_csv_file = 'final_dataset/training/df_paths_train.csv'
    val_csv_file = 'final_dataset/validation/df_paths_val.csv'
    name_test = 'test11_alex_net_one_hot'
    path_model = f'models/{name_test}'
    batch_size = 32
    num_epochs = 50
    patience = 5
    learning_rate = 0.001
    classes = 38  # 2 o 38
    average = 'weighted'
    model_used = 'AlexNet'
    weights = 'IMAGENET1K_V1'

    parameters = {
        'path_model': path_model,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'patience': patience,
        'learning_rate': learning_rate,
        'classes': classes,
        'average': average,
        'model_used': model_used,
        'weights': weights
    }

    if not os.path.exists(path_model):
        os.makedirs(path_model)

    write_parameters_to_txt(parameters, f'{path_model}/parameters.txt')

    # Creazione dei dataset e dataloader con le trasformazioni necessarie
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        'train': SpectrogramDataset(train_csv_file, classes, transform=data_transforms['train']),
        'val': SpectrogramDataset(val_csv_file, classes, transform=data_transforms['val'])
    }
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=4)
    }

    if model_used == 'GoogLeNet':
        model = models.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, classes)
    elif model_used == 'AlexNet':
        model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, classes)

    # Spostare il modello sul dispositivo
    if platform.system() == 'Windows':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss() if classes > 2 else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Addestramento del modello
    model, metrics_df = train_model(model, dataloaders, criterion, optimizer, average, classes, num_epochs=num_epochs, patience=patience)

    # Salvataggio del modello migliore
    torch.save(model.state_dict(), f'{path_model}/best_model.pt')

    # Salvataggio delle metriche
    metrics_df.to_csv(f'results/{name_test}.csv', index=False)
