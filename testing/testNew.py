import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pandas as pd
from tqdm import tqdm
import platform
from torchvision.models import AlexNet_Weights, GoogLeNet_Weights, ResNet50_Weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, average_precision_score
import csv
import matplotlib.pyplot as plt

def load_model(model_path, model_type, num_classes, device):
    if model_type == 'googlenet':
        model = models.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.aux1 = None
        model.aux2 = None
    elif model_type == 'alexnet':
        model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_type == 'resnet50':
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def process_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def calculate_metrics(labels, outputs, average):
    accuracy = accuracy_score(labels, outputs)
    precision = precision_score(labels, outputs, average=average)
    recall = recall_score(labels, outputs, average=average, zero_division=1)
    f1 = f1_score(labels, outputs, average=average)
    return accuracy, precision, recall, f1

def save_metrics(accuracy, precision, recall, f1, path):
    with open(path, 'w', newline='') as csvfile:
        fieldnames = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            'test_accuracy': accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1
        })

def save_predictions(targets, predicted_probs, path):
    with open(path, 'w', newline='') as csvfile:
        fieldnames = ['true_label', 'predicted_prob']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for true_label, predicted_prob in zip(targets, predicted_probs):
            writer.writerow({
                'true_label': true_label,
                'predicted_prob': predicted_prob
            })

def get_device():
    if platform.system() == 'Windows':
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def test_model(model_path, model_type, average, num_classes, is_binary=True):
    csv_path = '../final_dataset/test/df_paths_test.csv'
    device = get_device()
    model = load_model(model_path, model_type, num_classes, device)
    df = pd.read_csv(csv_path)
    predictions = []
    targets = []
    predicted_probs = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Testing"):
        image_path = os.path.join('../', row['FilePath'])
        image = process_image(image_path).to(device)

        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            prob = torch.softmax(outputs, dim=1)

        if is_binary:
            predicted_label = 1 if predicted.item() == 1 else 0
            target_label = 1 if row['Label'] == 'Target' else 0
            predicted_prob = prob[0, 1].item()
        else:
            predicted_label = predicted.item()
            target_label = row['Classe']
            predicted_prob = prob[0, predicted.item()].item()

        predictions.append(predicted_label)
        targets.append(target_label)
        predicted_probs.append(predicted_prob)

    if not is_binary:
        unique_labels = df['Classe'].unique()
        label_to_index = {label: index for index, label in enumerate(unique_labels)}
        targets = [label_to_index[label] for label in targets]

    test_accuracy, test_precision, test_recall, test_f1 = calculate_metrics(targets, predictions, average)
    model_name = model_path.split('/')[2]
    save_metrics(test_accuracy, test_precision, test_recall, test_f1, f'test_results/{model_name}_metrics.csv')
    save_predictions(targets, predicted_probs, f'test_results/{model_name}_predictions.csv')
    print(f'Accuracy: {test_accuracy:.4f} - Precision: {test_precision:.4f} - Recall: {test_recall:.4f} - F1: {test_f1:.4f}')


def plot_precision_recall(csv_prediction_path):
    data = pd.read_csv(csv_prediction_path)

    true_labels = data['true_label']
    predicted_probs = data['predicted_prob']

    # Calcola precision e recall
    precision, recall, _ = precision_recall_curve(true_labels, predicted_probs)
    average_precision = average_precision_score(true_labels, predicted_probs)

    # Calcola il livello di chance
    chance_level = sum(true_labels) / len(true_labels)

    # Plotta la curva precision-recall
    plt.figure()
    plt.step(recall, precision, where='post', label='Precision-Recall curve')
    plt.axhline(y=chance_level, color='r', linestyle='--', label='Chance level')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall curve: AP={average_precision:0.2f}')
    plt.legend()
    plt.show()
