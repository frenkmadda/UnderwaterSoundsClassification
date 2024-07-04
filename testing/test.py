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
    """
    Load the specified model with weights and adjust for the number of classes.

    :param model_path: Path to the model file.
    :param model_type: Type of the model ('googlenet', 'alexnet', 'resnet50').
    :param num_classes: Number of classes for the model output.
    :param device: The device to load the model onto.

    :return: The loaded and adjusted model.
    """
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
    """
    Process an image for model prediction.

    :param image_path: Path to the image file.

    :return: The processed image tensor.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def calculate_metrics(labels, outputs, average):
    """
    Calculate accuracy, precision, recall, and F1 score for the predictions.

    :param labels: The true labels.
    :param outputs: The predicted labels.
    :param average: The type of averaging performed on the data ('binary', 'micro', 'macro', 'weighted').

    :return: The calculated metrics.
    """
    accuracy = accuracy_score(labels, outputs)
    precision = precision_score(labels, outputs, average=average)
    recall = recall_score(labels, outputs, average=average, zero_division=1)
    f1 = f1_score(labels, outputs, average=average)
    return accuracy, precision, recall, f1

def save_metrics(accuracy, precision, recall, f1, path):
    """
    Save the calculated metrics to a CSV file.

    :param accuracy: The accuracy score.
    :param precision: The precision score.
    :param recall: The recall score.
    :param f1: The F1 score.
    :param path: The path to save the CSV file.

    :return: None
    """
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
    """
    Save the true labels and predicted probabilities to a CSV file.

    :param targets: It contains the true labels.
    :param predicted_probs: It contains the predicted probabilities.
    :param path: The path to save the CSV file.

    :return: None
    """
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
    """
    Get the appropriate device for model inference.

    :return: The device ('cpu', 'cuda', or 'mps').
    """
    if platform.system() == 'Windows':
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def test_model(model_path, model_type, average, num_classes, is_binary=True):
    """
    Test a model with given parameters and save the metrics.

    :param model_path: Path to the model file.
    :param model_type: Type of the model ('googlenet', 'alexnet', 'resnet50').
    :param average: The type of averaging performed on the data ('binary', 'micro', 'macro', 'weighted').
    :param num_classes: Number of classes for the model output.
    :param is_binary: Flag indicating if the classification is binary.

    :return: None
    """
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

            if is_binary:
                prob = torch.softmax(outputs, dim=1)

        if is_binary:
            predicted_label = 1 if predicted.item() == 1 else 0
            target_label = 1 if row['Label'] == 'Target' else 0
            predicted_prob = prob[0, 1].item()
        else:
            predicted_label = predicted.item()
            target_label = row['Classe']

        if is_binary:
            predicted_probs.append(predicted_prob)

        predictions.append(predicted_label)
        targets.append(target_label)


    if not is_binary:
        unique_labels = df['Classe'].unique()
        label_to_index = {label: index for index, label in enumerate(unique_labels)}
        targets = [label_to_index[label] for label in targets]

    test_accuracy, test_precision, test_recall, test_f1 = calculate_metrics(targets, predictions, average)
    model_name = model_path.split('/')[2]
    save_metrics(test_accuracy, test_precision, test_recall, test_f1, f'test_results/{model_name}_metrics.csv')

    if is_binary:
        save_predictions(targets, predicted_probs, f'test_results/{model_name}_predictions.csv')

    print(f'Accuracy: {test_accuracy:.4f} - Precision: {test_precision:.4f} - Recall: {test_recall:.4f} - F1: {test_f1:.4f}')


def plot_precision_recall(csv_prediction_path):
    """
    Plots the precision-recall curve for the predictions in the CSV file.

    :param csv_prediction_path: The path to the CSV file containing the predictions.

    :return: None
    """
    data = pd.read_csv(csv_prediction_path)

    true_labels = data['true_label']
    predicted_probs = data['predicted_prob']

    precision, recall, _ = precision_recall_curve(true_labels, predicted_probs)
    average_precision = average_precision_score(true_labels, predicted_probs)

    chance_level = sum(true_labels) / len(true_labels)

    plt.figure()
    plt.step(recall, precision, where='post', label='Precision-Recall curve')
    plt.axhline(y=chance_level, color='r', linestyle='--', label='Chance level')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall curve: AP={average_precision:0.2f}')
    plt.legend()
    plt.show()
