import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(model, optimizer, epoch, path="checkpoint.pth"):
    """Save the complete model, including its classifier head."""

    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(model, optimizer, path="checkpoint.pth"):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["epoch"]


def evaluate(model, dataloader, criterion, device):
    """Evaluate with the trained classifier and probability-based AUROC."""

    model.eval()
    total_loss = 0.0
    probabilities, predictions, labels = [], [], []

    with torch.no_grad():
        for original, roi_position, roi, numerical, target in dataloader:
            original = original.to(device)
            roi_position = roi_position.to(device)
            roi = roi.to(device)
            numerical = numerical.to(device)
            target = target.to(device)

            logits = model(original, roi_position, roi, numerical)
            total_loss += criterion(logits, target).item()
            positive_probability = torch.softmax(logits, dim=1)[:, 1]

            probabilities.extend(positive_probability.cpu().numpy())
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels.extend(target.cpu().numpy())

    accuracy = accuracy_score(labels, predictions)
    auc = roc_auc_score(labels, probabilities) if len(set(labels)) > 1 else np.nan
    return total_loss / max(len(dataloader), 1), accuracy, auc
