import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(model, optimizer, epoch, path="checkpoint.pth"):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }, path)

def load_checkpoint(model, optimizer, path="checkpoint.pth"):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["epoch"]

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, preds, labels = 0, [], []

    with torch.no_grad():
        for i1, i2, i3, v, y in dataloader:
            i1, i2, i3, v, y = i1.to(device), i2.to(device), i3.to(device), v.to(device), y.to(device)
            out = model(i1, i2, i3, v)
            logits = nn.Linear(out.size(1), 2).to(device)(out)  # 假设二分类
            loss = criterion(logits, y)
            total_loss += loss.item()

            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels.extend(y.cpu().numpy())

    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, preds)

    return total_loss / len(dataloader), acc, auc
