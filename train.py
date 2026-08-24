"""Minimal training utilities for the principle-level CMLM framework."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import MultiModalModel
from utils import get_device, save_checkpoint


DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-3


class FocalLoss(nn.Module):
    """Binary/multiclass focal loss for imbalanced MLM classification."""

    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        cross_entropy = F.cross_entropy(logits, targets, reduction="none")
        probability_of_true_class = torch.exp(-cross_entropy)
        loss = (1.0 - probability_of_true_class).pow(self.gamma) * cross_entropy

        if self.alpha is not None:
            class_weight = torch.where(
                targets == 1,
                torch.as_tensor(self.alpha, device=logits.device),
                torch.as_tensor(1.0 - self.alpha, device=logits.device),
            )
            loss = class_weight * loss
        return loss.mean()


def build_training_components(model, learning_rate=DEFAULT_LEARNING_RATE):
    """Use the paper-aligned optimizer and loss defaults."""

    criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return criterion, optimizer


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train one epoch using a caller-provided multimodal DataLoader."""

    model.train()
    total_loss = 0.0
    for original, roi_position, roi, numerical, labels in dataloader:
        original = original.to(device)
        roi_position = roi_position.to(device)
        roi = roi.to(device)
        numerical = numerical.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(original, roi_position, roi, numerical)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


def fit(model, dataloader, epochs=100, checkpoint_every=20):
    """A compact training loop; full cross-validation is intentionally omitted."""

    device = get_device()
    model = model.to(device)
    criterion, optimizer = build_training_components(model)

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(
            model, dataloader, criterion, optimizer, device
        )
        print(f"Epoch [{epoch}/{epochs}] Loss: {loss:.4f}")
        if checkpoint_every and epoch % checkpoint_every == 0:
            save_checkpoint(model, optimizer, epoch)
    return model


def main():
    model = MultiModalModel()
    criterion, optimizer = build_training_components(model)
    print(model.__class__.__name__)
    print(f"Suggested batch size: {DEFAULT_BATCH_SIZE}")
    print(f"Optimizer: {optimizer.__class__.__name__}")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"Loss: {criterion.__class__.__name__}")
    print("Provide a project-specific DataLoader and call fit(model, dataloader).")


if __name__ == "__main__":
    main()
