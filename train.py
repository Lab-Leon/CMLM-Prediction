import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import MultiModalModel
from dataset import MultiModalDataset
from utils import get_device, save_checkpoint, evaluate

def main():
    # Args
    csv_file = "data/train.csv"
    img_dir = "data/images"
    batch_size = 16
    lr = 1e-4
    epochs = 100

    device = get_device()
    print("Using device:", device)

    # data
    train_dataset = MultiModalDataset(csv_file, img_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # model
    model = MultiModalModel().to(device)

    # classifier head
    classifier = nn.Linear(512, 2).to(device)

    # loss function & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=lr)

    # training
    for epoch in range(1, epochs + 1):
        model.train()
        classifier.train()
        total_loss = 0

        for i1, i2, i3, v, y in train_loader:
            i1, i2, i3, v, y = i1.to(device), i2.to(device), i3.to(device), v.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(i1, i2, i3, v)  # B x 512
            logits = classifier(out)    # B x 2
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch}/{epochs}] Loss: {total_loss/len(train_loader):.4f}")

        if epoch % 20 == 0:
            save_checkpoint(model, optimizer, epoch)

if __name__ == "__main__":
    main()
