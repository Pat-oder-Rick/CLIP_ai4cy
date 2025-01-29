import os
import torch
import clip
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# GPU oder CPU verwenden
device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP-Modell laden
model, preprocess = clip.load("ViT-B/32", device=device)

# CSV-Datei mit Bildpfaden und Textlabels einlesen
csv_filename = "image_labels.csv"
df = pd.read_csv(csv_filename)  # Erwartet Spalten: "image_path", "text_label"

# Custom Dataset für CLIP
class CLIPDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.data = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 0]
        label_text = self.data.iloc[idx, 1]

        # Bild laden und vorverarbeiten
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # Text tokenisieren
        text = clip.tokenize([label_text])[0]

        return image, text

# Dataset & DataLoader erstellen
dataset = CLIPDataset(df, preprocess)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Optimizer & Verlustfunktion
optimizer = optim.Adam(model.parameters(), lr=1e-5)
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

# Fine-Tuning starten
num_epochs = 5
for epoch in range(num_epochs):
    total_loss = 0
    for images, texts in dataloader:
        images, texts = images.to(device), texts.to(device)

        # Features berechnen
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)

        # Normalisierung (Kein In-Place-Fehler!)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Ähnlichkeit berechnen
        logits_per_image = image_features @ text_features.T
        logits_per_text = logits_per_image.T

        # Labels für Cross-Entropy (Identitätsmatrix)
        ground_truth = torch.arange(len(images), dtype=torch.long, device=device)

        # Verluste berechnen
        loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
        total_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss:.4f}")

# Fine-Tuned Modell speichern
torch.save(model.state_dict(), "clip_finetuned.pth")
print("Fine-Tuning abgeschlossen. Modell gespeichert als clip_finetuned.pth")
