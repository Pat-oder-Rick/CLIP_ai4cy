import torch
import clip
from PIL import Image
import pandas as pd

# Gerät wählen (GPU oder CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP-Modell laden
model, preprocess = clip.load("ViT-B/32", device=device)

# Das feingetunte Modell laden
model.load_state_dict(torch.load("clip_finetuned.pth", map_location=device))
model.eval()  # Modell in den Evaluierungsmodus versetzen

# Liste mit möglichen Textlabels laden (aus der gleichen CSV wie beim Training)
csv_filename = "image_labels.csv"
df = pd.read_csv(csv_filename)  # Erwartet Spalten: "image_path", "text_label"
text_labels = df["text_label"].tolist()

# Texte für CLIP tokenisieren
text_inputs = clip.tokenize(text_labels).to(device)

# **🔹 Funktion zum Erkennen eines neuen Bildes**
def predict_image(image_path):
    """Verwendet das feingetunte CLIP-Modell, um das Bild zu klassifizieren."""
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        # Bild- und Text-Features berechnen
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_inputs)

        # Normalisieren
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Ähnlichkeiten berechnen
        similarity = (image_features @ text_features.T).cpu().numpy()
        best_match = similarity.argmax()  # Index mit höchstem Wert

        return text_labels[best_match], similarity[0][best_match]

# **🔹 Beispielaufruf:**
rdp_image = "bitmaps\\Cache0000.bin_0016.bmp"  # Ersetze mit deinem Pfad
predicted_label, confidence = predict_image(rdp_image)

print(f"Vorhersage: {predicted_label} (Konfidenz: {confidence:.2f})")
