import os
import csv

# Verzeichnis, das die Bildordner enthält
dataset_dir = "./bitmaps_textlabel"  # Passe diesen Pfad an

# CSV-Datei, in die die Zuordnungen gespeichert werden
csv_filename = "image_labels.csv"

# Liste für CSV-Daten
csv_data = []

# Durchlaufe alle Ordner in dataset_dir
for folder in os.listdir(dataset_dir):
    folder_path = os.path.join(dataset_dir, folder)

    # Prüfe, ob es ein Ordner ist
    if os.path.isdir(folder_path):
        # Durchlaufe alle Bilder im Ordner
        for image_file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_file)

            # Prüfe, ob es eine Bilddatei ist (optional: nur bestimmte Formate zulassen)
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                csv_data.append([image_path, folder])  # Speichere Bildpfad + Label

# CSV-Datei schreiben
with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["image_path", "text_label"])  # Spaltenüberschriften
    writer.writerows(csv_data)

print(f"CSV-Datei '{csv_filename}' mit {len(csv_data)} Einträgen erstellt.")