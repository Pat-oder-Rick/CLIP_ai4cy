from xml.etree.ElementTree import tostring

import torch
import clip
from PIL import Image
import numpy as np
import os
from os import listdir

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device = device)

textLabels = []

folder_dir = "./images"
for images in os.listdir(folder_dir):


    testImagePath = "./images/" + images
    image = preprocess(Image.open(testImagePath)).unsqueeze(0).to(device)

    text = clip.tokenize(textLabels).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)

        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        probs = probs[0]
        answer = np.argmax(probs)
        test = textLabels[answer]

        print("for " + images + "Predicted : " + test)