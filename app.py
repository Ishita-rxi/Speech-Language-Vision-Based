import cv2
import torch
import torch.nn as nn
import numpy as np
import pyttsx3
import threading
from collections import deque

class StaticCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.fc(self.conv(x))

# Load Settings
ACTIONS = open("labels.txt").read().splitlines()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StaticCNN(len(ACTIONS)).to(device)
model.load_state_dict(torch.load("static_model.pth", map_location=device))
model.eval()

engine = pyttsx3.init()
def speak(text):
    engine.say(text); engine.runAndWait()

# Buffer for temporal stability (15 frames)
prediction_window = deque(maxlen=15)
cap = cv2.VideoCapture(0)
last_spoken = ""

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    roi = frame[50:450, 50:450]
    
    # Preprocess Color Mask
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 30, 60), (20, 255, 255))
    res = cv2.bitwise_and(roi, roi, mask=mask)
    
    img = cv2.resize(res, (64, 64)) / 255.0
    tensor = torch.tensor(img).permute(2, 0, 1).float().unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        conf, pred = torch.max(torch.softmax(output, dim=1), 1)
        
        # If confidence is low, we assume "NONE"
        if conf.item() > 0.80:
            prediction_window.append(ACTIONS[pred.item()])
        else:
            prediction_window.append("NONE")

    # Consensus Logic: Only speak if 12/15 frames agree
    if len(prediction_window) == 15:
        most_common = max(set(prediction_window), key=prediction_window.count)
        if most_common != "NONE" and prediction_window.count(most_common) > 12:
            if most_common != last_spoken:
                threading.Thread(target=speak, args=(most_common,), daemon=True).start()
                last_spoken = most_common
        elif most_common == "NONE":
            last_spoken = ""

    cv2.rectangle(frame, (50, 50), (450, 450), (255, 0, 0), 2)
    cv2.putText(frame, f"SIGN: {last_spoken}", (60, 40), 0, 1, (0, 255, 0), 2)
    cv2.imshow('Sign Translator', frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release(); cv2.destroyAllWindows()