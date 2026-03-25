# Real-Time Sign Language to Speech Translator

This project is a computer vision application that recognizes hand gestures (Signs) through a webcam and converts them into audible speech. It uses a custom-built Convolutional Neural Network (CNN) and a temporal buffer to ensure stable, real-time translation.

## Key Features
* **Custom CNN Architecture:** A 2-layer Convolutional Neural Network built from scratch.
* **Color-Based Skin Masking:** Uses the HSV color space to isolate the hand and remove background noise.
* **Temporal Smoothing:** A 15-frame sliding window ensures the AI only speaks when a sign is held steady.
* **Automatic Speech:** Uses the `pyttsx3` library to announce signs instantly.

## Project Structure
* `collect_data.py`: Script to capture training images (100 per class).
* `train.py`: Trains the AI and saves the model as `static_model.pth`.
* `app.py`: The main application for real-time sign-to-speech translation.
* `labels.txt`: Automatically generated list of signs the model knows.

## Installation
Ensure you have Python installed, then run:
```bash
pip install opencv-python torch torchvision numpy pyttsx3
