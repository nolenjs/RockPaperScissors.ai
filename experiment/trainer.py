# -*- coding: utf-8 -*-
"""Copy of AndroidML-HandGestureRecognitionModelTraining.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1SVqtMoB0HBCS6OuDo8877cWMIgZyAdy8

##### Copyright 2023 The MediaPipe Authors. All Rights Reserved.
"""
"""@title Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

#TODO below are operations to get the training data for the basic rock?paper/scissors gestures
# !wget -q -O rps.zip https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/rps_data_sample.zip
# !unzip -qq rps.zip
# !pip install -q mediapipe-model-maker

import matplotlib.pyplot as plt
import os

NUM_EXAMPLES = 5
IMAGES_PATH = "rps_data_sample"

# Show examples of gestures the model will be trained on
labels = []
for i in os.listdir(IMAGES_PATH):
  if os.path.isdir(os.path.join(IMAGES_PATH, i)):
    labels.append(i)

# Show the images.
for label in labels:
  label_dir = os.path.join(IMAGES_PATH, label)
  example_filenames = os.listdir(label_dir)[:NUM_EXAMPLES]
  fig, axs = plt.subplots(1, NUM_EXAMPLES, figsize=(10,2))
  for i in range(NUM_EXAMPLES):
    axs[i].imshow(plt.imread(os.path.join(label_dir, example_filenames[i])))
    axs[i].get_xaxis().set_visible(False)
    axs[i].get_yaxis().set_visible(False)
  fig.suptitle(f'Showing {NUM_EXAMPLES} examples for {label}')

plt.show()

# """## Making a New Model


# """

# Import the necessary modules.
from mediapipe_model_maker import gesture_recognizer

# Load the rock-paper-scissor image archive.
data = gesture_recognizer.Dataset.from_folder(
    dirname=IMAGES_PATH,
    hparams=gesture_recognizer.HandDataPreprocessingParams()
)

# Split the archive into training, validation and test dataset.
train_data, rest_data = data.split(0.8)
validation_data, test_data = rest_data.split(0.5)

# Train the model
hparams = gesture_recognizer.HParams(export_dir="rock_paper_scissors_model")
options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
model = gesture_recognizer.GestureRecognizer.create(
    train_data=train_data,
    validation_data=validation_data,
    options=options
)

loss, acc = model.evaluate(test_data, batch_size=1)
print(f"Test loss:{loss}, Test accuracy:{acc}")

# Export the model bundle.
model.export_model()

# Rename the file to be more descriptive.
# !mv rock_paper_scissors_model/gesture_recognizer.task rock_paper_scissors.task

# # Imports neccessary modules.
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Ensure the model file exists
model_path = os.path.abspath("rock_paper_scissors.task")
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

# Create a GestureRecognizer object
recognizer = vision.GestureRecognizer.create_from_model_path(model_path)

# Ensure the image file exists
image_path = './photo.jpg'
if not os.path.isfile(image_path):
    raise FileNotFoundError(f"Image file not found: {image_path}")

# Load the input image
image = mp.Image.create_from_file(image_path)

# Run gesture recognition
recognition_result = recognizer.recognize(image)

# Display the most likely gesture
top_gesture = recognition_result.gestures[0][0]
print(f"Gesture recognized: {top_gesture.category_name} ({top_gesture.score})")