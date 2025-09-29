# AI-Project

## About the project

Few facts about this project:
* This project was made for a class "Into to Artificial Intelligence" (BIU course code 8588981001).
* The goal of the class is to teach us the math behind deep learning. The goal of the project is to see how it reflect on real life uses.
* Our project goal is to classify images into finite number of categories using a CNN model - Convolutional Neural Network.
* In this project, we will build a working image classifier using a CNN model, and in the end we will evaluate its qulity.

## Project's files

- `keys.py` – Contains the list of categories to include in the model.
- `loadData.py` – Downloads and prepares data from the web.
- `modelTraining.py` – Trains the CNN and saves the model as `QuickDraw_CNN_35_classes.h5`.
- `QuickDraw_CNN_35_classes.h5` – The trained CNN model (output file).
- `doodle_app.py` – A simple Pygame-based app to draw and classify doodles using the trained model. This file must be in the same directory as the saved model.
