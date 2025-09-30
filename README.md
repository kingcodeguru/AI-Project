# AI-Project

## About the project

[**Full project portfolio**](https://docs.google.com/document/d/1AqE1GmoLSi2JsGDwR_rHOaMSdR18RuVNBslziSr0rLs/edit?usp=sharing)  

Few facts about this project:
* This project was made for a class *"Into to Artificial Intelligence"* (BIU course code 8588981001).
* The goal of the class is to teach us the math behind deep learning. The goal of the project is to see how it reflect on real life uses.
* Our project goal is to **classify images** into finite number of categories using a CNN model - Convolutional Neural Network.
* In this project, we will build a working image classifier using a CNN model, and in the end we will evaluate its qulity.
* More specifically, we will classify fruits and vegtables from 20 different categories.
* The project classifies image that are 128 by 128 and have color.
  If you enter a grayscale image, the model will have a hard time classifing. A different resolution is fine because we made it change it to 128x128.

## Project's files

- `constants.py` - saves the project constants: model filename, resolution, class names, etc.
- `our_model.py` - this python file is responsible of handling the classification of images, given the model.
- `application.py` - the finished product of our model - connects to the camera and show the classified image with the confidence graph.
- [ai-project.ipynb](https://www.kaggle.com/code/lielavraham/ai-project) - all of the code that prepare and trains the model.

## Installing the project and running it

### Download libraries

`pip install pygame tensorflow numpy keras matplotlib`
basically, every library you don't have and the project uses - you will need to download.

### Running the project

* If you have a working camera connected to your computer, I recommend using our nice application we have build. just run
  `python3 application.py`
* If you can't connect your camera, or you just want to test the model on images you have on your computer. You need to run
  `python3 our_model.py`.
  In each iteration, it will ask you a new path of an image to classify. When you're done type *'n'*.
