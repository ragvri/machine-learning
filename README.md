# machine-learning

A repository which contains all of my snippets and projects related to Machine Learning.

* **classical_ml** : Consists of all the basic machine learning algorithms. All of them have been first coded without using **sklearn** in order to understand how the algorithm actually works. Later, they have been coded using sklearn.<br>
**Libraries used** : numpy, sklearn and pandas

* **deep_learning** : Consists of snippets of various deep learning libraries like Tensorflow and Keras. It also includes my projects in deep learning.<br>
**Frameworks used** : Tensorflow, Keras, Theano

## The various projects that I have done are: <br>
* ### **Image Classifier model :** <br>
1) First made my own Image Classifier model using Tensorflow and Keras on a small dataset. Achieved 90% accuracy. Needed to use Image Augumentation and a heavy Dropout in order to achieve this.<br>
2) Applied Transfer Learning on the VGG 16 model by training my model just on the final fully connected layer of VGG16 model. Accuracy > 95%

* ### **Google Dinosaur using CNN and Reinforcement Learning:** <br>
1) Model is still in development phase (It has some bugs). Want to develop a model that is able to play the Google Dinosaur Game on its own.

* ### **Sentiment Analysis of Movie Reviews:** <br>
1) Given any movie review, the model is able to predict whether the review was "positive" or "negative". <br>
2) Accuracy > 80%

## Sources

* [Andrew NG's Machine learning course on coursera](https://www.coursera.org/learn/machine-learning): The most basic course. Everybody does it. Hello world of machine learning.
* [Stanford's CS231n](https://cs231n.github.io/): Introduction to Deep learning and Convolutional Neural Networks<br>
* [Sentdex' playlist for Machine Learning with Python](https://www.youtube.com/watch?v=OGxgnH8y2NM&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v) : Awesome Channel in general for any Python related stuff. This playlist especially focuses on how to use Python for Machine Learning.
* [Jeremy Howard's fast.ai](http://www.fast.ai/) : An awesome MOOC which teaches the different frameworks in Python available for Deep Learning.
* [Andrew NG's new course on Deep learning(paid)](https://www.coursera.org/specializations/deep-learning) : New course being offered by Andrew NG in Deep learning on coursera. It is paid though. However financial help is available like for any other coursera course.

## Installation Tutorials (Just Google it):
* [Tensorflow](https://www.tensorflow.org/install/)
* [Keras](https://keras.io/#installation)
* sklearn, numpy, pandas : Can be installed using pip

**NOTE**: In order to train various deep learning models, it is recommended that you have a GPU which supports CUDA framework to speed up things. 
