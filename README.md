# Trady, your personal trading assistant!

## Table of Content
  * [Overview](#overview)
  * [Demo](#demo)
  * [Motivation](#motivation)
  * [Technical Aspects](#technical-aspects)
  * [Results](#result)
  * [Technologies](#technologies)
  * [To Do](#to-do)
  * [File List](#file-list)
  * [Credits](#credits)


  
## Overview <a name="overview" />
In this project I built Trady. It is a chat-bot on Telegram that every day
1. Loads information about a portfolio of stocks and liquid dollars
2. Downloads updated stock market prices
3. Uses an LSTM model to predict the future price of the stock of a selected company after a selected amount of time
4. Computes the long term moving average and the short term moving average of the price of the stock of the selected company
5. Using these data, decides wheter to wait or to buy/sell stocks
6. Locally saves the new composition of the portfolio

## Demo <a name="demo" />


## Motivation <a name="motivation" />
I have always been interested in stock market because it is one of the most interesting field to apply machine learning on. It offers a huge amount of free data, and very complex relations to infer. For this reason, non standard models and innovative idea are needed. Moreover, the better your models are, the more money you can directly earn.

Moreover, I wanted to build something easy to use for non technical users. Because of that, I provided the outputs as Telegram messages and scheduled script to be automatically executed once a day. 

## Technical Aspect <a name="technical-aspects" />
The main issue of this project was selection of features. I used a neural network model where feature selection is tipically non necessary. But in this case I had a huge amount of historical data so the model would have been prone to overfitting. Moreover, a lot of computational resources would have been needed to train the model on all data.



## Result <a name="result" />
The confusion matrixes obtained from the predictions of the 4 analyzed models are the following:

<img src="https://user-images.githubusercontent.com/29163695/122113334-3fd3ff00-ce22-11eb-80e2-741cc13019e5.png" height="400">
<img src="https://user-images.githubusercontent.com/29163695/122112158-d0114480-ce20-11eb-85b8-47b4912d23ca.png" height="400">

<img src="https://user-images.githubusercontent.com/29163695/122112221-e0292400-ce20-11eb-8703-a550ec62404e.png" height="400">
<img src="https://user-images.githubusercontent.com/29163695/122112292-f0d99a00-ce20-11eb-9e06-88a16a05e469.png" height="400">

I noticed that:
1. The classification model provides a higher accuracy even if some predictions are heavily wrong (i.e. a lot of "5" in place of "1" or vice versa). The MAE is the highest because classes are independent and sorting information is not used.
2. Even if the regression models have the same complexity of the classification model, they do not provide so high accuracy. However, thanks to the optimization function used in the training phase, prediction errors are often lower.
3. Both classification and regression models, even if they are very simple, allow reaching the same precision of a human being. It is a proof of the potential of LSTM layers and convolutional layers in neural networks.

## Technologies <a name="technologies" />
I used *nltk* library for text preprocessing, *Tensorflow* for model building and *AWS Ground Truth* to make data classified by human beings.

<img src="https://user-images.githubusercontent.com/29163695/122077900-726b0100-cdfc-11eb-90d4-9e45d3a3f53f.png" height="200">
<img src="https://user-images.githubusercontent.com/29163695/122078058-94fd1a00-cdfc-11eb-93d4-fe4159a0675a.png" height="200">
<img src="https://user-images.githubusercontent.com/29163695/122078294-c675e580-cdfc-11eb-95d6-bdd137cf2847.png" height="200">


## To Do <a name="to-do" />
* Since the accuracy of the model is similar to the accuracy of a human classifier, we may try to create a model only for reviews of a particular kind of object to obtain better performance.
* We may analyze full reviews and see whether performances improve or not.
* We may train the model on the full dataset (the number of reviews to load has been limited to speed up model training)

## File List <a name="file-list" />
* **main.py** Data loading, data preprocessing, model training and model evaluation.
* **my_utils.py** Useful functions to load data and plot confusion matrixes

## Credits <a name="credits" />
* [DTrimarchi10](https://github.com/DTrimarchi10) - Thanks for the confusion_matrix function I took inspiration from
* [Hands-On Machine Learning with Scikit-Learn, Keras & Tensorflow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) - Thansk of the authors of the book for the advice about model building
* [Xiang Zhang](https://figshare.com/articles/dataset/Amazon_Reviews_Full/13232537/1) - Thanks for the complete dataset of Amazon reviews
