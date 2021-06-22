# Trady, your personal trading assistant!

<img src="https://user-images.githubusercontent.com/29163695/122943420-4d840a00-d377-11eb-8bd4-d75d55a9aa28.png" height="150">


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
In this project I built Trady. He is a chat-bot on Telegram that every day
1. Loads information about a fake portfolio of stocks and liquid dollars
2. Downloads updated stock market prices
3. Uses an LSTM model to predict the future price of the stock of a selected company after a selected amount of time
4. Computes the long term moving average and the short term moving average of the price of the stock of the selected company
5. Using these data, decides wheter to wait or to buy/sell stocks
6. Locally saves the new composition of the fake portfolio

Trady suggests only long-term investments so it analyze only daily closure prices. His strategy is based on his predictions and moving averages technique.

## Demo <a name="demo" />
You can find Trady in action in the following video demo:

[![Everything Is AWESOME](https://user-images.githubusercontent.com/29163695/123008937-1e43bc00-d3bc-11eb-98ee-744287abbf41.png)](https://user-images.githubusercontent.com/29163695/123006138-8c39b480-d3b7-11eb-89e4-f5a49cf1b247.mp4)

## Motivation <a name="motivation" />
I have always been interested in stock market because it is one of the most interesting field to apply machine learning on. It offers a huge amount of free data, and very complex relations to infer. For this reason, non standard models and innovative idea are needed. Moreover, the better your models are, the more money you can directly earn.

Moreover, I wanted to build something easy to use for non technical users. Because of that, I provided the outputs as Telegram messages and scheduled script to be automatically executed once a day. 

## Technical Aspects <a name="technical-aspects" />
The main issue of this project was the selection of features. I used a neural network model where feature selection is tipically non necessary. But in this case I had a huge amount of historical data so the model would have been prone to overfitting. Moreover, a lot of computational resources would have been needed to train the model with all data. My approach consisted on computing the correlation between historical stock prices of a particular company and the historical stock prices of all of the other ones. Only the most correlated stocks prices are then been considered as features for model. 

Every script has been fully parametrized. It allows to easily improve algorithm and scale it up to monitor and predict the prices of the stocks of more companies at the same time and automatically implement a more satisfacing trading strategy.

## Results <a name="result" />

My first target was to test the performance of the model and in particular the hypotetical long-term earnings of investments suggested by Trady. I therefore built a model to predict only the value of Tesla stock prices (TSLA) in 20 working days. The model was trained only with the most correlated features. The following are the trends of loss functions in training phase:

<img src="https://user-images.githubusercontent.com/29163695/122837527-84fda280-d2f4-11eb-9173-6aac0217c509.png" height="400">

The MAE of the prodiction on the validation set is 46.3. By considering that Tesla stock prices is about 600 dollars and that the average of the stock price daily variation is 19.1 dollars, the performances of the model are pretty good. Moreover, on one hand the feature selection made through the correlations analysis allowed to reduce overfitting. On the other hand, the use of time shifted features and LSTM layers in NN allows to decrease underfitting.

I noticed that the final value of loss function was strongly dependant on the choice of the inizial values of parameters of neural network. It was due to the fact that the loss function to minimize was very complex, so the optimizer got stuck into local minima. I tackled the problem by setting a decreasing learning rate and by training model more time with random choices of initial values of parameters.


I have no feedback about the long-term earnings based on Trady strategy yet. I will update this section in 3-4 months with a complete review.


## Technologies <a name="technologies" />
I used *Telegram API* to create the chat-bot,  *Tensorflow* for model building and *Alpha Vantage API* to download updated data about stock prices.

<img src="https://user-images.githubusercontent.com/29163695/123009731-92329400-d3bd-11eb-8959-b5b4484f724f.png" height="200">
<img src="https://user-images.githubusercontent.com/29163695/122078058-94fd1a00-cdfc-11eb-93d4-fe4159a0675a.png" height="200">
<img src="https://user-images.githubusercontent.com/29163695/123010555-20f3e080-d3bf-11eb-9d63-f737be36d98b.png" height="200">


Moreover, I used the software *Hitfilm Express* to create the previous video demo.

<img src="https://user-images.githubusercontent.com/29163695/123010341-b8a4ff00-d3be-11eb-8db5-c5729f7ea61f.png" height="200">


## To Do <a name="to-do" />
* After some months of test, integrate it with a true Broker to make Trady work with real money.
* Monitor the stocks of more companies at the same time.
* Store data downloaded by stock API to avoid downloading old data more times and limit API costs.
* Deploy the application on a cloud server i.e. GCP, AWS to easily schedule it, and save models. By using more computational power it would be also possible to create a model for each company.

## File List <a name="file-list" />
* **download_hist_data.py** Data loading, data preprocessing, model training and model evaluation.
* **data cleaning.py** Useful functions to load data and plot confusion matrixes
* **corr_table_generator.py** Useful functions to load data and plot confusion matrixes
* **features_selection.py** Useful functions to load data and plot confusion matrixes
* **training.py** Useful functions to load data and plot confusion matrixes
* **autotrading.py** Useful functions to load data and plot confusion matrixes
* **my_utils.py** Useful functions to load data and plot confusion matrixes

## Credits <a name="credits" />
* [Studio Envato](https://studio.envato.com/explore/caricatures-cartoon-design/133-mascot-and-character-design?per=1000) - Thanks for the avatar of Trady
* [Investopedia](https://www.investopedia.com/articles/active-trading/052014/how-use-moving-average-buy-stocks.asp) - Thanks for explenation about moving averages technique

