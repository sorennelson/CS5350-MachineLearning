# Perceptron
This is the standard, voted, and averaged Perceptron algorithm for binary classification.

## Perceptron Loss
<img src="https://github.com/solosoren/CS5350-MachineLearning/blob/master/Perceptron/Images/Loss.png" height="75">    

## Standard Algorithm
*The learning rate is kept a constant 1.*
### Update
The perceptron is mistake driven, thus we only update when our algorithm incorectly classifies an example.
<img src="https://github.com/solosoren/CS5350-MachineLearning/blob/master/Perceptron/Images/Update.png" height="75">  

### Pediction 
The below function works because our labels are {-1, 1}.  
<img src="https://github.com/solosoren/CS5350-MachineLearning/blob/master/Perceptron/Images/Classification.png" height="75">

### Algorithm In Detail
1. Initialize the weight vector to zeros
2. For t=1...10
3. Shuffle the data
4. **For example in training data**:
    1. If the weight vector incorrectly classifies the example (i.e. the prediction doesn't equal the actual label):
    2. Update the weight vector
#### Testing
After training the model, use the learned weight vector to calculate the error on both the train and test sets.
<img src="https://github.com/solosoren/CS5350-MachineLearning/blob/master/Perceptron/Images/Error.png" height="75">  

#### Results
Using the bank note dataset described below, the algorithm was able to get an average test set error of 6%.

#### How To Run
```
python3 Perceptron.py standard
```


## Voted Algorithm
The voted algorithm is similar to the standard except we take into account previous weight vectors. We use the number of correct predictions by that weight vector to weight it. *The learning rate is kept a constant 1.* 
### Update
#### Incorrect Prediction
*Starts a new vote sum*  

<img src="https://github.com/solosoren/CS5350-MachineLearning/blob/master/Perceptron/Images/Voted_incorrect_update.png" height="75">  

#### Correct Prediction
*Updates the number of votes for that weight vector*  

<img src="https://github.com/solosoren/CS5350-MachineLearning/blob/master/Perceptron/Images/Voted_correct_update.png" height="75">  


### Prediction
The below functions work because our labels are {-1, 1}.  
#### For a given weight vector  

<img src="https://github.com/solosoren/CS5350-MachineLearning/blob/master/Perceptron/Images/Classification.png" height="75">  

#### Voted Prediction  

<img src="https://github.com/solosoren/CS5350-MachineLearning/blob/master/Perceptron/Images/Voted_prediction.png" height="75">


### Algorithm In Detail
1. Initialize the weight vector to zeros and m to zero
2. For t=1...10
3. Shuffle the data
4. **For example in training data**:
    1. If the weight vector incorrectly classifies the example (i.e. the prediction doesn't equal the actual label):
    2. Update the weight vector
    3. Otherwise:
    4. Update the number of votes
#### Testing
After training the model, use the learned weight vector to calculate the error on both the train and test sets.
<img src="https://github.com/solosoren/CS5350-MachineLearning/blob/master/Perceptron/Images/Error.png" height="75">  

#### Results
Using the bank note dataset described below, the algorithm was able to get an average test set error of 5%.

#### How To Run
```
python3 Perceptron.py voted
```


## Averaged Algorithm
The averaged algorithm is similar to the voted algorithm except we take into account the scale of the previous weight vectors. This is different from the voted algorithm where we only care about the number of correct predictions by a weight vector. *The learning rate is kept a constant 1.* 
### Update

<img src="https://github.com/solosoren/CS5350-MachineLearning/blob/master/Perceptron/Images/Update.png" height="75">  


### Prediction
The below functions work because our labels are {-1, 1}.  
#### For a given weight vector  

<img src="https://github.com/solosoren/CS5350-MachineLearning/blob/master/Perceptron/Images/Classification.png" height="75">  

#### Average Prediction  

<img src="https://github.com/solosoren/CS5350-MachineLearning/blob/master/Perceptron/Images/Average_prediction.png" height="75">


### Algorithm In Detail
1. Initialize the weight vector to zeros and a to zero
2. For t=1...10
3. Shuffle the data
4. **For example in training data**:
    1. If the weight vector incorrectly classifies the example (i.e. the prediction doesn't equal the actual label):
        1. Update the weight vector
    2. Add the weight vector to a
#### Testing
After training the model, use the learned weight vector to calculate the error on both the train and test sets.
<img src="https://github.com/solosoren/CS5350-MachineLearning/blob/master/Perceptron/Images/Error.png" height="75">  

#### Results
Using the bank note dataset described below, the algorithm was able to get an average test set error of 5%.

#### How To Run
```
python3 Perceptron.py avg
```


## Data
The dataset is from [UCI repository](https://archive.ics.uci.edu/ml/datasets/banknote+authentication). This dataset contains 4 bank note features and is used to calculate the whether a bank note was forged or genuine. There are 872 training examples and 500 test examples.
#### Attributes
1. Variance of Wavelet Transformed image (continuous) 
2. Skewness of Wavelet Transformed image (continuous) 
3. Curtosis of Wavelet Transformed image (continuous) 
4. Entropy of image (continuous) 
