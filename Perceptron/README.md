# Perceptron
This is the standard, voted, and averaged Perceptron algorithm for binary classification.

## Perceptron Loss
<img src="https://github.com/solosoren/CS5350-MachineLearning/blob/master/Perceptron/Images/Loss.png" height="75">  

## Update
The perceptron is mistake driven, thus we only update when our algorithm incorectly classifies an example.
<img src="https://github.com/solosoren/CS5350-MachineLearning/blob/master/Perceptron/Images/Update.png" height="75">  

## Classification
<img src="https://github.com/solosoren/CS5350-MachineLearning/blob/master/Perceptron/Images/Classification.png" height="75">  

## Standard Algorithm
The learning rate is kept a constant 1.

### How To Run
```
python3 Perceptron.py standard
```
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

### Results
Using the bank dataset described below, the algorithm was able to get an average test set error of 6%.



## Data
The dataset is from [UCI repository](https://archive.ics.uci.edu/ml/datasets/Concrete+Slump+Test). This dataset contains 7 concrete features and is used to calculate the SLUMP of the concrete. There are 53 training examples and 50 test examples.
#### Attributes and Labels
1. Cement
2. Slag
3. Fly ash
4. Water
5. SP
6. Coarse Aggr
7. Fine Aggr
