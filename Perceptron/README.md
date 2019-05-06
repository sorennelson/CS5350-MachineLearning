# Perceptron
This is the standard, voted, and averaged Perceptron algorithm for binary classification.

## Perceptron Loss
<img src="https://github.com/solosoren/CS5350-MachineLearning/blob/master/Perceptron/Images/Loss.png" height="75">  

## Update
The perceptron is mistake driven, thus we only update when our algorithm incorectly classifies an example
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
    1. If the weight vector incorrectly classifies the example:
    2. Update the weight vector
#### Testing
Error = actual

### Results
Using the concrete dataset described below, the algorithm converged at a learning rate of 0.0078125. The final test cost was 31.96%.

## Stochastic Gradient Descent
Uses the gradient of the Least Mean Squares loss function to minimize the cost. The algorithm is said to converge when ||new weight vector - prev weight vector|| is less than 0.0001. The learning rate starts at 1 and is halved every 100 iterations.
### How To Run
```
python3 LinearRegression.py sgd
```

### Algorithm In Detail
1. Initialize the learning rate to 1
2. While ||new weight vector - prev weight vector|| > 0.0001:
3. Create a weight matrix weights to store the weight of each example for each iteration and initialize it to 0.
4. For t=1...100
5. Shuffle the data
6. **For example in training data**:
    1. Compute the gradient of the cost function with respect to the weight vector **for a single example**:
        <img src="https://github.com/solosoren/CS5350-MachineLearning/blob/master/LinearRegression/Images/Gradient.png" height="75">  
    2. Update weights[iter + 1, feature] = weights[iter, feature] - (learning rate * gradient)
7. Compute the LMS cost of the train data to ensure the algorithm is decreasing.
8. If ||new weight vector - prev weight vector|| < 0.0001, the algorithm has converged. The final weight vector and the learning rate are the last ones to be updated.
9. Otherwise: half the learning rate
#### Testing
1. Use the learned weight vector and learning rate to calculate the cost of the weights on the test data.

### Results
Using the concrete dataset described below, the algorithm converged at a learning rate of 0.0009765625. The final test cost was 29.17%.

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
