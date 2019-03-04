# Ensemble
This is a collection of ensemble methods to boost/bag a decision tree.

## Adaboost
### How To Run
```
python3 Ensemble.py ada
```
### Algorithm
Trains 1000 decision stump trees using the ID3 algorithm. Learns a vote and calculates predictions for each example using each tree. Then averages the voted predictions to find a final prediction for each example. The training set prediction error can then be calculated using the final predictions. To calculate the test set prediction error (see how well the learned hypothesis generalizes) it uses the above trees and votes. The trees are used to calculate the test set predictions. Uses these to average the voted predictions to find the final test predictions. The test set prediction error is then just the number of incorrect predictions / number of examples.

### Algorithm In Detail````
1. Create a weight matrix weights to store the weight of each example for each iteration
2. Initialize weights[0] to be 1/m for all m examples (equal weight for each example)
3. for t=1...1000:
#### Training
1. Run id3 algorithm with weights[t] to train a decision stump
2. Use the decision stump to calculate predictions for all examples
3. Calculate the error by summing up the weights[t] where the prediction != label
4. Use the error to calculate the vote for the predictions:
<img src="https://github.com/solosoren/CS5350-MachineLearning/blob/master/EnsembleLearning/Images/Vote.png" height="50">  

5. Calculate the example weights for the next iteration:
<img src="https://github.com/solosoren/CS5350-MachineLearning/blob/master/EnsembleLearning/Images/Weights.png" height="75">  

6. Calculate the final predictions up until this point using the calculated predictions and votes (this allows us to see whether our algorithm is decreasing the error as it should):
<img src="https://github.com/solosoren/CS5350-MachineLearning/blob/master/EnsembleLearning/Images/Final_Predictions.png" height="40">  

#### Testing
1. Using the above tree, calculate the test predictions
2. Calculate the final test predictions up until this point by using the same function as above

### Results
Using the bank dataset described below over 1000 iterations, the algorithm has a training set error of 10.28% and a test set error of 10.76%. 
##### Curve of prediction errors at each iteration
<img src="https://github.com/solosoren/CS5350-MachineLearning/blob/master/EnsembleLearning/Images/AdaBoost_Error.png" width="400">  

## Bagged Decision Trees
### How To Run
```
python3 Ensemble.py bag
```
### Algorithm
Trains T different random samples (with replacement) of 2500 training examples using the ID3 algorithm. After training the decision tree with no depth constraint, it calculates the predictions for the whole training dataset. It uses the sign of the majority prediction to calculate the final prediction. The training set prediction error can then be calculated using the final predictions. To calculate the test set prediction error (see how well the learned hypothesis generalizes) it uses the above trees to calculate the predictions for the test dataset. It again uses the sign of the majority prediction for each test example to calculate the test final predictions. The test set prediction error then is just the number of incorrect predictions / number of examples. The sampling of the dataset is used to create a more reliable and robust hypothesis by reducing the variance.
### Results
Using the bank dataset described below with of 2500 examples randomly sampled (with replacement) over 1000 iterations, the algorithm has a training set error of 5.42% and a test set error of 11.04%. 
##### Curve of prediction errors at each iteration
<img src="https://github.com/solosoren/CS5350-MachineLearning/blob/master/EnsembleLearning/Images/Bagged_Error.png" width="400">  

## Random Forest
### How To Run
```
python3 Ensemble.py forest
```
### Algorithm
For each attribute subset size in {2, 4, 6}, using the Bagged Decision Tree algorithm, it trains T different random samples (with replacement) of 2500 training examples using the ID3 algorithm. The attribute subset size is used to randomly sample a set of attributes (of the specified size) to use at each split of the decision tree in the ID3 algorithm. Just as in the Bagged Decision tree, the sampling of the attributes is used to create an even more more robust hypothesis by reducing the correlation between predictions of trees.
### Results
Using the bank dataset described below with of 2500 examples randomly sampled (with replacement) over 1000 iterations.
Attribute subset size: 2 - training set error: %, test set error: %
Attribute subset size: 4 - training set error: %, test set error: %
Attribute subset size: 6 - training set error: %, test set error: %
##### Curve of prediction errors at each iteration
<img src="https://github.com/solosoren/CS5350-MachineLearning/blob/master/EnsembleLearning/Images/Bagged_Error.png" width="400">  

## Data
The dataset is from [UCI repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing). This dataset contains 16 bank attributes for a given customer, including both numerical and categorical ones. The numerical attributes were altered to be equal to or under the median value, or above the median. Unknown attributes were considered an attribute value. The label is whether the client subscribed to a term deposit. There are 5000 training examples and 5000 test examples.
#### Attributes and Labels
1. Age - (numeric)
2. Job - admin., unknown, unemployed, management, housemaid, entrepreneur, student, blue-collar, self-employed, retired, technician, services
3. Marital - married, divorced, single
4. Education - unknown, secondary, primary, tertiary
5. Default - yes, no
6. Balance - (numeric)
7. Housing - yes, no
8. Loan - yes, no
9. Contact - unknown, telephone, cellular
10. Day - (numeric)
11. Month - jan, feb, mar, ..., nov, dec
12. Duration - (numeric)
13. Campaign - (numeric)
14. Pdays - (numeric) -1 means client was not previously contacted
15. Previous - (numeric)
16. Poutcome - unknown, other, failure, success
17. Label - yes, no
