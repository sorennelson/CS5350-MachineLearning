# Decision Tree
This is an ID3 Decision Tree learning algorithm for two evaluation tasks. There are three types of purity used: Entropy, Majority Error, and Gini Index.
## Car
The dataset is from the [UCI repository](https://archive.ics.uci.edu/ml/datasets/car+evaluation). In this task, we have 6 car attributes, and the label is the evaluation of the car. There are 1000 training examples and 728 test examples.
#### Attributes and Labels
1. Buying - Very High, High, Medium, Low
2. Maintenance - Very High, High, Medium, Low
3. Number of Doors - 2, 3, 4, 5+
4. Persons - 2, 4, More than 4
5. Lug Boot - Big, Medium, Small
6. Safety - High, Medium, Small
7. Label - Unacceptable, Acceptable, Good, Very Good
#### How To Run
```
python3 ID3.py car ig
```
To change the purity, replace ig with any of the following:
- Information Gain: ig
- Majority Error: me
- Gini Index: gi  

If you would like to add a depth constraint to the tree, add the depth to the end of the line.  
Ex: `python3 ID3.py car ig 2`


## Bank
The dataset is from [UCI repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing). This dataset contains 16 bank attributes for a given customer, including both numerical and categorical ones. The numerical attributes were altered to be equal to or under the median value, or above the median. Any unknown attributes were altered to be the majority attribute. The label is whether the client subscribed to a term deposit. There are 5000 training examples and 5000 test examples.
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
#### How To Run
```
python3 ID3.py bank ig
```
As above, to change the purity, replace ig with any of the following:
- Information Gain: ig
- Majority Error: me
- Gini Index: gi  

If you would like to add a depth constraint to the tree, add the depth to the end of the line.  
Ex: `python3 ID3.py bank ig 2`
