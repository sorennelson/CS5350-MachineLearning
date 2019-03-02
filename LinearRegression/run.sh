#!/bin/bash

echo Batch Gradient Descent
python3 LinearRegression.py bgd
echo ____________________________
echo Stochastic Gradient Descent
python3 LinearRegression.py sgd
