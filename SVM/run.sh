#!/bin/bash

echo SVM Primal:
python3 SVM.py primal
echo ____________________________
echo SVM Primal 2:
python3 SVM.py primal2
echo ____________________________
echo SVM Linear Dual:
python3 SVM.py linear_dual
echo ____________________________
echo SVM Gaussian Dual:
python3 SVM.py gaussian_dual
