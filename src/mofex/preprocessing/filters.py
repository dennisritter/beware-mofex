"""This module contains 3-D transformation function as well as geometric calculations."""
import math
import numpy as np
import sklearn.preprocessing as preprocessing



def filter_outliers_iqr(data: 'np.ndarray', factor: float = 1.5):
    q1 = np.quantile(data, 0.25)
    q3 = np.quantile(data, 0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return data[np.where((data > lower) & (data < upper))]