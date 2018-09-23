# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 08:53:41 2018

@author: abdar
"""

from sklearn import cross_validation
from sklearn import datasets
import pandas as pd
import numpy as np

iris = datasets.load_iris()
type(iris)